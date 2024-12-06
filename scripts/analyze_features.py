import sys
import os
import torch
import numpy as np
from bisect import insort_right
from tqdm import tqdm
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import yaml
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import TopKSparseAutoencoder
from src.data.activation_extractor import ActivationDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze feature activations in a trained autoencoder")
    parser.add_argument("--config", default='config/default_config.yaml', type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", default='checkpoints/20241202_175534/best_model.pt', type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--output-dir", default='output/analysis/', type=str, required=True, help="Directory to save analysis results")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to analyze")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top/bottom examples to keep per feature")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--features", type=str, default="all", 
                       help="Comma-separated list of feature indices to analyze, or 'all'")
    return parser.parse_args()

def analyze_feature_activations(
    dataloader: DataLoader,
    autoencoder: TopKSparseAutoencoder,
    tokenizer,
    n_samples: int = 1000,
    top_k: int = 5,
    feature_indices: list = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """Analyze activations of autoencoder features on a dataset"""
    autoencoder.eval()
    
    feature_stats = {}
    samples_seen = 0
    
    # Initialize storage for each feature
    n_features = autoencoder.latent_dim
    if feature_indices is None:
        feature_indices = range(n_features)
    
    for feat_idx in feature_indices:
        feature_stats[feat_idx] = {
            'top_activations': [],  # Will store (activation_value, input_ids, position)
            'bottom_activations': [],
            'total_activations': 0,
            'sum_activations': 0.0,
            'sum_squared': 0.0
        }
    
    with torch.no_grad():
        pbar = tqdm(total=n_samples, desc="Processing samples")
        
        for batch in dataloader:
            if samples_seen >= n_samples:
                break
                
            activations = batch['activations'].to(device)
            input_ids = batch['input_ids']
            
            # Get autoencoder latents
            _, latents = autoencoder(activations)
            
            # Update batch statistics for each feature
            for feat_idx in feature_indices:
                feat_activations = latents[..., feat_idx]
                
                # Find top activations in this batch
                top_vals, top_indices = torch.topk(feat_activations.view(-1), min(top_k, feat_activations.numel()))
                bottom_vals, bottom_indices = torch.topk(-feat_activations.view(-1), min(top_k, feat_activations.numel()))
                bottom_vals = -bottom_vals
                
                # Convert flat indices to batch and position indices
                batch_size = feat_activations.size(0)
                if len(feat_activations.shape) > 1:
                    seq_len = feat_activations.size(1)
                else:
                    seq_len = 1
                    
                top_batch_idx = top_indices // seq_len
                top_pos_idx = top_indices % seq_len
                bottom_batch_idx = bottom_indices // seq_len
                bottom_pos_idx = bottom_indices % seq_len
                
                # Store examples
                for val, b_idx, p_idx in zip(top_vals, top_batch_idx, top_pos_idx):
                    example = {
                        'activation': val.item(),
                        'input_ids': input_ids[b_idx].tolist(),
                        'position': p_idx.item()
                    }
                    insort_right(feature_stats[feat_idx]['top_activations'], 
                               example, 
                               key=lambda x: -x['activation'])
                    
                for val, b_idx, p_idx in zip(bottom_vals, bottom_batch_idx, bottom_pos_idx):
                    example = {
                        'activation': val.item(),
                        'input_ids': input_ids[b_idx].tolist(),
                        'position': p_idx.item()
                    }
                    insort_right(feature_stats[feat_idx]['bottom_activations'], 
                               example, 
                               key=lambda x: x['activation'])
                
                # Keep only top_k examples
                feature_stats[feat_idx]['top_activations'] = feature_stats[feat_idx]['top_activations'][:top_k]
                feature_stats[feat_idx]['bottom_activations'] = feature_stats[feat_idx]['bottom_activations'][:top_k]
                
                # Update statistics
                feat_activations = feat_activations.view(-1)
                feature_stats[feat_idx]['total_activations'] += (feat_activations > 0).sum().item()
                feature_stats[feat_idx]['sum_activations'] += feat_activations.sum().item()
                feature_stats[feat_idx]['sum_squared'] += (feat_activations ** 2).sum().item()
            
            samples_seen += activations.size(0)
            pbar.update(activations.size(0))
            
        pbar.close()
    
    # Calculate final statistics
    for feat_idx in feature_indices:
        stats = feature_stats[feat_idx]
        n = samples_seen * seq_len  # total positions seen
        stats['mean_activation'] = stats['sum_activations'] / n
        stats['activation_rate'] = stats['total_activations'] / n
        variance = (stats['sum_squared'] / n) - (stats['mean_activation'] ** 2)
        stats['std_activation'] = np.sqrt(variance) if variance > 0 else 0
    
    return feature_stats

def save_feature_analysis(
    feature_stats: dict,
    tokenizer,
    output_dir: Path,
    context_tokens: int = 10
):
    """Save analysis results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full statistics as JSON
    stats_file = output_dir / "feature_stats.json"
    
    # Convert stats to JSON-serializable format
    json_stats = {}
    for feat_idx, stats in feature_stats.items():
        json_stats[str(feat_idx)] = {
            'activation_rate': stats['activation_rate'],
            'mean_activation': stats['mean_activation'],
            'std_activation': stats['std_activation'],
            'top_examples': [],
            'bottom_examples': []
        }
        
        # Process top examples
        for example in stats['top_activations']:
            input_ids = example['input_ids']
            pos = example['position']
            
            # Get context around the activation
            start = max(0, pos - context_tokens)
            end = min(len(input_ids), pos + context_tokens + 1)
            context = input_ids[start:end]
            
            json_stats[str(feat_idx)]['top_examples'].append({
                'activation': example['activation'],
                'position': pos,
                'context': tokenizer.decode(context)
            })
            
        # Process bottom examples
        for example in stats['bottom_activations']:
            input_ids = example['input_ids']
            pos = example['position']
            
            start = max(0, pos - context_tokens)
            end = min(len(input_ids), pos + context_tokens + 1)
            context = input_ids[start:end]
            
            json_stats[str(feat_idx)]['bottom_examples'].append({
                'activation': example['activation'],
                'position': pos,
                'context': tokenizer.decode(context)
            })
    
    with open(stats_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    # Save human-readable summary
    summary_file = output_dir / "feature_summary.txt"
    with open(summary_file, 'w') as f:
        for feat_idx in sorted(feature_stats.keys(), key=int):
            stats = feature_stats[feat_idx]
            
            f.write(f"\nFeature {feat_idx}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Activation Rate: {stats['activation_rate']:.3%}\n")
            f.write(f"Mean Activation: {stats['mean_activation']:.4f}\n")
            f.write(f"Std Activation: {stats['std_activation']:.4f}\n")
            
            f.write("\nTop Activating Examples:\n")
            f.write("-" * 50 + "\n")
            for i, example in enumerate(stats['top_activations'], 1):
                f.write(f"\n{i}. Activation: {example['activation']:.4f}\n")
                f.write(f"Position: {example['position']}\n")
                input_ids = example['input_ids']
                pos = example['position']
                start = max(0, pos - context_tokens)
                end = min(len(input_ids), pos + context_tokens + 1)
                context = input_ids[start:end]
                f.write(f"Context: {tokenizer.decode(context)}\n")
            
            f.write("\nBottom Activating Examples:\n")
            f.write("-" * 50 + "\n")
            for i, example in enumerate(stats['bottom_activations'], 1):
                f.write(f"\n{i}. Activation: {example['activation']:.4f}\n")
                f.write(f"Position: {example['position']}\n")
                input_ids = example['input_ids']
                pos = example['position']
                start = max(0, pos - context_tokens)
                end = min(len(input_ids), pos + context_tokens + 1)
                context = input_ids[start:end]
                f.write(f"Context: {tokenizer.decode(context)}\n")

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load autoencoder
    logger.info("Loading autoencoder...")
    checkpoint = torch.load(args.checkpoint)
    
    autoencoder = TopKSparseAutoencoder(
        input_dim=768,  # This should match your model's hidden size
        **config['autoencoder']
    ).to(device)
    
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # Create dataset and dataloader
    logger.info("Loading activation dataset...")
    activation_dataset = ActivationDataset(config['data']['activation_cache_dir'])
    
    dataloader = DataLoader(
        activation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    # Determine which features to analyze
    if args.features == "all":
        feature_indices = None
    else:
        feature_indices = [int(x) for x in args.features.split(",")]
    
    logger.info(f"Analyzing {args.n_samples} samples...")
    feature_stats = analyze_feature_activations(
        dataloader,
        autoencoder,
        tokenizer,
        n_samples=args.n_samples,
        top_k=args.top_k,
        feature_indices=feature_indices,
        device=device
    )
    
    logger.info("Saving results...")
    save_feature_analysis(feature_stats, tokenizer, args.output_dir)
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Clean up
    if hasattr(activation_dataset, 'h5f'):
        activation_dataset.h5f.close()

if __name__ == "__main__":
    main()
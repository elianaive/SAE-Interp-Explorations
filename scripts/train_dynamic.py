import sys
import os
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset
import wandb
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import TopKSparseAutoencoder
from src.training.trainer_dynamic import SparseAutoencoderTrainer
from src.data.dynamic_activation_extractor import setup_dynamic_activation_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories(config: dict):
    """Create necessary directories."""
    dirs = [
        Path(config['data']['activation_cache_dir']),
        Path(config['training']['checkpoint_dir']),
        Path(config['wandb']['dir'])
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    parser.add_argument('--config', type=str, default='config/default_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directories(config)
    
    if config['use_wandb']:
        wandb.init(
            project=config['wandb']['project_name'],
            name=run_name,
            config=config,
            dir=config['wandb']['dir']
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Loading model and tokenizer...")
    model = HookedTransformer.from_pretrained(config['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    #if tokenizer.pad_token is None:
        #tokenizer.pad_token = tokenizer.eos_token
        #model.cfg["pad_token_id"] = model.cfg["eos_token_id"]
    
    logger.info("Loading dataset...")
    train_size = config['data']['train_size']
    raw_dataset = load_dataset(
        config['data']['name'], 
        split=f"train[:{train_size}]" if train_size not in [None, -1, "None"] else "train"
    )
    
    logger.info("Setting up dynamic activation training...")
    activation_loader, activation_dataset = setup_dynamic_activation_training(
        config=config,
        run_name=run_name,
        model=model,
        tokenizer=tokenizer,
        raw_dataset=raw_dataset
    )
    
    logger.info("Getting input dimensions...")
    sample_activation = next(iter(activation_loader))[0]
    input_dim = sample_activation.shape[-1]
    logger.info(f"Input dimension: {input_dim}")
    
    logger.info("Initializing autoencoder...")
    autoencoder = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=config['autoencoder']['latent_dim'],
        k=config['autoencoder']['k'],
        init_scale=config['autoencoder']['init_scale'],
        tied_weights=config['autoencoder']['tied_weights'],
        multi_k=config['autoencoder']['multi_k']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=float(config['training']['learning_rate']),
        eps=float(config['training']['adam_eps']),
        betas=(0.9, 0.999)
    )

    def get_batch_fn(batch_size: int, sequence_length: int):
        """Get a batch of activations."""
        try:
            return next(iter(activation_loader))
        except StopIteration:
            activation_loader.dataset.current_index = 0
            return next(iter(activation_loader))
    
    trainer = SparseAutoencoderTrainer(
        model=autoencoder,
        optimizer=optimizer,
        get_batch_fn=get_batch_fn,
        batch_size=config['training']['batch_size'],
        sequence_length=config['training']['sequence_length'],
        tokens_per_segment=config['training']['tokens_per_checkpoint'],
        k_aux=config['training']['k_aux'],
        aux_scale=config['training']['aux_scale'],
        use_wandb=config['use_wandb'],
        device=device
    )
    
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    total_tokens = config['training']['total_tokens']
    tokens_processed = 0
    previous_l_c = float('inf')
    stable_segments = 0
    best_metrics = {
        'normalized_mse': float('inf'),
        'l_c': float('inf')
    }
    
    logger.info(f"Starting training, processing {total_tokens:,} tokens...")
    pbar = tqdm(total=total_tokens, initial=tokens_processed, unit=" tokens")

    checkpoint_dir = Path(config['training']['checkpoint_dir']) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    previous_metrics = None

    # Main training loop
    while tokens_processed < total_tokens:
        metrics = trainer.train_segment(tokens_processed, total_tokens)
        tokens_processed += trainer.tokens_per_segment
        pbar.update(trainer.tokens_per_segment)
        
        logger.info(
            f"\nTokens processed: {tokens_processed:,}/{total_tokens:,}\n"
            f"Normalized MSE: {metrics['normalized_mse']:.4f}\n"
            f"Relative MSE change: {metrics.get('relative_mse_change', float('inf')):.6f}\n"
            f"Dead latents: {metrics['dead_latents_pct']:.1f}%\n"
            f"L(C): {metrics.get('l_c', 0):.4f}\n"
            f"Mean steps since activation: {metrics.get('mean_steps_since_activation', 0):.1f}"
        )
        
        pbar.set_postfix({
            'mse': f"{metrics['normalized_mse']:.4f}",
            'dead': f"{metrics['dead_latents_pct']:.1f}%",
            'L(C)': f"{metrics.get('l_c', 0):.4f}"
        })
        
        # Track best metrics and save checkpoint if improved
        improved = False
        for metric_name in best_metrics:
            if metric_name in metrics and metrics[metric_name] < best_metrics[metric_name]:
                best_metrics[metric_name] = metrics[metric_name]
                improved = True
        
        if improved:
            checkpoint = {
                'tokens_processed': tokens_processed,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_metrics': best_metrics
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
        
        # Regular checkpoint
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_{tokens_processed}.pt')
        
        # Check convergence
        is_converged = trainer.check_convergence(metrics, previous_metrics)
        previous_metrics = metrics.copy()
        
        if is_converged:
            stable_segments += 1
            logger.info(
                f"Convergence criteria met for {stable_segments} consecutive segments\n"
                f"- Dead latent change: {abs(metrics['dead_latents_pct'] - previous_metrics['dead_latents_pct']):.2f}%"
            )
        else:
            stable_segments = 0
            
        # Stop if converged for enough segments
        convergence_patience = config.get('training', {}).get('convergence_patience', 3)
        if stable_segments >= convergence_patience:
            logger.info(
                f"\nModel converged after {tokens_processed:,} tokens:\n"
                f"- Final normalized MSE: {metrics['normalized_mse']:.4f}\n"
                f"- Best normalized MSE: {best_metrics['normalized_mse']:.4f}\n"
                f"- Final L(C): {metrics.get('l_c', 0):.4f}\n"
                f"- Best L(C): {best_metrics['l_c']:.4f}\n"
                f"- Dead latents: {metrics['dead_latents_pct']:.1f}%\n"
                f"- Mean steps since activation: {metrics.get('mean_steps_since_activation', 0):.1f}"
            )
            break
            
        if config.get('training', {}).get('eval_tokens'):
            eval_metrics = trainer.evaluate(config['training']['eval_tokens'])
            logger.info(f"Evaluation metrics: {eval_metrics}")
    
    if config['use_wandb']:
        wandb.log({
            "final/normalized_mse": metrics['normalized_mse'],
            "final/l_c": metrics.get('l_c', 0),
            "final/dead_latents_pct": metrics['dead_latents_pct'],
            "final/relative_mse_change": metrics.get('relative_mse_change', float('inf')),
            "best/normalized_mse": best_metrics['normalized_mse'],
            "best/l_c": best_metrics['l_c'],
            "training/total_tokens": tokens_processed,
            "training/converged": stable_segments >= convergence_patience
        })
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
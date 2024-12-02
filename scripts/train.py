import sys
import os
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import logging
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import TopKSparseAutoencoder
from src.training.trainer import SparseAutoencoderTrainer
from src.data.activation_extractor import ActivationExtractor, ActivationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'][0],
            'attention_mask': tokens['attention_mask'][0]
        }

def ensure_directories(config: dict):
    """Create necessary directories."""
    dirs = [
        Path(config['data']['activation_cache_dir']),
        Path(config['training']['checkpoint_dir']),
        Path(config['wandb']['dir'])
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_activation_cache_path(config: dict, run_name: str) -> str:
    """Generate path for activation cache file."""
    cache_dir = Path(config['data']['activation_cache_dir'])
    return str(cache_dir / f"activations_{run_name}.hdf5")

def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    parser.add_argument('--config', type=str, default='config/default_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set run name and create directories
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directories(config)
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb']['project_name'],
            name=run_name,
            config=config,
            dir=config['wandb']['dir']
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    logger.info("Loading dataset...")
    raw_dataset = load_dataset(
        config['data']['name'],
        split=f"train[:{config['data']['train_size']}]"
    )
    dataset = TokenizedDataset(
        raw_dataset,
        tokenizer,
        max_length=config['data']['sequence_length']
    )
    
    # Generate activation cache path for this run
    activation_cache_path = get_activation_cache_path(config, run_name)
    
    logger.info("Setting up activation extraction...")
    extractor = ActivationExtractor(
        model=model,
        layer_num=config['model']['layer_num'],
        batch_size=config['data']['extraction_batch_size'],
        cache_file_path=activation_cache_path
    )
    
    logger.info("Extracting activations...")
    cache_file = extractor.extract_activations(dataset)
    
    logger.info("Setting up activation dataset...")
    activation_dataset = ActivationDataset(cache_file)
    logger.info(f"Number of activation samples: {len(activation_dataset)}")
    
    activation_loader = DataLoader(
        activation_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Get input dimension from a sample
    sample_activation = activation_dataset[0]
    input_dim = sample_activation.shape[-1]
    logger.info(f"Input dimension: {input_dim}")
    
    # Initialize autoencoder
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
        eps=float(config['training']['adam_eps'])
    )
    
    trainer = SparseAutoencoderTrainer(
        model=autoencoder,
        optimizer=optimizer,
        k_aux=config['training']['k_aux'],
        aux_scale=config['training']['aux_scale'],
        use_wandb=config['use_wandb'],
        device=device
    )
    
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    best_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in tqdm(range(config['training']['num_epochs']), desc="Training", position=0):
        metrics = trainer.train_epoch(activation_loader, epoch, config['training']['num_epochs'])
        
        # Log dead latent percentage
        dead_latent_percentage = (metrics['dead_latents'] / config['autoencoder']['latent_dim']) * 100
        logger.info(f"Epoch {epoch} - Dead latents: {dead_latent_percentage:.2f}%")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        current_loss = metrics['total_loss']
        is_best = current_loss < best_loss
        
        if is_best:
            best_loss = current_loss
            early_stopping_counter = 0
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
        else:
            early_stopping_counter += 1
            
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_{epoch:03d}.pt')
        
        # Check early stopping
        if early_stopping_counter >= config['training']['patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
            
        # Check dead latent threshold
        if dead_latent_percentage > config['training']['max_dead_latent_ratio']:
            logger.warning(f"Dead latent ratio {dead_latent_percentage:.2f}% exceeded threshold "
                         f"{config['training']['max_dead_latent_ratio']}%")
            break
    
    # Clean up
    if hasattr(activation_dataset, 'h5f'):
        activation_dataset.h5f.close()
    
    if config['use_wandb']:
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
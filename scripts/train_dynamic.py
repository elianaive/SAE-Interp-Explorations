import sys
import os
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    train_size = config['data']['train_size']

    if train_size is None or train_size == -1 or train_size == "None":
        raw_dataset = load_dataset(config['data']['name'], split="train")
    else:
        raw_dataset = load_dataset(config['data']['name'], split=f"train[:{train_size}]")
    
    logger.info("Setting up dynamic activation training...")
    activation_loader, activation_dataset = setup_dynamic_activation_training(
        config=config,
        run_name=run_name,
        model=model,
        tokenizer=tokenizer,
        raw_dataset=raw_dataset
    )
    
    # Get input dimension from first chunk
    logger.info("Getting input dimensions...")
    sample_activation = next(iter(activation_loader))[0]
    input_dim = sample_activation.shape[-1]
    logger.info(f"Input dimension: {input_dim}")
    
    # Initialize autoencoder
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
        betas=(0.9, 0.999)  # Default Adam betas
    )

    def get_batch_fn(batch_size: int, sequence_length: int):
        """Get a batch of activations."""
        try:
            return next(iter(activation_loader))
        except StopIteration:
            # Reset the loader if we've exhausted it
            activation_loader.dataset.current_index = 0
            return next(iter(activation_loader))
    
    # Initialize trainer
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
    best_loss = float('inf')
    total_tokens = config['training']['total_tokens']
    tokens_processed = 0
    early_stopping_counter = 0
    logger.info(f"Starting training, processing {total_tokens:,} tokens...")
    pbar = tqdm(total=total_tokens, initial=tokens_processed, unit=" tokens")

    # Main training loop
    while tokens_processed < total_tokens:
        # Train for one segment
        metrics = trainer.train_segment(tokens_processed, total_tokens)
        tokens_processed += trainer.tokens_per_segment
        pbar.update(trainer.tokens_per_segment)
        
        dead_latent_percentage = metrics['dead_latents_pct']
        
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'dead': f"{metrics['dead_latents']}"
        })
        
        logger.info(
            f"Tokens processed: {tokens_processed:,}/{total_tokens:,} - "
            f"Loss: {metrics['total_loss']:.4f} - "
            f"Dead latents: {dead_latent_percentage:.2f}%"
        )
        
        checkpoint = {
            'tokens_processed': tokens_processed,
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
            
        # Regular checkpoint
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_{tokens_processed}.pt')
        
        # Optional evaluation
        if config.get('training', {}).get('eval_tokens'):
            eval_metrics = trainer.evaluate(config['training']['eval_tokens'])
            logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Optional early stopping check
        if config.get('training', {}).get('patience'):
            if early_stopping_counter >= config['training']['patience']:
                logger.info(f"Early stopping triggered after {tokens_processed:,} tokens")
                break
        
        # Optional dead latent check
        # if config.get('training', {}).get('max_dead_latent_ratio'):
        #     if dead_latent_percentage > config['training']['max_dead_latent_ratio']:
        #         logger.warning(
        #             f"Dead latent ratio {dead_latent_percentage:.2f}% exceeded threshold "
        #             f"{config['training']['max_dead_latent_ratio']}%"
        #         )
        #         break
    
    if config['use_wandb']:
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
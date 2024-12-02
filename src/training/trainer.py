import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import wandb
import logging
from tqdm import tqdm
from ..models.autoencoder import TopKSparseAutoencoder

class SparseAutoencoderTrainer:
    def __init__(
        self,
        model: TopKSparseAutoencoder,
        optimizer: torch.optim.Optimizer,
        k_aux: int = 512,
        aux_scale: float = 1/32,
        use_wandb: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.k_aux = k_aux
        self.aux_scale = aux_scale
        self.device = device
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get("project", "SparseAutoencoder"),
                config=wandb_config
            )
            wandb.config.update({"device": self.device})
            logging.info("WandB initialized")
            
    def compute_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        latents: torch.Tensor,
        dead_latents: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """Compute main reconstruction loss and auxiliary loss."""
        # Main reconstruction loss
        main_loss = F.mse_loss(recon_x, x)
        metrics = {"main_loss": main_loss.item()}

        # Auxiliary loss for dead latents
        aux_loss = torch.tensor(0.0, device=self.device)
        if dead_latents is not None and dead_latents.any():
            # Reshape latents if necessary
            batch_size = latents.size(0)
            seq_len = latents.size(1) if len(latents.shape) == 3 else 1
            latent_dim = latents.size(-1)
            
            # Reshape to [batch * seq, latent_dim]
            flat_latents = latents.view(-1, latent_dim)
            
            # Get activations for dead latents
            dead_latent_activations = flat_latents[:, dead_latents]
            
            if dead_latent_activations.size(1) > 0:
                k = min(self.k_aux, dead_latent_activations.size(1))
                top_aux = torch.topk(dead_latent_activations, k, dim=1)[0]
                aux_loss = F.mse_loss(top_aux, torch.zeros_like(top_aux))
                metrics["aux_loss"] = aux_loss.item()

        # Compute sparsity metrics
        metrics.update({
            "active_latents": (latents != 0).float().mean().item(),
            "dead_latents": dead_latents.sum().item() if dead_latents is not None else 0
        })

        total_loss = main_loss + self.aux_scale * aux_loss
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics

    def train_step(self, batch):
        """Single training step."""
        if isinstance(batch, dict):
            x = batch['input_ids']
        else:
            x = batch
            
        x = x.to(self.device)
        self.optimizer.zero_grad()
        
        # Get dead latents before forward pass
        dead_latents = self.model.get_dead_latents()
        #if dead_latents.any():
            #print("Dead latent found")
        
        # Forward pass
        recon_x, latents = self.model(x)
        
        # Compute loss
        loss, metrics = self.compute_loss(x, recon_x, latents, dead_latents)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Normalize decoder weights if using tied weights
        if self.model.tied_weights:
            with torch.no_grad():
                self.model.encoder.weight.data = F.normalize(
                    self.model.encoder.weight.data, dim=1
                )
        
        return metrics
    
    def train_epoch(
        self, 
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics: Dict[str, float] = {}
        
        # Create description for epoch progress bar
        epoch_desc = f"Epoch {epoch}/{total_epochs-1}"
        
        # Progress bar for batches within epoch
        pbar = tqdm(dataloader, desc=epoch_desc, leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            
            # Update epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            
            # Update progress bar postfix with current loss
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'dead': f"{metrics['dead_latents']}"
            })
            
            # Log batch metrics
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({"epoch": epoch, **{f"batch/{k}": v for k, v in metrics.items()}})
        
        # Average epoch metrics
        num_batches = len(dataloader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Log epoch metrics
        if self.use_wandb:
            wandb.log({"epoch": epoch, **{f"epoch/{k}": v for k, v in epoch_metrics.items()}})
        
        return epoch_metrics

    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        eval_metrics: Dict[str, float] = {}
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", leave=True)
            for batch in pbar:
                batch = batch.to(self.device)
                recon_batch, latents = self.model(batch, update_counts=False)
                
                # Compute metrics without auxiliary loss
                _, metrics = self.compute_loss(batch, recon_batch, latents)
                
                # Update eval metrics
                for k, v in metrics.items():
                    eval_metrics[k] = eval_metrics.get(k, 0) + v
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{metrics['total_loss']:.4f}"})
        
        # Average metrics
        num_batches = len(dataloader)
        eval_metrics = {k: v / num_batches for k, v in eval_metrics.items()}
        
        return eval_metrics
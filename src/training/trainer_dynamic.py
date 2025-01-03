import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, Dict, Any, Callable
import wandb
import logging
from tqdm import tqdm

class ContinuousTokenDataset(IterableDataset):
    def __init__(
        self,
        get_batch_fn: Callable,
        batch_size: int,
        sequence_length: int,
        tokens_per_segment: int
    ):
        super().__init__()
        self.get_batch_fn = get_batch_fn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.steps = tokens_per_segment // (batch_size * sequence_length)
        
    def __iter__(self):
        for _ in range(self.steps):
            yield self.get_batch_fn(self.batch_size, self.sequence_length)

class SparseAutoencoderTrainer:
    def __init__(
        self,
        model: "TopKSparseAutoencoder",
        optimizer: torch.optim.Optimizer,
        get_batch_fn: Callable,
        batch_size: int,
        sequence_length: int,
        tokens_per_segment: int,
        k_aux: int = 512,
        aux_scale: float = 1/32,
        use_wandb: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The autoencoder model
            optimizer: The optimizer
            get_batch_fn: Function that returns batches of tokens/activations
            batch_size: Batch size
            sequence_length: Sequence length per item
            tokens_per_segment: Number of tokens to process before checkpointing
            k_aux: Number of auxiliary latents for dead feature recovery
            aux_scale: Scale factor for auxiliary loss
            use_wandb: Whether to use Weights & Biases logging
            device: Device to train on
            wandb_config: Additional W&B configuration
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.get_batch_fn = get_batch_fn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokens_per_segment = tokens_per_segment
        self.k_aux = k_aux
        self.aux_scale = aux_scale
        self.device = device
        self.use_wandb = use_wandb

        self.tokens_per_batch = batch_size * sequence_length
        self.steps_per_segment = tokens_per_segment // self.tokens_per_batch
        
        self.mse_history = []
        self.window_size = 100
        self.tokens_processed = 0
        
        if self.use_wandb:
            wandb_config = wandb_config or {}
            wandb_config.update({
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tokens_per_segment": tokens_per_segment,
                "k_aux": k_aux,
                "aux_scale": aux_scale,
            })
            wandb.init(
                project=wandb_config.get("project", "SparseAutoencoder"),
                config=wandb_config
            )
            logging.info("WandB initialized")
        self.register_baseline_stats()
            
    def register_baseline_stats(self):
        """Calculate baseline reconstruction error for normalization."""
        with torch.no_grad():
            batch = self.get_batch_fn(self.batch_size, self.sequence_length)
            x = batch['input_ids'].to(self.device) if isinstance(batch, dict) else batch.to(self.device)
            
            self.activation_mean = x.mean(dim=0, keepdim=True)
            
            self.baseline_mse = F.mse_loss(x, self.activation_mean.expand_as(x)).item()

    def compute_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        latents: torch.Tensor,
        dead_latents: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute main reconstruction loss and auxiliary loss for dead latent recovery.
        """
        # Main reconstruction loss
        main_loss = F.mse_loss(recon_x, x)
        metrics = {"main_loss": main_loss.item()}

        # Add normalized MSE and track convergence
        normalized_mse = main_loss.item() / self.baseline_mse
        metrics['normalized_mse'] = normalized_mse
        
        self.mse_history.append(normalized_mse)
        if len(self.mse_history) > self.window_size:
            self.mse_history.pop(0)
            recent_mse = torch.tensor(self.mse_history)
            relative_change = (recent_mse[1:] - recent_mse[:-1]).abs().mean() / recent_mse.mean()
            metrics['relative_mse_change'] = relative_change.item()

        # Auxiliary loss for dead latents
        aux_loss = torch.tensor(0.0, device=self.device)
        if dead_latents is not None and dead_latents.any():
            error = x - recon_x
            
            latent_dim = latents.size(-1)
            flat_latents = latents.view(-1, latent_dim)
            flat_error = error.view(-1, error.size(-1))
            
            dead_latent_activations = flat_latents[:, dead_latents]
            
            if dead_latent_activations.size(1) > 0:
                k = min(self.k_aux, dead_latent_activations.size(1))
                values, indices = torch.topk(dead_latent_activations, k, dim=1)
                
                sparse_dead_latents = torch.zeros_like(dead_latent_activations)
                sparse_dead_latents.scatter_(1, indices, values)
                
                if self.model.tied_weights:
                    dead_weights = self.model.encoder.weight[dead_latents].t()
                else:
                    dead_weights = self.model.decoder.weight.t()[dead_latents]
                dead_recon = F.linear(sparse_dead_latents, dead_weights)
                aux_loss = F.mse_loss(dead_recon, flat_error)
                metrics["aux_loss"] = aux_loss.item()

        # Compute latent statistics
        total_latents = latents.size(-1)
        active_mask = (latents != 0)
        if len(active_mask.shape) == 3:
            active_mask = active_mask.view(-1, active_mask.size(-1))
        unique_active = active_mask.any(dim=0).sum().item()
        active_percent = (unique_active / total_latents * 100)
        
        dead_count = dead_latents.sum().item() if dead_latents is not None else 0
        dead_percent = (dead_count / total_latents * 100) if dead_latents is not None else 0
        
        # Basic metrics
        metrics.update({
            "active_latents_pct": active_percent,
            "active_latents": unique_active,
            "dead_latents_pct": dead_percent,
            "dead_latents": dead_count,
            "total_latents": total_latents
        })

        # Add activation timing metrics
        if hasattr(self.model, 'get_activation_metrics'):
            metrics.update(self.model.get_activation_metrics())

        # Compute L(C) metric from paper
        tokens_seen = max(1, self.tokens_processed / self.tokens_per_batch)  # Avoid division by zero
        metrics['l_c'] = normalized_mse + 0.056 * (tokens_seen ** -0.084)

        # Combine losses
        total_loss = main_loss + self.aux_scale * aux_loss
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step."""
        if isinstance(batch, dict):
            x = batch['input_ids']
        else:
            x = batch
        
        x = x.to(self.device)
        self.optimizer.zero_grad()
        
        # Forward pass
        recon_x, latents = self.model(x)
        
        # Update activation counts
        self.model.update_activation_counts(latents)

        # Get dead latents after updating counts
        dead_latents = self.model.get_dead_latents()

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

    def train_segment(self, tokens_processed: int, total_tokens: int) -> Dict[str, float]:
        """Train on a segment of tokens."""
        self.model.train()
        segment_metrics: Dict[str, float] = {}
        
        dataset = ContinuousTokenDataset(
            self.get_batch_fn,
            self.batch_size,
            self.sequence_length,
            self.tokens_per_segment
        )
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        
        pbar = tqdm(
            enumerate(dataloader),
            total=self.steps_per_segment,
            desc=f"Training segment",
            leave=False
        )
        
        for batch_idx, batch in pbar:
            metrics = self.train_step(batch)
            
            # Update segment metrics
            for k, v in metrics.items():
                segment_metrics[k] = segment_metrics.get(k, 0) + v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'dead': f"{metrics['dead_latents']}"
            })
            
            if self.use_wandb and batch_idx % 10 == 0:
                current_tokens = tokens_processed + (batch_idx + 1) * self.tokens_per_batch
                wandb.log({
                    "tokens": current_tokens,
                    **{f"batch/{k}": v for k, v in metrics.items()}
            })
        
        pbar.close()
        
        # Average segment metrics
        segment_metrics = {k: v / self.steps_per_segment for k, v in segment_metrics.items()}
        
        # Log segment metrics
        if self.use_wandb:
            wandb.log({
                "tokens": tokens_processed + self.tokens_per_segment,
                **{f"segment/{k}": v for k, v in segment_metrics.items()}
            })
        
        return segment_metrics

    def evaluate(self, num_eval_tokens: int) -> Dict[str, float]:
        self.model.eval()
        eval_metrics: Dict[str, float] = {}
        
        dataset = ContinuousTokenDataset(
            self.get_batch_fn,
            self.batch_size,
            self.sequence_length,
            num_eval_tokens
        )
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", total=dataset.steps)
            for batch in pbar:
                batch = batch.to(self.device)
                recon_batch, latents = self.model(batch, update_counts=False)
                
                # Get dead latents before computing loss
                dead_latents = self.model.get_dead_latents()
                _, metrics = self.compute_loss(batch, recon_batch, latents, dead_latents)
                
                for k, v in metrics.items():
                    eval_metrics[k] = eval_metrics.get(k, 0) + v
                
                pbar.set_postfix({
                    'mse': f"{metrics['normalized_mse']:.4f}",
                    'dead': f"{metrics['dead_latents']}",
                })
        
        # Average metrics
        eval_metrics = {k: v / dataset.steps for k, v in eval_metrics.items()}
        
        return eval_metrics
    
    def check_convergence(self, metrics: Dict[str, float], previous_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Check if model has converged based on paper criteria + dead latent stability."""
        # Need enough history for stable measurements
        if len(self.mse_history) < self.window_size:
            return False
            
        if previous_metrics is None:
            return False

        # Calculate changes in key metrics
        dead_latent_change = abs(metrics['dead_latents_pct'] - previous_metrics['dead_latents_pct'])
        l_c_change = abs(metrics['l_c'] - previous_metrics['l_c']) / previous_metrics['l_c']
        
        # Check key criteria
        criteria = [
            # MSE changes are small
            metrics.get('relative_mse_change', float('inf')) < 1e-4,
            
            # Dead latents are stabilizing (less than 0.5% change)
            dead_latent_change < 0.5,
            
            # L(C) isn't getting worse
            l_c_change < 0.01,  # 1% change threshold
            
            # Normalized MSE is better than mean prediction
            metrics['normalized_mse'] < 1.0
        ]
        
        return all(criteria)
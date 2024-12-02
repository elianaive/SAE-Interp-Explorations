import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TopKSparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int,
        init_scale: float = 1.0,
        norm_input: bool = True,
        tied_weights: bool = True,
        multi_k: bool = False,  # Whether to use Multi-TopK
        dead_threshold: int = 10_000_000,  # Tokens before considering a latent dead
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.norm_input = norm_input
        self.tied_weights = tied_weights
        self.multi_k = multi_k
        self.dead_threshold = dead_threshold
        
        # Initialize encoder and bias
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.encoder_bias = self.encoder.bias
        
        if tied_weights:
            self.decoder = None
        else:
            self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
            
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Initialize activation tracking
        self.register_buffer('activation_counts', torch.zeros(latent_dim))
        self.register_buffer('tokens_seen', torch.tensor(0))
        
        self._init_weights(init_scale)

    def _init_weights(self, scale: float = 1.0):
        """Initialize weights following paper's scheme."""
        with torch.no_grad():
            weights = torch.randn(self.latent_dim, self.input_dim)
            weights = F.normalize(weights, dim=1) * scale
            self.encoder.weight.data = weights
            
            if not self.tied_weights and self.decoder is not None:
                self.decoder.weight.data = weights.t()

    def top_k_activation(self, x: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """Apply TopK activation function."""
        if k is None:
            k = self.k
        
        orig_shape = x.shape
        if len(orig_shape) == 3:  # [batch, seq, latent]
            x = x.view(-1, x.size(-1))
            
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1)
        x = x * (x >= threshold)
        
        if len(orig_shape) == 3:
            x = x.view(orig_shape)
            
        return x

    def multi_top_k_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Multi-TopK activation as described in the paper."""
        # Main activation with k latents
        main_activation = self.top_k_activation(x, self.k)
        
        # Additional activation with 4k latents (scaled down by 8 as per paper)
        extra_activation = self.top_k_activation(x, 4 * self.k) / 8.0
        
        return main_activation + extra_activation

    def update_activation_counts(self, latents: torch.Tensor):
        """Update tracking of neuron activation counts."""
        with torch.no_grad():
            # Sum across batch and sequence dimensions if present
            active_latents = (latents != 0)
            while len(active_latents.shape) > 1:
                active_latents = active_latents.sum(0)
            self.activation_counts += active_latents
            
            # Update tokens seen (multiply batch size by sequence length if relevant)
            num_tokens = latents.size(0)
            if len(latents.shape) == 3:
                num_tokens *= latents.size(1)
            self.tokens_seen += num_tokens

    def get_dead_latents(self, threshold: Optional[int] = None) -> torch.Tensor:
        """Get mask of dead latents (not activated in threshold tokens)."""
        if threshold is None:
            threshold = self.dead_threshold
            
        if self.tokens_seen < threshold:
            return torch.zeros(self.latent_dim, dtype=torch.bool, device=self.activation_counts.device)
            
        return self.activation_counts < (self.tokens_seen / threshold)

    def refine_activations(self, x: torch.Tensor, latents: torch.Tensor, 
                          num_iterations: int = 100) -> torch.Tensor:
        """
        Refine activation values through iterative optimization while keeping
        the sparsity pattern fixed, as described in the paper.
        """
        sparsity_pattern = (latents != 0)
        refined_latents = latents.clone().detach()
        refined_latents.requires_grad = True
        
        # Use SGD for refinement
        optimizer = torch.optim.SGD([refined_latents], lr=0.01)
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Reconstruct with current latents
            if self.tied_weights:
                recon = F.linear(refined_latents, self.encoder.weight.t())
            else:
                recon = self.decoder(refined_latents)
                
            # Compute reconstruction loss
            loss = F.mse_loss(recon, x)
            loss.backward()
            
            # Update only non-zero positions
            with torch.no_grad():
                refined_latents.grad *= sparsity_pattern
                optimizer.step()
                # Ensure non-negative values (as we use ReLU in forward pass)
                refined_latents.clamp_(min=0)
                # Reset gradient for non-active positions
                refined_latents.data *= sparsity_pattern
        
        return refined_latents.detach()

    def forward(
        self, 
        x: torch.Tensor,
        refine: bool = False,
        update_counts: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            refine: Whether to refine activations
            update_counts: Whether to update activation counts
            
        Returns:
            tuple of:
                - Reconstructed input
                - Latent activations
        """
        orig_shape = x.shape
        
        # Reshape if necessary to handle sequence dimension
        if len(x.shape) == 3:
            x = x.view(-1, x.size(-1))
            
        if self.norm_input:
            x = x - self.pre_bias
            x = F.normalize(x, dim=1)
            
        # Encode
        latents = self.encoder(x)
        
        # Apply activation function
        if self.multi_k:
            sparse_latents = self.multi_top_k_activation(latents)
        else:
            sparse_latents = self.top_k_activation(latents)
        
        # Optionally refine activations
        if refine:
            sparse_latents = self.refine_activations(x, sparse_latents)
        
        # Decode
        if self.tied_weights:
            output = F.linear(sparse_latents, self.encoder.weight.t())
        else:
            output = self.decoder(sparse_latents)
            
        if self.norm_input:
            output = output + self.pre_bias
            
        # Restore original shape if necessary
        if len(orig_shape) == 3:
            output = output.view(orig_shape)
            sparse_latents = sparse_latents.view(orig_shape[0], orig_shape[1], -1)
        
        # Update activation counts if training
        if update_counts:
            self.update_activation_counts(sparse_latents)
            
        return output, sparse_latents
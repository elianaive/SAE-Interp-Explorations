import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Optional, List, Tuple
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import os

logger = logging.getLogger(__name__)

class ActivationExtractor:
    def __init__(
        self,
        model: PreTrainedModel,
        layer_num: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        cache_file_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.layer_num = layer_num
        self.device = device
        self.batch_size = batch_size
        self.activation = None  # Store only the last activation
        self.cache_file_path = cache_file_path or 'activations.hdf5'
        self._register_hook()
        
    def _register_hook(self):
        """Register forward hook to capture activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Usually the first element is the hidden states
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store activation
            self.activation = hidden_states.detach().cpu()
        
        # Find the target layer based on model architecture
        if hasattr(self.model, 'transformer'):
            # For GPT-Neo style models, we want to capture the output of the MLP
            target_layer = self.model.transformer.h[self.layer_num].mlp
        elif hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                target_layer = self.model.model.layers[self.layer_num].mlp
            else:
                # Print model structure to help debug
                logger.info(f"Model structure: {self.model}")
                raise ValueError("Unsupported model architecture")
        else:
            # Print model structure to help debug
            logger.info(f"Model structure: {self.model}")
            raise ValueError("Unsupported model architecture")
        
        # Register hook after MLP
        target_layer.register_forward_hook(hook)
    
    @torch.no_grad()
    def extract_activations(
        self,
        dataset: Dataset,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Extract activations from the model and store them in a file."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        total_size = 0
        activation_shape = None  # To be determined after first batch
        
        # Prepare the HDF5 file
        with h5py.File(self.cache_file_path, 'w') as h5f:
            # First pass to determine total size
            for batch in tqdm(dataloader, desc="Calculating total size"):
                batch_size = batch['input_ids'].size(0) if isinstance(batch, dict) else batch.size(0)
                total_size += batch_size
            
            # Reset dataloader iterator
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Second pass to extract activations
            data_idx = 0
            for batch in tqdm(dataloader, desc="Extracting activations"):
                # Move to device and forward pass
                batch_dict = batch if isinstance(batch, dict) else {'input_ids': batch}
                batch_size = batch_dict['input_ids'].size(0)
                batch_dict = {k: v.to(self.device) if torch.is_tensor(v) else v 
                              for k, v in batch_dict.items()}
                self.model(**batch_dict)
                
                # Get the activation from the last hook call
                activations = self.activation
                self.activation = None  # Clear stored activation to save memory
                
                activations_np = activations.numpy()
                if activation_shape is None:
                    activation_shape = activations_np.shape[1:]  # Exclude batch dimension
                    h5_dataset = h5f.create_dataset(
                        'activations',
                        shape=(total_size, *activation_shape),
                        dtype=activations_np.dtype
                    )
                
                h5_dataset[data_idx:data_idx + batch_size] = activations_np
                data_idx += batch_size
        
        # Return a reference to the HDF5 file (path)
        return self.cache_file_path

class ActivationDataset(Dataset):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path
        # Don't keep the file handle open
        with h5py.File(self.hdf5_file_path, 'r') as f:
            self.length = len(f['activations'])
            self.shape = f['activations'].shape[1:]  # Store shape for later use
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Open and close the file for each access
        with h5py.File(self.hdf5_file_path, 'r') as f:
            activation = f['activations'][idx]
        return torch.from_numpy(activation.astype('float32'))

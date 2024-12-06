import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Iterator, Dict
import h5py
import logging
from tqdm import tqdm
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DynamicTokenizedDataset(Dataset):
    """Dataset that handles tokenization of raw text data."""
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

class DynamicActivationExtractor:
    """Extracts activations on-the-fly with caching capabilities."""
    def __init__(
        self,
        model: AutoModelForCausalLM,
        layer_num: int,
        cache_dir: str,
        chunks_to_keep: int = 2,
        chunk_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32
    ):
        self.model = model.to(device)
        self.layer_num = layer_num
        self.cache_dir = Path(cache_dir)
        self.chunks_to_keep = chunks_to_keep
        self.chunk_size = chunk_size
        self.device = device
        self.batch_size = batch_size
        self.activation = None
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._register_hook()
        
    def _register_hook(self):
        """Register forward hook to capture activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activation = hidden_states.detach().cpu()
        
        # Find and register hook on target layer
        if hasattr(self.model, 'transformer'):
            target_layer = self.model.transformer.h[self.layer_num]
        elif hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                target_layer = self.model.model.layers[self.layer_num]
            else:
                raise ValueError("Unsupported model architecture")
        else:
            raise ValueError("Unsupported model architecture")
        
        target_layer.register_forward_hook(hook)

class DynamicActivationDataset(IterableDataset):
    """Dataset that generates and manages activations dynamically."""
    def __init__(
        self,
        source_dataset: Dataset,
        extractor: DynamicActivationExtractor,
        run_name: str
    ):
        self.source_dataset = source_dataset
        self.extractor = extractor
        self.run_name = run_name
        self.total_size = len(source_dataset)
        self.cached_chunks: Dict[int, str] = {}
        
    def _get_chunk_path(self, chunk_id: int) -> Path:
        """Get path for chunk cache file."""
        return self.extractor.cache_dir / f"{self.run_name}_chunk_{chunk_id}.h5"
    
    @torch.no_grad()
    def _generate_chunk(self, chunk_id: int) -> Path:
        """Generate activations for a specific chunk."""
        start_idx = chunk_id * self.extractor.chunk_size
        end_idx = min(start_idx + self.extractor.chunk_size, self.total_size)
        
        chunk_path = self._get_chunk_path(chunk_id)
        
        # Create subset of source dataset
        chunk_data = torch.utils.data.Subset(
            self.source_dataset,
            range(start_idx, end_idx)
        )
        
        dataloader = DataLoader(
            chunk_data,
            batch_size=self.extractor.batch_size,
            shuffle=False
        )
        
        self.extractor.model.eval()
        
        with h5py.File(chunk_path, 'w') as h5f:
            data_idx = 0
            
            for batch in tqdm(dataloader, desc=f"Generating chunk {chunk_id}", position=2, leave=False):
                # Process batch
                batch_dict = batch if isinstance(batch, dict) else {'input_ids': batch}
                batch_dict = {k: v.to(self.extractor.device) if torch.is_tensor(v) else v 
                            for k, v in batch_dict.items()}
                
                # Forward pass
                self.extractor.model(**batch_dict)
                
                # Get and clear activation
                activations = self.extractor.activation
                self.extractor.activation = None
                activations_np = activations.numpy()
                
                # Create dataset if not exists
                if data_idx == 0:
                    h5_dataset = h5f.create_dataset(
                        'activations',
                        shape=(end_idx - start_idx, *activations_np.shape[1:]),
                        dtype=activations_np.dtype
                    )
                
                # Store activations
                current_batch_size = activations_np.shape[0]
                h5_dataset[data_idx:data_idx + current_batch_size] = activations_np
                data_idx += current_batch_size
        
        return chunk_path
    
    def _manage_cache(self, needed_chunk: int):
        """Manage chunk cache, removing old chunks if necessary."""
        if needed_chunk not in self.cached_chunks:
            # Remove oldest chunks if at capacity
            while len(self.cached_chunks) >= self.extractor.chunks_to_keep:
                oldest_chunk = min(self.cached_chunks.keys())
                chunk_path = self.cached_chunks[oldest_chunk]
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                del self.cached_chunks[oldest_chunk]
            
            # Generate and cache new chunk
            chunk_path = str(self._generate_chunk(needed_chunk))
            self.cached_chunks[needed_chunk] = chunk_path
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            per_worker = int(self.total_size / worker_info.num_workers)
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else self.total_size
        else:
            start_idx = 0
            end_idx = self.total_size
        
        for idx in range(start_idx, end_idx):
            chunk_id = idx // self.extractor.chunk_size
            within_chunk_idx = idx % self.extractor.chunk_size
            
            self._manage_cache(chunk_id)
            
            with h5py.File(self.cached_chunks[chunk_id], 'r') as f:
                activation = f['activations'][within_chunk_idx]
            
            yield torch.from_numpy(activation.astype('float32'))
    
    def __len__(self) -> int:
        return self.total_size

def setup_dynamic_activation_training(config: dict, run_name: str, model, tokenizer, raw_dataset):
    """Setup function to initialize dynamic activation training."""
    tokenized_dataset = DynamicTokenizedDataset(
        raw_dataset,
        tokenizer,
        max_length=config['data']['sequence_length']
    )
    
    extractor = DynamicActivationExtractor(
        model=model,
        layer_num=config['model']['layer_num'],
        cache_dir=config['data']['activation_cache_dir'],
        chunks_to_keep=config.get('data', {}).get('chunks_to_keep', 2),
        chunk_size=config.get('data', {}).get('chunk_size', 1000),
        batch_size=config['data']['extraction_batch_size']
    )
    
    activation_dataset = DynamicActivationDataset(
        source_dataset=tokenized_dataset,
        extractor=extractor,
        run_name=run_name
    )
    
    activation_loader = DataLoader(
        activation_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('data', {}).get('num_workers', 0),
        pin_memory=True
    )
    
    return activation_loader, activation_dataset
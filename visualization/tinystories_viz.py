import torch
import yaml
import os
import sys
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_vis.utils_fns import get_device
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.data_config_classes import SaeVisConfig
from IPython.display import HTML

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.autoencoder import TopKSparseAutoencoder

class SAEVisWrapper:
    def __init__(self, encoder):
        self.encoder = encoder
        self.W_enc = encoder.encoder.weight.t()  # Transpose to match expected shape
        self.W_dec = encoder.encoder.weight if encoder.tied_weights else encoder.decoder.weight.t()
        self.b_enc = encoder.encoder.bias
        self.b_dec = encoder.pre_bias
        # Add config attributes
        self.d_in = encoder.input_dim
        self.d_hidden = encoder.latent_dim
        
    def state_dict(self):
        return {
            'W_enc': self.W_enc,
            'W_dec': self.W_dec,
            'b_enc': self.b_enc,
            'b_dec': self.b_dec
        }

def visualize_features(checkpoint_dir: str, n_features: int = 64, n_samples: int = 2048):
    device = get_device()
    
    # Load config and checkpoint
    config_path = os.path.join("config/default_config.yaml")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extract dimensions from checkpoint
    input_dim = checkpoint['model_state_dict']['encoder.weight'].shape[1]
    latent_dim = checkpoint['model_state_dict']['encoder.weight'].shape[0]
    
    # Get autoencoder config, removing dimensions if they exist
    ae_config = config['autoencoder'].copy()
    ae_config.pop('input_dim', None)
    ae_config.pop('latent_dim', None)
    
    # Create and load autoencoder
    encoder = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        **ae_config
    ).to(device)
    
    encoder.load_state_dict(checkpoint['model_state_dict'])
    
    # Add required cfg attribute for visualization
    encoder.cfg = type('Config', (), {
        'd_mlp': min(input_dim, latent_dim),
        'dict_mult': max(input_dim, latent_dim) // min(input_dim, latent_dim)
    })()
    
    # Set up model and data using HookedTransformer
    model = HookedTransformer.from_pretrained(
        config['model']['name'],
        device=device
    )
    
    data = load_dataset(config['data']['name'], split="train")
    
    # Tokenize data using the model's tokenizer
    tokenized_data = []
    for i in range(min(n_samples, len(data))):
        tokens = model.tokenizer(
            data[i]["text"],
            max_length=config['training']['sequence_length'],
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )["input_ids"]
        tokenized_data.append(tokens)
    
    all_tokens = torch.cat(tokenized_data, dim=0).to(device)
    
    # Create visualization config and data
    sae_vis_config = SaeVisConfig(
        hook_point="blocks.3.hook_resid_pre",  # MLP of block 3
        features=range(n_features),
        verbose=True,
    )
    
    encoder_wrapped = SAEVisWrapper(encoder)
    
    sae_vis_data = SaeVisData.create(
        encoder=encoder_wrapped,
        model=model,
        tokens=all_tokens,
        cfg=sae_vis_config,
    )
    
    # Save visualizations
    os.makedirs("visualizations", exist_ok=True)
    feature_vis_path = "visualizations/feature_vis.html"
    prompt_vis_path = "visualizations/prompt_vis.html"
    
    # Feature-centric visualization
    sae_vis_data.save_feature_centric_vis(feature_vis_path, feature_idx=0)
    
    # Prompt-centric visualization
    prompt = "Once upon a time, there was a little girl named".strip()
    tokens = model.tokenizer.encode(prompt)
    str_toks_list = [f"{t!r} ({i})" for i, t in enumerate(tokens)]
    print("Tokens:", tokens)
    print("str_toks_list:", str_toks_list)
    seq_pos = 3
    sae_vis_data.save_prompt_centric_vis(
        prompt=prompt,
        filename=prompt_vis_path,
        metric='act-quantiles',
        seq_pos=seq_pos
    )
    
    return feature_vis_path, prompt_vis_path

if __name__ == "__main__":
    feature_vis_path, prompt_vis_path = visualize_features("checkpoints/20241208_015406")
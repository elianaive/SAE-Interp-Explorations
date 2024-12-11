# SAE-Interp-Explorations

An implementation of TopK sparse autoencoders for mechanistic interpretability, replicating key techniques from ["Scaling and Evaluating Sparse Autoencoders"](https://arxiv.org/abs/2406.04093) (Gao et al., 2024).

## Features

- TopK sparse autoencoder with dead feature prevention
- Dynamic activation extraction and caching system for memory efficiency 
- Multi-TopK activation function support
- Activation refinement through iterative optimization
- Integration with TransformerLens and HuggingFace models
- Weights & Biases logging and visualization
- SAE-VIS visualization support

## Installation

```bash
git clone https://github.com/elianaive/SAE-Interp-Explorations.git
cd SAE-Interp-Explorations
pip install -r requirements.txt
```

## Usage

```python
from models.autoencoder import TopKSparseAutoencoder
from data.activation_extractor import DynamicActivationExtractor
from training.trainer_dynamic import SparseAutoencoderTrainer
from transformer_lens import HookedTransformer
from datasets import load_dataset

# Load model and dataset
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Setup activation extraction
activation_loader, activation_dataset = DynamicActivationExtractor.setup_dynamic_activation_training(
        config=config,
        run_name=run_name,
        model=model,
        tokenizer=tokenizer,
        raw_dataset=raw_dataset
    )

# Initialize autoencoder
sae = TopKSparseAutoencoder(
    input_dim=768,  # Hidden dim of your model
    latent_dim=12288,  # Usually 16x input_dim
    k=32  # Number of active features per token
)

# Setup training
trainer = SparseAutoencoderTrainer(
    model=sae,
    optimizer=optimizer,
    get_batch_fn=activation_loader.get_batch,
    batch_size=128,
    sequence_length=64
)
```

## Acknowledgments

Implementation based on techniques described in Scaling and Evaluating Sparse Autoencoders.

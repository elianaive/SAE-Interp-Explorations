import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.autoencoder import TopKSparseAutoencoder
from src.training.trainer import SparseAutoencoderTrainer
from src.data.activation_extractor import ActivationExtractor, ActivationDataset

class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure we have a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
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

def main():
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    
    # Set padding token before creating dataset
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("Loading dataset...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[0:50000]")
    dataset = TokenizedDataset(raw_dataset, tokenizer)
    
    print("Setting up extractor...")
    print(f"Model layers available: {model.config.num_hidden_layers}")
    extractor = ActivationExtractor(
        model=model,
        layer_num=3,
        cache_file_path="data/activations.hdf5"
    )
    
    print("Extracting activations...")
    activations_file = extractor.extract_activations(dataset)

    print("Setting up training...")
    activation_dataset = ActivationDataset(activations_file)
    print(f"Extracted activations length: {len(activation_dataset)}")

    activation_loader = DataLoader(
        activation_dataset,
        batch_size=10,
        shuffle=True
    )

    
    sample_activation = activation_dataset[0]
    input_dim = sample_activation.shape[-1]
    print(f"Initializing autoencoder with input_dim={input_dim}")
    
    autoencoder = TopKSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=1024,
        k=8
    )
    
    # Initialize trainer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    trainer = SparseAutoencoderTrainer(
        model=autoencoder,
        optimizer=optimizer,
        use_wandb=True
    )
    
    print("Starting training...")
    num_epochs = 20
    for epoch in tqdm(range(num_epochs), desc="Training", position=0):
        metrics = trainer.train_epoch(activation_loader, epoch, num_epochs)
        print(f"Epoch {epoch} metrics:", metrics)

if __name__ == "__main__":
    main()
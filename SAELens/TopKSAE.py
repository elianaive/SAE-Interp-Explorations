# Trying to use the same config as my personal replication for debugging

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_sae():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 24 * 64
    total_training_tokens = 1_000_000 * 50 # Maybe issue is with reusing tokens?
    total_training_steps = total_training_tokens // batch_size

    cfg = LanguageModelSAERunnerConfig(
        architecture="topk",
        activation_fn_kwargs={"k": 32},
        
        model_name="roneneldan/TinyStories-33M",
        #hook_name="blocks.3.hook_resid_post",
        hook_name="blocks.3.hook_mlp_out",
        hook_layer=3,
        d_in=768,  # TinyStories hidden dim
        
        dataset_path="roneneldan/TinyStories",
        context_size=64,
        training_tokens=total_training_tokens,
        
        expansion_factor=16,
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",
        normalize_sae_decoder=True,
        
        lr=1e-4,
        train_batch_size_tokens=batch_size,
        
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        
        log_to_wandb=True,
        wandb_project="SAELens",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        
        device=device,
        dtype="float32",
        n_checkpoints=5,
        checkpoint_path="checkpoints",
    )
    
    sparse_autoencoder = SAETrainingRunner(cfg).run()
    return sparse_autoencoder

if __name__ == "__main__":
    trained_sae = train_sae()
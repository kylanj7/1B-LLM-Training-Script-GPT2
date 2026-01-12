"""
Training script for 1B parameter LLM from scratch
Uses PyTorch, Transformers, AdamW optimizer, and WandB for logging
60/20/20 train/val/test split on GPU 0
Data loader left empty for custom HuggingFace dataset import
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
import wandb
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Configuration for 1B parameter LLM training"""
    
    # Model architecture (approximately 1B parameters)
    vocab_size = 50257      # GPT-2 tokenizer vocabulary size
    n_positions = 1024      # Maximum sequence length
    n_embd = 1536           # Embedding dimension
    n_layer = 24            # Number of transformer layers
    n_head = 16             # Number of attention heads
    n_inner = 6144          # FFN inner dimension (4 * n_embd)
    
    # Training hyperparameters
    batch_size = 4          # Per-device batch size
    learning_rate = 6e-4    # Peak learning rate
    weight_decay = 0.1      # AdamW weight decay
    max_epochs = 10         # Number of training epochs
    warmup_steps = 2000     # Warmup steps for learning rate
    gradient_accumulation_steps = 8  # Effective batch size = 4 * 8 = 32
    max_grad_norm = 1.0     # Gradient clipping
    
    # Data split ratios
    train_split = 0.6
    val_split = 0.2
    test_split = 0.2
    
    # Device
    device = "cuda:0"       # GPU 0
    
    # Paths
    output_dir = "./outputs"
    checkpoint_dir = "./checkpoints"
    
    # WandB configuration
    use_wandb = True
    wandb_project = "llm-1b-training"
    wandb_run_name = "gpt2-1b-scratch"
    
    # Logging
    log_every_n_steps = 100
    save_every_n_steps = 5000
    eval_every_n_steps = 1000
    
    # Seed for reproducibility
    seed = 42


# ============================================================================
# Custom Dataset Class (Template)
# ============================================================================

class TextDataset(Dataset):
    """
    Template dataset class - replace with your HuggingFace dataset
    
    Example usage with HuggingFace:
        from datasets import load_dataset
        dataset = load_dataset("your_dataset_name")
        # Process and tokenize here
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        """
        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Returns tokenized text with input_ids and attention_mask
        """
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For language modeling
        }


# ============================================================================
# Data Loading Function (TO BE FILLED BY USER)
# ============================================================================

def load_custom_dataset(tokenizer, config: TrainingConfig):
    """
    Load and prepare your custom dataset here.
    
    This function should:
    1. Load your HuggingFace dataset
    2. Preprocess and tokenize the data
    3. Split into train/val/test (60/20/20)
    4. Return DataLoaders
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Training configuration
    
    Returns:
        train_loader, val_loader, test_loader
    
    Example implementation:
    ```python
    from datasets import load_dataset
    
    # Load dataset from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.n_positions,
            padding="max_length"
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Split into train/val/test
    train_size = int(config.train_split * len(tokenized_dataset["train"]))
    val_size = int(config.val_split * len(tokenized_dataset["train"]))
    test_size = len(tokenized_dataset["train"]) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        tokenized_dataset["train"],
        [train_size, val_size, test_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
    ```
    """
    
    # ========================================================================
    # TODO: ADD YOUR DATASET LOADING CODE HERE
    # ========================================================================
    
    raise NotImplementedError(
        "Please implement the load_custom_dataset function with your "
        "HuggingFace dataset. See function docstring for example."
    )
    
    # Placeholder return (remove when implementing)
    return None, None, None


# ============================================================================
# Model Initialization
# ============================================================================

def create_model(config: TrainingConfig) -> GPT2LMHeadModel:
    """
    Create a GPT-2 style model from scratch with ~1B parameters
    
    Args:
        config: Training configuration
    
    Returns:
        Initialized model
    """
    model_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_inner=config.n_inner,
        activation_function="gelu_new",
        resid_pdrop=0.0,        # No dropout (as requested)
        embd_pdrop=0.0,         # No dropout
        attn_pdrop=0.0,         # No dropout
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,        # Disable cache for training
    )
    
    print("Creating model from scratch...")
    model = GPT2LMHeadModel(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params / 1e9:.2f}B parameters")
    
    return model


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    global_step: int
) -> Tuple[float, int]:
    """
    Train for one epoch
    
    Args:
        model: The language model
        train_loader: Training data loader
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        epoch: Current epoch number
        global_step: Current global step
    
    Returns:
        Average loss, updated global step
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item()
        
        # Update weights every gradient_accumulation_steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if global_step % config.log_every_n_steps == 0:
                avg_loss = total_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                if config.use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step
                    })
            
            # Save checkpoint
            if global_step % config.save_every_n_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, config)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    config: TrainingConfig,
    split_name: str = "val"
) -> Dict[str, float]:
    """
    Evaluate the model
    
    Args:
        model: The language model
        eval_loader: Evaluation data loader
        config: Training configuration
        split_name: Name of the split (val or test)
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {split_name}"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += attention_mask.sum().item()
    
    avg_loss = total_loss / len(eval_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    metrics = {
        f"{split_name}/loss": avg_loss,
        f"{split_name}/perplexity": perplexity,
    }
    
    return metrics


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    config: TrainingConfig
):
    """Save model checkpoint"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"checkpoint_step_{global_step}.pt"
    )
    
    torch.save({
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    config: TrainingConfig
) -> int:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    global_step = checkpoint['global_step']
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from step {global_step}")
    
    return global_step


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training function"""
    
    # Configuration
    config = TrainingConfig()
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize WandB
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config)
        )
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    print(f"Using device: {config.device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset (USER MUST IMPLEMENT THIS FUNCTION)
    print("Loading dataset...")
    try:
        train_loader, val_loader, test_loader = load_custom_dataset(tokenizer, config)
    except NotImplementedError:
        print("\n" + "="*80)
        print("ERROR: Dataset loading not implemented!")
        print("Please implement the load_custom_dataset() function")
        print("See function docstring for example implementation")
        print("="*80 + "\n")
        return
    
    # Create model
    model = create_model(config)
    model.to(config.device)
    
    # Create optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
    
    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining configuration:")
    print(f"  Total epochs: {config.max_epochs}")
    print(f"  Batch size per device: {config.batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Peak learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"\nData splits:")
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Validation: {len(val_loader.dataset):,} samples")
    print(f"  Test: {len(test_loader.dataset):,} samples")
    print()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.max_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch, global_step
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, config, split_name="val")
        val_loss = val_metrics["val/loss"]
        val_perplexity = val_metrics["val/perplexity"]
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")
        
        # Log to WandB
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                **val_metrics
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (val_loss: {val_loss:.4f})")
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
    
    test_metrics = evaluate(model, test_loader, config, split_name="test")
    test_loss = test_metrics["test/loss"]
    test_perplexity = test_metrics["test/perplexity"]
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")
    
    if config.use_wandb:
        wandb.log(test_metrics)
        wandb.finish()
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {os.path.join(config.output_dir, 'best_model.pt')}")
    print(f"Final model saved to: {final_model_path}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()

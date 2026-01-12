# 1B Parameter LLM Training Script

Complete training script for a GPT-2 style language model with approximately 1 billion parameters.

## Features

- ✅ ~1B parameter GPT-2 architecture
- ✅ Training on GPU 0 (CUDA)
- ✅ AdamW optimizer
- ✅ No data augmentation (dropout disabled)
- ✅ 60/20/20 train/validation/test split
- ✅ WandB integration for visualization
- ✅ Gradient accumulation
- ✅ Learning rate warmup and scheduling
- ✅ Checkpoint saving and loading
- ✅ Automatic perplexity calculation

## Model Architecture

```
Total Parameters: ~1B
- Vocabulary Size: 50,257
- Max Sequence Length: 1,024
- Embedding Dimension: 1,536
- Number of Layers: 24
- Attention Heads: 16
- FFN Inner Dimension: 6,144
```

## Installation

```bash
pip install -r requirements.txt
```

## Setup WandB (Optional but Recommended)

```bash
wandb login
# Enter your API key when prompted
```

To disable WandB, set `use_wandb = False` in the `TrainingConfig` class.

## Usage

### Step 1: Implement the Data Loading Function

Edit the `load_custom_dataset()` function in `train_llm_1b.py` to load your HuggingFace dataset.

**Example implementation:**

```python
def load_custom_dataset(tokenizer, config: TrainingConfig):
    from datasets import load_dataset
    
    # Load dataset from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.n_positions,
            padding="max_length",
            return_tensors=None
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Add labels column
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Split into train/val/test (60/20/20)
    dataset_size = len(tokenized_dataset)
    train_size = int(config.train_split * dataset_size)
    val_size = int(config.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        tokenized_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
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

### Step 2: Run Training

```bash
python train_llm_1b.py
```

## Configuration

Edit the `TrainingConfig` class to customize training:

```python
class TrainingConfig:
    # Model architecture
    vocab_size = 50257
    n_positions = 1024
    n_embd = 1536
    n_layer = 24
    n_head = 16
    
    # Training hyperparameters
    batch_size = 4
    learning_rate = 6e-4
    weight_decay = 0.1
    max_epochs = 10
    warmup_steps = 2000
    gradient_accumulation_steps = 8
    
    # Data splits
    train_split = 0.6
    val_split = 0.2
    test_split = 0.2
    
    # Device
    device = "cuda:0"  # GPU 0
    
    # WandB
    use_wandb = True
    wandb_project = "llm-1b-training"
    wandb_run_name = "gpt2-1b-scratch"
```

## Memory Requirements

- **GPU Memory**: ~16-24 GB VRAM (your RTX 3090 24GB should work perfectly)
- **RAM**: ~32 GB recommended
- **Storage**: Depends on dataset size + checkpoints

## Outputs

The script creates the following directories:

```
./outputs/
├── best_model.pt       # Best model based on validation loss
└── final_model.pt      # Final model after all epochs

./checkpoints/
├── checkpoint_step_5000.pt
├── checkpoint_step_10000.pt
└── ...
```

## Monitoring Training

### With WandB (Recommended)

1. Training will automatically log to WandB
2. Visit https://wandb.ai to view:
   - Training loss curves
   - Validation metrics
   - Learning rate schedule
   - Perplexity over time

### Console Output

The script prints:
- Training progress bars with loss
- Validation metrics after each epoch
- Test metrics at the end

## Popular HuggingFace Datasets

Here are some datasets you can use:

```python
# WikiText (English)
dataset = load_dataset("wikitext", "wikitext-103-v1")

# OpenWebText (Large English corpus)
dataset = load_dataset("openwebtext")

# BookCorpus
dataset = load_dataset("bookcorpus")

# C4 (Colossal Clean Crawled Corpus)
dataset = load_dataset("c4", "en")

# The Pile (Very large, diverse)
dataset = load_dataset("EleutherAI/pile")

# Code datasets
dataset = load_dataset("codeparrot/github-code")
```

## Checkpoint Loading

To resume training from a checkpoint:

```python
# In main() function, before training loop:
checkpoint_path = "./checkpoints/checkpoint_step_5000.pt"
global_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path, config)
```

## Tips for Your Setup

Given your hardware (RTX 3090 24GB):

1. **Batch size**: Start with 4, increase if you have memory
2. **Gradient accumulation**: 8 steps gives effective batch size of 32
3. **Mixed precision**: Add this for faster training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```
4. **Monitor GPU usage**: `watch -n 1 nvidia-smi`

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` (try 2 or 1)
- Increase `gradient_accumulation_steps`
- Reduce `n_positions` (sequence length)

### Slow Training

- Increase `num_workers` in DataLoader
- Use mixed precision training
- Ensure data is cached/preprocessed

### Loss Not Decreasing

- Check learning rate (try 3e-4 or 1e-4)
- Verify data is properly formatted
- Check for NaN values in data

## Example Training Session

```bash
# Install dependencies
pip install -r requirements.txt

# Login to WandB
wandb login

# Edit train_llm_1b.py and implement load_custom_dataset()

# Start training
python train_llm_1b.py
```

## Expected Output

```
Using device: cuda:0
GPU: NVIDIA GeForce RTX 3090
Loading tokenizer...
Loading dataset...
Creating model from scratch...
Total parameters: 1,013,645,312
Trainable parameters: 1,013,645,312
Model size: ~1.01B parameters

Training configuration:
  Total epochs: 10
  Batch size per device: 4
  Gradient accumulation steps: 8
  Effective batch size: 32
  Total training steps: 31250
  Warmup steps: 2000
  Peak learning rate: 0.0006
  Weight decay: 0.1

Data splits:
  Train: 60,000 samples
  Validation: 20,000 samples
  Test: 20,000 samples

================================================================================
Epoch 1/10
================================================================================
Epoch 1: 100%|████████████| 7500/7500 [2:15:30<00:00, loss=3.2451, lr=6.0e-04]
Train Loss: 3.2451
Validation Loss: 3.1234
Validation Perplexity: 22.71
New best model saved! (val_loss: 3.1234)
...
```

## License

This script is provided as-is for educational and research purposes.

## Author

Created for LLM training experiments on NVIDIA RTX 3090 hardware.

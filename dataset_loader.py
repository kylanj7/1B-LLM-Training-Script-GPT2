"""
Example implementation of load_custom_dataset() function
Using WikiText-103 dataset from HuggingFace

Copy this function into train_llm_1b.py to replace the placeholder function
"""

import torch
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset

def load_custom_dataset(tokenizer, config):
    """
    Load WikiText-103 dataset from HuggingFace
    
    Args:
        tokenizer: GPT2Tokenizer
        config: TrainingConfig instance
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading WikiText-103 dataset from HuggingFace...")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Filter out empty texts
    def filter_empty(example):
        return len(example["text"].strip()) > 0
    
    dataset = dataset.filter(filter_empty)
    
    print(f"Dataset loaded: {len(dataset['train'])} training examples")
    
    # Tokenization function
    def tokenize_function(examples):
        """Tokenize a batch of examples"""
        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.n_positions,
            padding="max_length",
            return_tensors=None  # Return lists, not tensors
        )
        
        # Add labels (same as input_ids for language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Tokenize the dataset
    print("Tokenizing dataset... (this may take a few minutes)")
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    print(f"Tokenization complete: {len(tokenized_dataset)} samples")
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # Split into train/val/test (60/20/20)
    dataset_size = len(tokenized_dataset)
    train_size = int(config.train_split * dataset_size)
    val_size = int(config.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {train_size} samples ({config.train_split*100}%)")
    print(f"  Validation: {val_size} samples ({config.val_split*100}%)")
    print(f"  Test: {test_size} samples ({config.test_split*100}%)")
    
    # Use random_split with fixed seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        tokenized_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Create DataLoaders
    print("\nCreating DataLoaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print("DataLoaders created successfully!\n")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Alternative: Load from local text files
# ============================================================================

def load_custom_dataset_from_files(tokenizer, config, train_file, val_file, test_file):
    """
    Load dataset from local text files
    
    Args:
        tokenizer: GPT2Tokenizer
        config: TrainingConfig instance
        train_file: Path to training text file
        val_file: Path to validation text file
        test_file: Path to test text file
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import Dataset
    
    class TextFileDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.texts = f.readlines()
            
            # Filter empty lines
            self.texts = [t.strip() for t in self.texts if t.strip()]
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
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
                "labels": input_ids.clone()
            }
    
    # Create datasets
    train_dataset = TextFileDataset(train_file, tokenizer, config.n_positions)
    val_dataset = TextFileDataset(val_file, tokenizer, config.n_positions)
    test_dataset = TextFileDataset(test_file, tokenizer, config.n_positions)
    
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


# ============================================================================
# Alternative: Other popular HuggingFace datasets
# ============================================================================

def load_openwebtext(tokenizer, config):
    """Load OpenWebText dataset"""
    dataset = load_dataset("openwebtext")
    # ... (similar implementation as above)
    pass

def load_c4_dataset(tokenizer, config):
    """Load C4 dataset"""
    dataset = load_dataset("c4", "en", streaming=True)  # Use streaming for large datasets
    # ... (implementation with streaming)
    pass

def load_pile_dataset(tokenizer, config):
    """Load The Pile dataset"""
    dataset = load_dataset("EleutherAI/pile", streaming=True)
    # ... (implementation with streaming)
    passd

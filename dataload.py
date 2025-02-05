import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from model import load_config
from torch.utils.data import DataLoader

# ---------------------- Data Preparation Utilities ----------------------

def prepare_tinystories_split(filepath: str, split: str = "train", batch_size: int = 1000) -> None:
    """
    Prepares TinyStories dataset split (train/validation) as a text file.
    Processes data in batches to avoid memory issues.
    """
    if not os.path.exists(filepath):
        print(f"Creating TinyStories {split} split at '{filepath}'...")
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for batch in dataset.iter(batch_size=batch_size):
                texts = batch['text']
                f.write('\n'.join(texts) + '\n')
        print(f"Created '{filepath}'.")
    else:
        print(f"'{filepath}' already exists. Skipping creation.")


# ---------------------- Core Dataset Class ----------------------

class TokenizedDataset(Dataset):
    """
    Memory-efficient dataset for tokenized text blocks from a file.
    Converts data directly into PyTorch tensors for default DataLoader compatibility.
    """
    
    def __init__(
        self,
        filepath: str,
        tokenizer,
        block_size: int,
        use_data_fraction: float = 1.0,
        buffer_size: int = 10000,
        split: str = "train"
    ):
        super().__init__()
        
        # Validate file existence
        if not os.path.exists(filepath):
            print(f"{filepath} not found...")
            print("Creating TinyStories train split...")
            if split == "train": 
                dataset = prepare_tinystories_split(filepath, split="train")
            elif split == "validation":
                dataset = prepare_tinystories_split(filepath, split="validation")
            else:
                raise ValueError(f"Invalid split: {split}")

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size

        # Read and possibly truncate text
        with open(self.filepath, "r", encoding="utf-8") as f:
            full_text = f.read()

        total_chars = len(full_text)
        keep_chars = int(total_chars * use_data_fraction)
        if keep_chars < total_chars:
            full_text = full_text[:keep_chars]

        # Tokenize
        self.input_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # How many blocks fit?
        self.total_tokens = len(self.input_ids)
        self.num_blocks = self.total_tokens // self.block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        block_tokens = self.input_ids[start:end]  # [t_0, t_1, ..., t_{n-1}]

        # Convert to tensors (shifted for next-token prediction)
        input_ids = torch.tensor(block_tokens[:-1], dtype=torch.long)  # [t_0, ..., t_{block_size-2}]
        labels = torch.tensor(block_tokens[1:], dtype=torch.long)     # [t_1, ..., t_{block_size-1}]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


# ---------------------- Main Execution (for testing) ----------------------

if __name__ == "__main__":
    config_path = "config/tiny_lm_1M.config"
    config = load_config(config_path)
    print(f"[INFO] Loaded config from {config_path}")

    train_text_path = "data/raw/train.txt"
    valid_text_path = "data/raw/validation.txt"

    prepare_tinystories_split(train_text_path, split="train")
    prepare_tinystories_split(valid_text_path, split="validation")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    print("[INFO] Initialized tokenizer")

    train_dataset = TokenizedDataset(
        filepath=train_text_path,
        tokenizer=tokenizer,
        use_data_fraction=config.use_data_fraction,
        block_size=config.block_size,
        split="train"
    )

    print(f"[INFO] #blocks in train_dataset: {len(train_dataset)}")
    sample = train_dataset[0]
    print("Sample item (as tensors):")
    print(" - input_ids[:10]:", sample["input_ids"][:10])
    print(" - labels[:10]:   ", sample["labels"][:10])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    batch = next(iter(train_loader))
    print("Batch shapes:")
    print(" - input_ids.shape =", batch["input_ids"].shape)
    print(" - labels.shape    =", batch["labels"].shape)

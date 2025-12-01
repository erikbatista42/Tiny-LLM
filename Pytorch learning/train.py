"""
Training Script for TinyLLM
===========================

This script handles:
1. Loading and preparing text data
2. Creating training batches
3. The training loop
4. Saving/loading checkpoints
5. Generating sample text during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import math

from model import TinyLLM, CharacterTokenizer, create_causal_mask


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_text_data(file_path: str) -> str:
    """Load text from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_sample_data() -> str:
    """
    Sample text for quick testing.
    For real training, you'd use a larger corpus like:
    - Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """
    return """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep: perchance to dream: ay, there's the rub;
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life.
    
    All the world's a stage,
    And all the men and women merely players;
    They have their exits and their entrances,
    And one man in his time plays many parts.
    
    Friends, Romans, countrymen, lend me your ears;
    I come to bury Caesar, not to praise him.
    The evil that men do lives after them;
    The good is oft interred with their bones.
    """ * 10  # Repeat for more training data


class TextDataset:
    """
    Creates training examples from text.
    
    Each example is a sequence of characters where:
    - Input: characters [0:seq_len]
    - Target: characters [1:seq_len+1] (shifted by 1)
    
    This teaches the model to predict the next character.
    """
    
    def __init__(self, text: str, tokenizer: CharacterTokenizer, seq_len: int):
        """
        TODO 25: Initialize the dataset
        
        Store:
        - self.tokenizer
        - self.seq_len
        - self.data: the full text encoded as a tensor
        """
        # YOUR CODE HERE
        self.tokenizer = None
        self.seq_len = None
        self.data = None
    
    def __len__(self) -> int:
        """
        TODO 26: Return the number of possible sequences
        
        We can start a sequence at any position from 0 to len(data) - seq_len - 1
        (need +1 for the target)
        """
        # YOUR CODE HERE
        pass
    
    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO 27: Get a random batch of training examples
        
        Steps:
        1. Randomly sample batch_size starting positions
        2. For each position, extract:
           - x: data[pos : pos + seq_len]
           - y: data[pos + 1 : pos + seq_len + 1]
        3. Stack into batches
        
        Returns:
        - x: (batch_size, seq_len) - input sequences
        - y: (batch_size, seq_len) - target sequences
        
        Hint: torch.randint(low, high, (size,)) for random positions
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_step(
    model: TinyLLM,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mask: torch.Tensor = None
) -> float:
    """
    TODO 28: Implement a single training step
    
    Steps:
    1. Zero the gradients (optimizer.zero_grad())
    2. Forward pass: get logits from model
    3. Compute loss:
       - Reshape logits to (batch * seq_len, vocab_size)
       - Reshape targets to (batch * seq_len,)
       - Use F.cross_entropy(logits, targets)
    4. Backward pass: loss.backward()
    5. Update weights: optimizer.step()
    6. Return the loss value (use loss.item())
    
    Hint for reshaping:
    - logits.view(-1, logits.size(-1)) flattens batch and seq dims
    - y.view(-1) flattens target
    """
    import torch.nn.functional as F
    
    # YOUR CODE HERE
    pass


def evaluate(
    model: TinyLLM,
    dataset: TextDataset,
    num_batches: int = 10,
    batch_size: int = 32
) -> float:
    """
    TODO 29: Evaluate model on random batches
    
    Steps:
    1. Set model to eval mode: model.eval()
    2. Use torch.no_grad() context
    3. Average loss over num_batches
    4. Set model back to train mode: model.train()
    
    Return the average loss.
    """
    import torch.nn.functional as F
    
    # YOUR CODE HERE
    pass


def train(
    model: TinyLLM,
    dataset: TextDataset,
    tokenizer: CharacterTokenizer,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    eval_interval: int = 100,
    generate_interval: int = 500,
    checkpoint_dir: str = "checkpoints"
):
    """
    TODO 30: Implement the full training loop
    
    Steps:
    1. Create optimizer: AdamW(model.parameters(), lr=learning_rate)
    2. Calculate steps_per_epoch = len(dataset) // batch_size
    3. Create checkpoint directory if needed
    
    For each epoch:
        For each step:
            a. Get batch from dataset
            b. Perform train_step
            c. Every eval_interval steps: print loss, evaluate
            d. Every generate_interval steps: generate sample text
        
        Save checkpoint after each epoch
    
    Hints:
    - Use Path(checkpoint_dir).mkdir(exist_ok=True) for directory
    - torch.save({'model': model.state_dict(), 'epoch': epoch}, path)
    """
    # YOUR CODE HERE
    pass


def generate_sample(model: TinyLLM, tokenizer: CharacterTokenizer, prompt: str = "To be", length: int = 100) -> str:
    """
    TODO 31: Generate text from a prompt
    
    Steps:
    1. Encode the prompt using tokenizer
    2. Add batch dimension: unsqueeze(0)
    3. Call model.generate()
    4. Decode the output
    
    Remember to set model.eval() and use torch.no_grad()
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main training script - run this after completing all TODOs
    """
    print("=" * 60)
    print("TinyLLM Training")
    print("=" * 60)
    
    # Hyperparameters
    SEQ_LEN = 64
    BATCH_SIZE = 32
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 5
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    text = get_sample_data()
    print(f"Text length: {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, SEQ_LEN)
    print(f"Number of sequences: {len(dataset):,}")
    
    # Create model
    print("\nCreating model...")
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Generate before training
    print("\n--- Sample BEFORE training ---")
    sample = generate_sample(model, tokenizer, "To be", 100)
    print(sample)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    train(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Generate after training
    print("\n--- Sample AFTER training ---")
    sample = generate_sample(model, tokenizer, "To be", 200)
    print(sample)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
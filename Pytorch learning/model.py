"""
Tiny LLM: A Character-Level Transformer Language Model
======================================================

This is a learning project to understand PyTorch and transformer architecture
by building a small language model from scratch.

We'll build a decoder-only transformer (like GPT) that predicts the next
character given a sequence of characters.

Architecture Overview:
    Input Text → Token Embeddings + Positional Embeddings 
               → N x Transformer Blocks 
               → Linear Output Head 
               → Next Character Probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# PHASE 1: PYTORCH FUNDAMENTALS
# =============================================================================

class Phase1_TensorBasics:
    """
    Before building the model, let's get comfortable with PyTorch tensors.
    Tensors are like NumPy arrays, but they can run on GPU and track gradients.
    
    Complete each TODO to build your intuition.
    """
    
    @staticmethod
    def exercise_1_creating_tensors():
        """
        Your tasks:
        a) Create a tensor from a Python list: [1, 2, 3, 4, 5]
        b) Create a 3x3 tensor of zeros
        c) Create a 2x4 tensor of random values between 0 and 1
        d) Create a 1D tensor with values from 0 to 9 (hint: torch.arange)
        
        Return them as a tuple: (a, b, c, d)
        """
        # YOUR CODE HERE
        a = torch.tensor([1, 2, 3, 4, 5])
        b = torch.zeros(3,3)
        c = torch.rand(2,4)
        d = torch.arange(10)     
        
        return a, b, c, d
    
    @staticmethod
    def exercise_2_tensor_operations():
        """
        TODO 2: Practice tensor operations
        
        Given two tensors x and y (created below), perform these operations:
        a) Element-wise addition: x + y
        b) Matrix multiplication: x @ y.T (y transposed)
        c) Element-wise multiplication: x * y
        d) Sum all elements in x
        e) Find the mean of each row in x (hint: use dim parameter)
        
        Return them as a tuple: (a, b, c, d, e)
        """
        x = torch.tensor([[1., 2., 3.], 
                          [4., 5., 6.]])  # Shape: (2, 3)
        y = torch.tensor([[7., 8., 9.], 
                          [10., 11., 12.]])  # Shape: (2, 3)
        
        # YOUR CODE HERE
        a = torch.add(x,y)
        b = torch.matmul(x,y.T)
        c = torch.multiply(x,y)
        d = x.sum()
        e = x.mean(dim=1)
        
        return a, b, c, d, e
    
    @staticmethod
    def exercise_3_reshaping():
        """
        TODO 3: Reshape tensors - crucial for transformer operations!

        Given tensor x of shape (2, 6), perform these reshaping operations:
        a) Reshape to (3, 4)
        b) Reshape to (2, 2, 3) - adding a dimension
        c) Flatten to 1D
        d) Add a dimension at position 0: (2, 6) → (1, 2, 6) (hint: unsqueeze)
        e) Transpose dimensions 0 and 1 of a (2, 3, 4) tensor

        Return them as a tuple: (a, b, c, d, e)
        """
        x = torch.arange(12).reshape(2, 6).float()
        x_3d = torch.arange(24).reshape(2, 3, 4).float()

        # YOUR CODE HERE
        a = x.reshape(3, 4)
        b = x.reshape(2, 2, 3)
        c = x.flatten()
        d = x.unsqueeze(0)
        e = x_3d.transpose(0, 1)

        return a, b, c, d, e


# =============================================================================
# PHASE 2: BUILDING NEURAL NETWORK MODULES
# =============================================================================

class Phase2_NNModuleBasics(nn.Module):
    """
    In PyTorch, we build neural networks by subclassing nn.Module.
    
    Key concepts:
    - __init__: Define your layers here
    - forward: Define how data flows through your layers
    - Parameters are automatically tracked when you use nn.Linear, nn.Embedding, etc.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        TODO 4: Initialize a simple feed-forward network
        
        Create a network with:
        - A linear layer from input_size to hidden_size (call it self.fc1)
        - A linear layer from hidden_size to output_size (call it self.fc2)
        
        Hints:
        - Always call super().__init__() first
        - Use nn.Linear(in_features, out_features)
        """
        super().__init__()
        
        # YOUR CODE HERE
        self.fc1 = None
        self.fc2 = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO 5: Define the forward pass
        
        Pass x through:
        1. self.fc1
        2. ReLU activation (use F.relu)
        3. self.fc2
        
        Return the output tensor.
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# PHASE 3: TRANSFORMER COMPONENTS
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1 Embeddings
# -----------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """
    Converts token indices into dense vectors.
    
    Example: If vocab_size=100 and embed_dim=64, this creates a lookup table
    of 100 vectors, each with 64 dimensions. Token index 5 retrieves row 5.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        TODO 6: Create an embedding layer
        
        Use nn.Embedding(num_embeddings, embedding_dim)
        """
        super().__init__()
        
        # YOUR CODE HERE
        self.embedding = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO 7: Apply the embedding
        
        Input x: tensor of token indices, shape (batch_size, seq_len)
        Output: tensor of embeddings, shape (batch_size, seq_len, embed_dim)
        """
        # YOUR CODE HERE
        pass


class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings.
    
    Transformers process all positions in parallel, so they need explicit
    position information. We use sinusoidal encodings (from the original paper):
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This creates unique patterns for each position that the model can learn from.
    """
    
    def __init__(self, embed_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        TODO 8: Initialize positional encoding
        
        Steps:
        1. Create a zeros tensor of shape (max_seq_len, embed_dim) for storing PE
        2. Create position indices: shape (max_seq_len, 1), values 0 to max_seq_len-1
        3. Create dimension indices for the formula (the 'i' values)
        4. Compute the division term: 10000^(2i/embed_dim)
        5. Apply sin to even indices, cos to odd indices
        6. Register as a buffer (not a parameter) using self.register_buffer('pe', pe)
        7. Create dropout layer
        
        Hint: Use torch.arange, torch.exp, torch.log for the computations
        """
        super().__init__()
        
        # YOUR CODE HERE
        # Step 1: Create empty PE matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        
        # Step 2: Position indices (0, 1, 2, ..., max_seq_len-1)
        position = None  # Shape: (max_seq_len, 1)
        
        # Step 3-4: Division term using log for numerical stability
        # div_term = exp(arange(0, embed_dim, 2) * -(log(10000.0) / embed_dim))
        div_term = None  # Shape: (embed_dim/2,)
        
        # Step 5: Apply sin and cos
        # pe[:, 0::2] = sin(position * div_term)  # Even indices
        # pe[:, 1::2] = cos(position * div_term)  # Odd indices
        
        # Step 6: Register buffer (adds unsqueeze(0) for batch dimension)
        # self.register_buffer('pe', pe.unsqueeze(0))
        
        # Step 7: Dropout
        self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO 9: Add positional encoding to input embeddings
        
        Input x: (batch_size, seq_len, embed_dim)
        Output: x + positional_encoding, then apply dropout
        
        Hint: Use self.pe[:, :x.size(1), :] to get the right sequence length
        """
        # YOUR CODE HERE
        pass


# -----------------------------------------------------------------------------
# 3.2 Attention Mechanism - THE HEART OF TRANSFORMERS
# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Single-head self-attention mechanism.
    
    The key insight: attention lets each position "look at" other positions
    and decide how much to attend to them.
    
    For each position, we compute:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What information do I provide?"
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
    
    The softmax(QK^T / sqrt(d_k)) gives us attention weights - how much each
    position should attend to every other position.
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        """
        TODO 10: Initialize self-attention
        
        Create:
        - self.query: Linear layer (embed_dim → embed_dim)
        - self.key: Linear layer (embed_dim → embed_dim)  
        - self.value: Linear layer (embed_dim → embed_dim)
        - self.dropout: Dropout layer
        - self.scale: Store sqrt(embed_dim) for scaling
        """
        super().__init__()
        
        # YOUR CODE HERE
        self.query = None
        self.key = None
        self.value = None
        self.dropout = None
        self.scale = None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        TODO 11: Implement self-attention forward pass
        
        Steps:
        1. Compute Q, K, V by passing x through respective linear layers
        2. Compute attention scores: Q @ K^T / scale
        3. Apply mask if provided (for causal/autoregressive attention)
           - Where mask is 0, set scores to -infinity (use float('-inf'))
        4. Apply softmax to get attention weights
        5. Apply dropout to attention weights
        6. Compute output: attention_weights @ V
        
        Input x: (batch_size, seq_len, embed_dim)
        Output: (batch_size, seq_len, embed_dim)
        
        Hints:
        - For K^T, use k.transpose(-2, -1) to transpose last two dims
        - For masking: scores = scores.masked_fill(mask == 0, float('-inf'))
        """
        # YOUR CODE HERE
        pass


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention runs multiple attention operations in parallel.
    
    Why multiple heads? Each head can learn to attend to different types
    of information (syntax, semantics, nearby words, etc.)
    
    We split embed_dim into num_heads, run attention on each, then concatenate.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        TODO 12: Initialize multi-head attention
        
        Requirements:
        - embed_dim must be divisible by num_heads
        - head_dim = embed_dim // num_heads
        
        Create:
        - self.num_heads, self.head_dim: store these values
        - self.query, self.key, self.value: Linear(embed_dim, embed_dim)
        - self.output_proj: Linear(embed_dim, embed_dim) - combines heads
        - self.dropout: Dropout layer
        - self.scale: sqrt(head_dim) for scaling
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # YOUR CODE HERE
        self.num_heads = num_heads
        self.head_dim = None
        
        self.query = None
        self.key = None
        self.value = None
        self.output_proj = None
        self.dropout = None
        self.scale = None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        TODO 13: Implement multi-head attention forward pass
        
        Steps:
        1. Get batch_size, seq_len from x.shape
        2. Compute Q, K, V using linear layers
        3. Reshape for multi-head: (batch, seq, embed) → (batch, seq, heads, head_dim)
        4. Transpose to: (batch, heads, seq, head_dim)
        5. Compute attention scores: Q @ K^T / scale
        6. Apply mask if provided
        7. Softmax and dropout
        8. Compute attention output: weights @ V
        9. Transpose back: (batch, heads, seq, head_dim) → (batch, seq, heads, head_dim)
        10. Reshape: (batch, seq, embed_dim)
        11. Apply output projection
        
        Hints:
        - reshape: tensor.view(batch, seq, num_heads, head_dim)
        - transpose: tensor.transpose(1, 2) swaps dims 1 and 2
        - For step 10, use .contiguous().view(batch, seq, -1)
        """
        batch_size, seq_len, _ = x.shape
        
        # YOUR CODE HERE
        pass


# -----------------------------------------------------------------------------
# 3.3 Feed-Forward Network
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Applied independently to each position. Typically expands to 4x the
    dimension, applies activation, then projects back down.
    
    FFN(x) = Linear(GELU(Linear(x)))
    """
    
    def __init__(self, embed_dim: int, ff_dim: int = None, dropout: float = 0.1):
        """
        TODO 14: Initialize feed-forward network
        
        Create:
        - self.fc1: Linear(embed_dim, ff_dim)  [expansion]
        - self.fc2: Linear(ff_dim, embed_dim)  [projection]
        - self.dropout: Dropout layer
        
        If ff_dim is None, default to 4 * embed_dim
        """
        super().__init__()
        
        ff_dim = ff_dim or 4 * embed_dim
        
        # YOUR CODE HERE
        self.fc1 = None
        self.fc2 = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO 15: Implement feed-forward forward pass
        
        x → fc1 → GELU → dropout → fc2 → dropout
        
        Use F.gelu for the activation
        """
        # YOUR CODE HERE
        pass


# -----------------------------------------------------------------------------
# 3.4 Transformer Block
# -----------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single transformer block combining attention and feed-forward.
    
    Uses pre-norm architecture (more stable training):
    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = None, dropout: float = 0.1):
        """
        TODO 16: Initialize transformer block
        
        Create:
        - self.attention: MultiHeadAttention
        - self.feed_forward: FeedForward
        - self.norm1: LayerNorm(embed_dim)
        - self.norm2: LayerNorm(embed_dim)
        - self.dropout: Dropout layer
        """
        super().__init__()
        
        # YOUR CODE HERE
        self.attention = None
        self.feed_forward = None
        self.norm1 = None
        self.norm2 = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        TODO 17: Implement transformer block forward pass
        
        Pre-norm residual connections:
        1. attention_out = self.attention(self.norm1(x), mask)
        2. x = x + self.dropout(attention_out)
        3. ff_out = self.feed_forward(self.norm2(x))
        4. x = x + self.dropout(ff_out)
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# PHASE 4: THE COMPLETE MODEL
# =============================================================================

class TinyLLM(nn.Module):
    """
    Complete character-level transformer language model.
    
    Architecture:
    - Token embedding + Positional encoding
    - N transformer blocks
    - Final layer norm
    - Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 256,
        ff_dim: int = None,
        dropout: float = 0.1
    ):
        """
        TODO 18: Initialize the complete model
        
        Create:
        - self.token_embedding: TokenEmbedding
        - self.pos_encoding: PositionalEncoding
        - self.blocks: nn.ModuleList of TransformerBlock layers
        - self.norm: Final LayerNorm
        - self.output_head: Linear(embed_dim, vocab_size)
        
        Important: Use nn.ModuleList for the transformer blocks!
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # YOUR CODE HERE
        self.token_embedding = None
        self.pos_encoding = None
        self.blocks = None
        self.norm = None
        self.output_head = None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        TODO 19: Implement the complete forward pass
        
        Steps:
        1. Token embedding
        2. Add positional encoding
        3. Pass through each transformer block
        4. Apply final layer norm
        5. Project to vocabulary size
        
        Input x: (batch_size, seq_len) - token indices
        Output: (batch_size, seq_len, vocab_size) - logits for each position
        """
        # YOUR CODE HERE
        pass
    
    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        TODO 20: Implement text generation
        
        Autoregressive generation: predict one token at a time, append to sequence.
        
        Steps (in a loop for max_new_tokens iterations):
        1. Crop sequence to max_seq_len if needed
        2. Get model predictions (forward pass)
        3. Take logits for the last position only
        4. Divide by temperature (higher = more random)
        5. Apply softmax to get probabilities
        6. Sample next token (use torch.multinomial)
        7. Append to sequence
        
        Return the full sequence including generated tokens.
        
        Hints:
        - Use torch.no_grad() context manager (no gradients needed for generation)
        - torch.multinomial(probs, num_samples=1) samples from distribution
        - torch.cat([sequence, new_token], dim=1) appends tokens
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# PHASE 5: TRAINING UTILITIES
# =============================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    TODO 21: Create a causal (autoregressive) attention mask
    
    In language modeling, position i should only attend to positions 0...i
    (can't look into the future!)
    
    Create a lower triangular matrix of ones:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    
    Hint: torch.tril(torch.ones(seq_len, seq_len))
    """
    # YOUR CODE HERE
    pass


class CharacterTokenizer:
    """
    Simple character-level tokenizer.
    
    Maps each unique character to an integer index.
    """
    
    def __init__(self, text: str):
        """
        TODO 22: Initialize tokenizer from text
        
        Create:
        - self.chars: sorted list of unique characters
        - self.vocab_size: number of unique characters
        - self.char_to_idx: dict mapping char → index
        - self.idx_to_char: dict mapping index → char
        """
        # YOUR CODE HERE
        self.chars = None
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None
    
    def encode(self, text: str) -> torch.Tensor:
        """
        TODO 23: Convert text to tensor of indices
        """
        # YOUR CODE HERE
        pass
    
    def decode(self, indices: torch.Tensor) -> str:
        """
        TODO 24: Convert tensor of indices back to text
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# TESTING YOUR IMPLEMENTATION
# =============================================================================

def test_phase1():
    """Test Phase 1: Tensor basics"""
    print("Testing Phase 1: Tensor Basics")
    print("=" * 50)
    
    # Test exercise 1
    a, b, c, d = Phase1_TensorBasics.exercise_1_creating_tensors()
    assert a is not None, "Exercise 1a not implemented"
    assert list(a.shape) == [5], f"Exercise 1a: expected shape [5], got {list(a.shape)}"
    assert list(b.shape) == [3, 3], f"Exercise 1b: expected shape [3,3], got {list(b.shape)}"
    assert list(c.shape) == [2, 4], f"Exercise 1c: expected shape [2,4], got {list(c.shape)}"
    assert list(d.shape) == [10], f"Exercise 1d: expected shape [10], got {list(d.shape)}"
    print("✓ Exercise 1 passed!")


    # Test exercise 2
    results = Phase1_TensorBasics.exercise_2_tensor_operations()
    assert all(r is not None for r in results), "Exercise 2 not fully implemented"
    print("✓ Exercise 2 passed!")
    return 
    
    # Test exercise 3
    results = Phase1_TensorBasics.exercise_3_reshaping()
    assert all(r is not None for r in results), "Exercise 3 not fully implemented"
    a, b, c, d, e = results
    assert list(a.shape) == [3, 4], f"Exercise 3a: expected [3,4], got {list(a.shape)}"
    assert list(b.shape) == [2, 2, 3], f"Exercise 3b: expected [2,2,3], got {list(b.shape)}"
    assert list(c.shape) == [12], f"Exercise 3c: expected [12], got {list(c.shape)}"
    assert list(d.shape) == [1, 2, 6], f"Exercise 3d: expected [1,2,6], got {list(d.shape)}"
    assert list(e.shape) == [3, 2, 4], f"Exercise 3e: expected [3,2,4], got {list(e.shape)}"
    print("✓ Exercise 3 passed!")
    
    print("\n✅ Phase 1 complete!\n")


def test_phase2():
    """Test Phase 2: nn.Module basics"""
    print("Testing Phase 2: nn.Module Basics")
    print("=" * 50)
    
    model = Phase2_NNModuleBasics(input_size=10, hidden_size=20, output_size=5)
    assert model.fc1 is not None, "fc1 not initialized"
    assert model.fc2 is not None, "fc2 not initialized"
    
    x = torch.randn(3, 10)  # batch of 3, input size 10
    output = model(x)
    assert output is not None, "forward not implemented"
    assert list(output.shape) == [3, 5], f"Expected shape [3,5], got {list(output.shape)}"
    
    print("✓ Phase 2 passed!")
    print("\n✅ Phase 2 complete!\n")


def test_phase3():
    """Test Phase 3: Transformer components"""
    print("Testing Phase 3: Transformer Components")
    print("=" * 50)
    
    batch_size, seq_len, embed_dim = 2, 8, 64
    num_heads = 4
    vocab_size = 100
    
    # Test TokenEmbedding
    tok_emb = TokenEmbedding(vocab_size, embed_dim)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    emb = tok_emb(x)
    assert emb is not None, "TokenEmbedding forward not implemented"
    assert list(emb.shape) == [batch_size, seq_len, embed_dim]
    print("✓ TokenEmbedding passed!")
    
    # Test PositionalEncoding
    pos_enc = PositionalEncoding(embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = pos_enc(x)
    assert out is not None, "PositionalEncoding forward not implemented"
    assert list(out.shape) == [batch_size, seq_len, embed_dim]
    print("✓ PositionalEncoding passed!")
    
    # Test SelfAttention
    attn = SelfAttention(embed_dim)
    out = attn(x)
    assert out is not None, "SelfAttention forward not implemented"
    assert list(out.shape) == [batch_size, seq_len, embed_dim]
    print("✓ SelfAttention passed!")
    
    # Test MultiHeadAttention
    mha = MultiHeadAttention(embed_dim, num_heads)
    out = mha(x)
    assert out is not None, "MultiHeadAttention forward not implemented"
    assert list(out.shape) == [batch_size, seq_len, embed_dim]
    print("✓ MultiHeadAttention passed!")
    
    # Test FeedForward
    ff = FeedForward(embed_dim)
    out = ff(x)
    assert out is not None, "FeedForward forward not implemented"
    assert list(out.shape) == [batch_size, seq_len, embed_dim]
    print("✓ FeedForward passed!")
    
    # Test TransformerBlock
    block = TransformerBlock(embed_dim, num_heads)
    out = block(x)
    assert out is not None, "TransformerBlock forward not implemented"
    assert list(out.shape) == [batch_size, seq_len, embed_dim]
    print("✓ TransformerBlock passed!")
    
    print("\n✅ Phase 3 complete!\n")


def test_phase4():
    """Test Phase 4: Complete model"""
    print("Testing Phase 4: Complete Model")
    print("=" * 50)
    
    vocab_size = 65
    batch_size, seq_len = 2, 32
    
    model = TinyLLM(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(x)
    assert logits is not None, "TinyLLM forward not implemented"
    assert list(logits.shape) == [batch_size, seq_len, vocab_size]
    print("✓ Forward pass works!")
    
    # Test generation
    start = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(start, max_new_tokens=10)
    assert generated is not None, "generate not implemented"
    assert generated.shape[1] == 15, f"Expected length 15, got {generated.shape[1]}"
    print("✓ Generation works!")
    
    print("\n✅ Phase 4 complete!\n")


def test_phase5():
    """Test Phase 5: Training utilities"""
    print("Testing Phase 5: Training Utilities")
    print("=" * 50)
    
    # Test causal mask
    mask = create_causal_mask(4)
    assert mask is not None, "create_causal_mask not implemented"
    expected = torch.tensor([
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]
    ])
    assert torch.equal(mask, expected), "Causal mask incorrect"
    print("✓ Causal mask correct!")
    
    # Test tokenizer
    text = "hello world"
    tokenizer = CharacterTokenizer(text)
    assert tokenizer.vocab_size is not None, "Tokenizer not initialized"
    encoded = tokenizer.encode("hello")
    assert encoded is not None, "encode not implemented"
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello", f"decode failed: got '{decoded}'"
    print("✓ Tokenizer works!")
    
    print("\n✅ Phase 5 complete!\n")


def run_all_tests():
    """Run all phase tests"""
    print("\n" + "=" * 60)
    print("TINY LLM TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_phase1()
        # test_phase2()
        # test_phase3()
        # test_phase4()
        # test_phase5()
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! Your TinyLLM is ready to train!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        print("Keep working on the TODOs and try again!")


if __name__ == "__main__":
    run_all_tests()
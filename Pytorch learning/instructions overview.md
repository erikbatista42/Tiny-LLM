# TinyLLM: Learn PyTorch by Building a Transformer

A hands-on learning project to understand PyTorch and transformer architecture by building a character-level language model from scratch.

## Project Structure

```
tiny_llm/
├── model.py      # Core model components with TODO sections
├── train.py      # Training script with TODO sections
└── README.md     # This file
```

## Learning Path

### Phase 1: PyTorch Tensor Fundamentals
- Creating tensors
- Tensor operations (add, multiply, matmul)
- Reshaping and transposing

### Phase 2: Neural Network Basics
- Subclassing `nn.Module`
- Defining layers in `__init__`
- Implementing `forward` method

### Phase 3: Transformer Components
- Token embeddings
- Positional encoding (sinusoidal)
- Self-attention mechanism
- Multi-head attention
- Feed-forward networks
- Transformer blocks with residual connections

### Phase 4: Complete Model
- Assembling all components
- Forward pass through the full model
- Autoregressive text generation

### Phase 5: Training
- Data preparation and batching
- Training loop implementation
- Loss computation and backpropagation

## How to Use

1. **Read through model.py** - understand the architecture
2. **Complete TODOs in order** - start with Phase 1
3. **Run tests** to verify your implementations:
   ```bash
   python model.py
   ```
4. **Once all tests pass**, complete train.py
5. **Train your model** and generate text!

## TODO Checklist

### model.py

- [ ] TODO 1-3: Tensor basics (creating, operations, reshaping)
- [ ] TODO 4-5: nn.Module basics
- [ ] TODO 6-7: Token embedding
- [ ] TODO 8-9: Positional encoding
- [ ] TODO 10-11: Self-attention
- [ ] TODO 12-13: Multi-head attention
- [ ] TODO 14-15: Feed-forward network
- [ ] TODO 16-17: Transformer block
- [ ] TODO 18-19: Complete TinyLLM model
- [ ] TODO 20: Text generation
- [ ] TODO 21: Causal mask
- [ ] TODO 22-24: Character tokenizer

### train.py

- [ ] TODO 25-27: TextDataset
- [ ] TODO 28-29: Training and evaluation steps
- [ ] TODO 30: Full training loop
- [ ] TODO 31: Generation helper

## Architecture Diagram

```
Input: "Hello"
    ↓
[Character Indices: [7, 4, 11, 11, 14]]
    ↓
┌─────────────────────────────────────┐
│         Token Embedding             │
│     (vocab_size → embed_dim)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│       Positional Encoding           │
│    (adds position information)      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│                                     │
│        Transformer Block            │
│   ┌─────────────────────────────┐   │
│   │   Multi-Head Attention      │   │
│   │   (Q, K, V projections)     │   │
│   └─────────────────────────────┘   │
│              + residual             │
│   ┌─────────────────────────────┐   │
│   │   Feed-Forward Network      │   │
│   │   (expand → GELU → project) │   │
│   └─────────────────────────────┘   │
│              + residual             │
│                                     │
│         (× num_layers)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│          Layer Norm                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         Output Head                 │
│     (embed_dim → vocab_size)        │
└─────────────────────────────────────┘
    ↓
[Logits for next character prediction]
```

## Key Concepts to Understand

### Self-Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

The attention scores (softmax of QK^T) determine how much each position attends to every other position.

### Causal Masking

For language modeling, we use a causal mask to prevent positions from attending to future positions:

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

Position 0 can only see itself, position 1 can see 0 and 1, etc.

### Positional Encoding

Transformers process all positions in parallel, so we need to explicitly encode position information:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This creates unique, smooth patterns for each position.

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this project

## Expected Results

After training on the sample Shakespeare data for ~5 epochs, your model should generate somewhat coherent (if nonsensical) Shakespeare-like text. With more data and training, quality improves significantly!

Example output (after training):
```
To be the sears of the soul of the conscience
That makes the world and all the men and women
```
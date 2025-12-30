# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Supporting code for "The Welch Labs Illustrated Guide to AI" (https://www.welchlabs.com/ai-book). Educational Jupyter notebooks teaching AI/ML concepts from perceptrons through diffusion models.

## Development Commands

```bash
# Install dependencies
uv sync

# Install with optional UMAP support (requires: brew install llvm)
uv sync --extra umap

# Run Jupyter notebooks
uv run jupyter notebook
```

## Architecture

### Notebook Organization

Notebooks are numbered by chapter and build progressively:

1. **Perceptron** - Manual learning algorithms, T vs J classification, cats vs dogs
2. **Gradient Descent** - Loss landscapes, Llama-3.2-1B parameter exploration
3. **Backpropagation** - Forward/backward passes, GPS city classification
4. **Deep Learning** - Multi-layer networks, decision boundary visualization
5. **AlexNet** - CNN feature visualization at different layers
6. **Neural Scaling** - GPT-3/4 scaling laws, MNIST scaling example
7. **Mechanistic Interpretability** - Gemma-2-2b with SAE steering
8. **Attention** - Transformer attention mechanisms, DeepSeek analysis
9. **Diffusion** - DDPM/DDIM samplers, CLIP guidance, Stable Diffusion

Each chapter has corresponding exercise notebooks in `exercises/`.

### Key Patterns

**Training loops** follow standard PyTorch convention:
```python
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()
```

**Inference/visualization** uses no-grad context:
```python
model.eval()
with torch.no_grad():
    activations = model(input)
```

**Mechanistic interpretability** uses hook-based activation capture with transformer_lens and sae_lens for sparse autoencoder analysis.

### Dependencies

- **torch 2.2.2** - Pinned version using CPU-only PyTorch index
- **transformer_lens/sae_lens** - For mechanistic interpretability (chapter 7-8)
- **diffusers** - For Stable Diffusion pipelines (chapter 9)
- **smalldiffusion** - Custom library for spiral diffusion examples
- **numpy <2.0** - Pinned to avoid breaking changes

### Device Handling

Most notebooks default to CPU. For GPU support, look for patterns like:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else device  # Mac M1/M2
```

### Data

Notebooks reference `data/` directory containing images for visualization examples. Model weights are downloaded from HuggingFace as needed.

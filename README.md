<div align="center">

# 1D-Condition-Method-PyTorch

_✨ Enhance your neural networks with advanced conditioning methods. ✨_

</div>

## Introduction 🚀

This repository presents PyTorch implementations of various methods to inject additional information, such as time embeddings in diffusion UNet or speaker embeddings in speech synthesis, into your models. Enhance your network's performance and capabilities with these advanced conditioning techniques.

## Features 🌟

- **FiLM Layer**: Incorporate the [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) into your models to dynamically influence their behavior based on external information.
- **Conditional Layer Norm**: Utilize the Conditional Layer Norm strategy from [AdaSpeech](https://arxiv.org/abs/2103.00993) for adaptive and context-aware normalization.
- **Style-Adaptive Layer Normalization**: Utilize the Style-Adaptive Layer Normalization from [Meta-StyleSpeech](https://arxiv.org/abs/2106.03153) for conditioning the normalization process with external data.


## Usage 📘

### FiLM Layer

```python
import torch
from layers import FiLMLayer

x = torch.randn((16,37,256)) # [batch_size, time, in_channels]
c = torch.randn((16,1,320)) # [batch_size, 1, cond_channels]

model = FiLMLayer(256, 320)
output = model(x, c) # [batch_size, time, in_channels]
```

### Conditional Layer Norm

```python
import torch
from layers import ConditionalLayerNorm

x = torch.randn((16,37,256)) # [batch_size, time, in_channels]
c = torch.randn((16,1,320)) # [batch_size, 1, cond_channels]

model = ConditionalLayerNorm(256, 320)
output = model(x, c) # [batch_size, time, in_channels]
```

### Style-Adaptive Layer Normalization

```python
import torch
from layers import StyleAdaptiveLayerNorm

x = torch.randn((16,37,256)) # [batch_size, time, in_channels]
c = torch.randn((16,1,320)) # [batch_size, 1, cond_channels]

model = StyleAdaptiveLayerNorm(256, 320)
output = model(x, c) # [batch_size, time, in_channels]
```
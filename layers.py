import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Feature-wise Linear Modulation (FiLM) layer
        
        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x, c):
        """
        Parameters:
        x (Tensor): The input feature maps with shape [batch_size, time, in_channels].
        c (Tensor): The conditioning input with shape [batch_size, 1, cond_channels].
        
        Returns:
        Tensor: The modulated feature maps with the same shape as input x.
        """
        film_params = self.film(c)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)
        
        return gamma * x + beta
    
class ConditionalLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels, eps=1e-5):
        """
        Conditional Layer Normalization module.
        
        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        eps: A small number to prevent division by zero in normalization.
        """
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        
        self.weight_transform = nn.Linear(cond_channels, in_channels)
        self.bias_transform = nn.Linear(cond_channels, in_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight_transform.weight, 0.0)
        nn.init.constant_(self.weight_transform.bias, 1.0)
        nn.init.constant_(self.bias_transform.weight, 0.0)
        nn.init.constant_(self.bias_transform.bias, 0.0)

    def forward(self, x, c):
        """
        Parameters:
        x (Tensor): The input feature maps with shape [batch_size, time, in_channels].
        c (Tensor): The conditioning input with shape [batch_size, 1, cond_channels].
        
        Returns:
        Tensor: The modulated feature maps with the same shape as input x.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        
        x_normalized = (x - mean) / (std + self.eps)
        
        gamma = self.weight_transform(c)
        beta = self.bias_transform(c)
        
        out = gamma * x_normalized + beta
        
        return out
    
class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.saln = nn.Linear(cond_channels, in_channels * 2)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        """
        Parameters:
        x (Tensor): The input feature maps with shape [batch_size, time, in_channels].
        c (Tensor): The conditioning input with shape [batch_size, 1, cond_channels].
        
        Returns:
        Tensor: The modulated feature maps with the same shape as input x.
        """
        saln_params = self.saln(c)
        gamma, beta = torch.chunk(saln_params, chunks=2, dim=-1)
        
        out = self.norm(x)
        out = gamma * out + beta
        
        return out
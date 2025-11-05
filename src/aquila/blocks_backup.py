"""
Architectural building blocks for genomic neural networks.
These blocks can be composed via YAML configuration to build custom architectures.
"""

import torch
import torch.nn as nn
from typing import Optional
from . import layers


############################################################
# Embedding Blocks
############################################################

def snp_embedding(embed_dim=128, dropout=0.1, **kwargs):
    """SNP token embedding block.
    
    Args:
        embed_dim: Embedding dimension
        dropout: Dropout rate
    
    Returns:
        SNPEmbedding module
    """
    return layers.SNPEmbedding(embed_dim=embed_dim, dropout=dropout)


def positional_encoding(d_model, max_len=50000, dropout=0.1, **kwargs):
    """Positional encoding block.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    
    Returns:
        PositionalEncoding module
    """
    return layers.PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)


############################################################
# Transformer Blocks
############################################################

class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = layers.MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        
        return x


def transformer(d_model, num_heads, d_ff, dropout=0.1, activation='gelu', **kwargs):
    """Single transformer encoder block.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        TransformerBlock module
    """
    return TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                           dropout=dropout, activation=activation)


class TransformerTower(nn.Module):
    """Stack of transformer encoder blocks."""
    def __init__(self, embed_dim, num_heads, d_ff, num_layers, dropout=0.1, activation='gelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower(embed_dim, num_heads, d_ff, num_layers, dropout=0.1, activation='gelu', **kwargs):
    """Stack of transformer blocks.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer layers
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        TransformerTower module
    """
    return TransformerTower(embed_dim=embed_dim, num_heads=num_heads, d_ff=d_ff,
                           num_layers=num_layers, dropout=dropout, activation=activation)


############################################################
# Convolution Blocks
############################################################

class ConvBlock(nn.Module):
    """1D convolution block with normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding='same', activation='relu', dropout=0.1, norm_type='layer'):
        super().__init__()
        
        if padding == 'same':
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Normalization
        if norm_type == 'layer':
            self.norm = nn.GroupNorm(1, out_channels)  # Layer norm for conv
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, channels) -> need (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
        x = layers.activate(x, self.activation)
        x = self.dropout(x)
        
        # Update mask if provided (downsample mask to match new sequence length)
        if mask is not None:
            # Downsample mask using max pooling (keep True if any position was True)
            # mask: (batch, original_seq_len) -> (batch, new_seq_len)
            new_seq_len = x.size(1)
            if mask.size(1) != new_seq_len:
                # Simple downsampling: take every stride-th element
                stride = self.conv.stride[0]
                kernel_size = self.conv.kernel_size[0]
                padding = self.conv.padding[0]
                
                # Calculate indices for downsampled positions
                # Use max pooling logic: a position is valid if any input position was valid
                mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                mask_pooled = torch.nn.functional.max_pool1d(
                    mask_float, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
                mask = mask_pooled.squeeze(1).bool()  # (batch, new_seq_len)
                
                # Ensure mask size matches output size
                if mask.size(1) > new_seq_len:
                    mask = mask[:, :new_seq_len]
                elif mask.size(1) < new_seq_len:
                    # Pad with True
                    padding_size = new_seq_len - mask.size(1)
                    mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        return x, mask


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same',
               activation='relu', dropout=0.1, norm_type='layer', **kwargs):
    """1D convolution block.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Stride
        padding: Padding mode
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type ('layer', 'batch', or None)
    
    Returns:
        ConvBlock module
    """
    return ConvBlock(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    activation=activation, dropout=dropout, norm_type=norm_type)


class ConvTower(nn.Module):
    """Stack of convolution blocks."""
    def __init__(self, in_channels, filters_list, kernel_size=3, activation='relu',
                 dropout=0.1, norm_type='layer'):
        super().__init__()
        layers_list = []
        current_channels = in_channels
        
        for out_channels in filters_list:
            layers_list.append(ConvBlock(
                current_channels, out_channels, kernel_size,
                activation=activation, dropout=dropout, norm_type=norm_type
            ))
            current_channels = out_channels
        
        self.layers = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.layers(x)


def conv_tower(in_channels, filters_list, kernel_size=3, activation='relu',
               dropout=0.1, norm_type='layer', **kwargs):
    """Stack of convolution blocks.
    
    Args:
        in_channels: Input channels
        filters_list: List of output channels for each layer
        kernel_size: Convolution kernel size
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type
    
    Returns:
        ConvTower module
    """
    return ConvTower(in_channels=in_channels, filters_list=filters_list,
                    kernel_size=kernel_size, activation=activation,
                    dropout=dropout, norm_type=norm_type)


############################################################
# Residual Blocks
############################################################

class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_ff)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = layers.activate(x, self.activation)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


def res_block(d_model, d_ff, dropout=0.1, activation='relu', **kwargs):
    """Residual block.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        ResidualBlock module
    """
    return ResidualBlock(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)


class ResidualTower(nn.Module):
    """Stack of residual blocks."""
    def __init__(self, d_model, d_ff, num_layers, dropout=0.1, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def res_tower(d_model, d_ff, num_layers, dropout=0.1, activation='relu', **kwargs):
    """Stack of residual blocks.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_layers: Number of residual blocks
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        ResidualTower module
    """
    return ResidualTower(d_model=d_model, d_ff=d_ff, num_layers=num_layers,
                        dropout=dropout, activation=activation)


############################################################
# MLP Blocks
############################################################

class MLPBlock(nn.Module):
    """MLP block with multiple layers."""
    def __init__(self, d_model, d_ff, num_layers=3, dropout=0.1, activation='gelu'):
        super().__init__()
        layer_list = []
        current_dim = d_model
        
        for i in range(num_layers):
            layer_list.append(nn.Linear(current_dim, d_ff))
            layer_list.append(nn.LayerNorm(d_ff))
            if activation:
                layer_list.append(nn.Lambda(lambda x: layers.activate(x, activation)))
            if dropout > 0:
                layer_list.append(nn.Dropout(dropout))
            current_dim = d_ff
        
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.network(x)


def mlp_block(d_model, d_ff, num_layers=3, dropout=0.1, activation='gelu', **kwargs):
    """MLP block.
    
    Args:
        d_model: Input dimension
        d_ff: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        MLPBlock module
    """
    # Simple implementation without nn.Lambda
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layer_list = []
            current_dim = d_model
            
            for i in range(num_layers):
                layer_list.append(nn.Linear(current_dim, d_ff))
                layer_list.append(nn.LayerNorm(d_ff))
                layer_list.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
                if dropout > 0:
                    layer_list.append(nn.Dropout(dropout))
                current_dim = d_ff
            
            self.network = nn.Sequential(*layer_list)
        
        def forward(self, x):
            return self.network(x)
    
    return SimpleMLP()


############################################################
# Dense Blocks
############################################################

class DenseBlock(nn.Module):
    """Dense transformation block."""
    def __init__(self, in_features, out_features, activation='relu', dropout=0.1, 
                 norm_type='layer', bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(out_features)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_features)
        else:
            self.norm = nn.Identity()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.linear(x)
        
        # Handle batch norm for 3D tensors
        if isinstance(self.norm, nn.BatchNorm1d) and x.dim() == 3:
            # (batch, seq, features) -> (batch, features, seq)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)
        
        if self.activation:
            x = layers.activate(x, self.activation)
        x = self.dropout(x)
        return x


def dense_block(in_features, out_features, activation='relu', dropout=0.1,
                norm_type='layer', bias=True, **kwargs):
    """Dense transformation block.
    
    Args:
        in_features: Input features
        out_features: Output features
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type
        bias: Use bias in linear layer
    
    Returns:
        DenseBlock module
    """
    return DenseBlock(in_features=in_features, out_features=out_features,
                     activation=activation, dropout=dropout, norm_type=norm_type, bias=bias)


############################################################
# Pooling Blocks
############################################################

def global_pool(pool_type='mean', d_model=None, **kwargs):
    """Global pooling block.
    
    Args:
        pool_type: Pooling type ('mean', 'max', 'attention')
        d_model: Model dimension (required for attention pooling)
    
    Returns:
        GlobalPooling module
    """
    if pool_type == 'attention' and d_model is None:
        raise ValueError("d_model required for attention pooling")
    return layers.GlobalPooling(d_model=d_model or 128, pool_type=pool_type)


############################################################
# Head Blocks
############################################################

class RegressionHead(nn.Module):
    """Regression head with optional hidden layers."""
    def __init__(self, in_features, num_targets, hidden_features=None, 
                 dropout=0.1, activation='gelu'):
        super().__init__()
        layer_list = []
        
        if hidden_features is not None:
            layer_list.extend([
                nn.Linear(in_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, num_targets)
            ])
        else:
            layer_list.append(nn.Linear(in_features, num_targets))
        
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.network(x)


def regression_head(in_features, num_targets, hidden_features=None, 
                    dropout=0.1, activation='gelu', **kwargs):
    """Regression head block.
    
    Args:
        in_features: Input features
        num_targets: Number of regression targets
        hidden_features: Hidden layer dimension (None for no hidden layer)
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        RegressionHead module
    """
    return RegressionHead(in_features=in_features, num_targets=num_targets,
                         hidden_features=hidden_features, dropout=dropout, activation=activation)


class ClassificationHead(nn.Module):
    """Classification head with optional hidden layers."""
    def __init__(self, in_features, num_tasks, num_classes=2, hidden_features=None,
                 dropout=0.1, activation='gelu'):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        
        layer_list = []
        if hidden_features is not None:
            layer_list.extend([
                nn.Linear(in_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            final_dim = hidden_features
        else:
            final_dim = in_features
        
        # Output layer
        if num_classes == 2:
            layer_list.append(nn.Linear(final_dim, num_tasks))
        else:
            layer_list.append(nn.Linear(final_dim, num_tasks * num_classes))
        
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        logits = self.network(x)
        if self.num_classes > 2:
            logits = logits.reshape(-1, self.num_tasks, self.num_classes)
        return logits


def classification_head(in_features, num_tasks, num_classes=2, hidden_features=None,
                       dropout=0.1, activation='gelu', **kwargs):
    """Classification head block.
    
    Args:
        in_features: Input features
        num_tasks: Number of classification tasks
        num_classes: Number of classes per task
        hidden_features: Hidden layer dimension (None for no hidden layer)
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        ClassificationHead module
    """
    return ClassificationHead(in_features=in_features, num_tasks=num_tasks,
                             num_classes=num_classes, hidden_features=hidden_features,
                             dropout=dropout, activation=activation)


############################################################
# Block Dictionary
############################################################

name_func = {
    # Embeddings
    'snp_embedding': snp_embedding,
    'positional_encoding': positional_encoding,
    
    # Transformers
    'transformer': transformer,
    'transformer_tower': transformer_tower,
    
    # Convolutions
    'conv_block': conv_block,
    'conv_tower': conv_tower,
    
    # Residual
    'res_block': res_block,
    'res_tower': res_tower,
    
    # MLP
    'mlp_block': mlp_block,
    
    # Dense
    'dense_block': dense_block,
    
    # Pooling
    'global_pool': global_pool,
    
    # Heads
    'regression_head': regression_head,
    'classification_head': classification_head,
}


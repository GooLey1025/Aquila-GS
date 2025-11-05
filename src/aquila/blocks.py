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
# RoPE Transformer Blocks
############################################################

class TransformerBlockRoPE(nn.Module):
    """Single transformer encoder block with Rotary Position Embedding (RoPE)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = layers.MultiHeadSelfAttentionRoPE(d_model, num_heads, dropout)
        self.ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual (Pre-LN)
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        
        return x


def transformer_rope(d_model, num_heads, d_ff, dropout=0.1, activation='gelu', **kwargs):
    """Single transformer encoder block with RoPE.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        TransformerBlockRoPE module
    """
    return TransformerBlockRoPE(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                                dropout=dropout, activation=activation)


class TransformerTowerRoPE(nn.Module):
    """Stack of transformer encoder blocks with RoPE (no separate positional encoding needed)."""
    def __init__(self, embed_dim, num_heads, d_ff, num_layers, dropout=0.1, activation='gelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockRoPE(embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower_rope(embed_dim, num_heads, d_ff, num_layers, dropout=0.1, activation='gelu', **kwargs):
    """Stack of transformer blocks with RoPE.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer layers
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        TransformerTowerRoPE module
    
    Note:
        RoPE encodes position information directly in the attention mechanism,
        so no separate positional encoding layer is needed.
    """
    return TransformerTowerRoPE(embed_dim=embed_dim, num_heads=num_heads, d_ff=d_ff,
                                num_layers=num_layers, dropout=dropout, activation=activation)


############################################################
# Learnable PE Transformer Blocks
############################################################

class TransformerBlockLearnable(nn.Module):
    """Single transformer encoder block (for use with learnable positional embeddings)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = layers.MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual (Pre-LN)
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        
        return x


def transformer_learnable(d_model, num_heads, d_ff, dropout=0.1, activation='gelu', **kwargs):
    """Single transformer encoder block with learnable PE.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        TransformerBlockLearnable module
    """
    return TransformerBlockLearnable(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                                    dropout=dropout, activation=activation)


class TransformerTowerLearnable(nn.Module):
    """Stack of transformer blocks with learnable positional embeddings."""
    def __init__(self, embed_dim, num_heads, d_ff, num_layers, max_positions=50000, 
                 dropout=0.1, activation='gelu'):
        super().__init__()
        
        # Learnable positional embedding (applied before transformer blocks)
        self.pos_embedding = layers.LearnablePositionalEmbedding(
            d_model=embed_dim,
            max_positions=max_positions,
            dropout=dropout
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlockLearnable(embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        # Apply learnable positional embeddings
        x = self.pos_embedding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower_learnable(embed_dim, num_heads, d_ff, num_layers, max_positions=50000,
                                dropout=0.1, activation='gelu', **kwargs):
    """Stack of transformer blocks with learnable positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer layers
        max_positions: Maximum number of positions for learnable embeddings
        dropout: Dropout rate
        activation: Activation function
    
    Returns:
        TransformerTowerLearnable module
    
    Note:
        Uses learnable position embeddings suitable for sparse variant positions.
        Position embeddings are added before the first transformer layer.
    """
    return TransformerTowerLearnable(embed_dim=embed_dim, num_heads=num_heads, d_ff=d_ff,
                                    num_layers=num_layers, max_positions=max_positions,
                                    dropout=dropout, activation=activation)


############################################################
# Multi-Query Attention Transformer Blocks
############################################################

class TransformerBlockMQA(nn.Module):
    """Transformer block with Multi-Query Attention and RoPE."""
    def __init__(self, d_model, num_query_heads=8, qk_head_dim=128, v_head_dim=192,
                 dropout=0.1, max_position=8192, norm_type='layer'):
        super().__init__()
        
        # Choose normalization type
        if norm_type == 'rms':
            self.norm1 = layers.RMSNorm(d_model)
            self.norm2 = layers.RMSNorm(d_model)
            self.norm3 = layers.RMSNorm(d_model)
            self.norm4 = layers.RMSNorm(d_model)
        else:  # layer
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.norm4 = nn.LayerNorm(d_model)
        
        # Multi-Query Attention with RoPE
        self.attention = layers.MultiQueryAttentionRoPE(
            d_model=d_model,
            num_query_heads=num_query_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dropout=dropout,
            max_position=max_position
        )
        
        # Feed-forward network (2x expansion with ReLU, AlphaGenome style)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Attention block with residual
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask)
        attn_out = self.norm2(attn_out)
        x = x + self.dropout1(attn_out)
        
        # FFN block with residual
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.norm4(ffn_out)
        x = x + self.dropout2(ffn_out)
        
        return x


def transformer_mqa(d_model, num_query_heads=8, qk_head_dim=128, v_head_dim=192,
                   dropout=0.1, max_position=8192, norm_type='layer', **kwargs):
    """Single transformer block with Multi-Query Attention.
    
    Args:
        d_model: Model dimension
        num_query_heads: Number of query heads (default 8)
        qk_head_dim: Dimension of each query/key head (default 128)
        v_head_dim: Dimension of value head (default 192)
        dropout: Dropout rate
        max_position: Maximum position for RoPE
        norm_type: 'layer' for LayerNorm (default) or 'rms' for RMSNorm
    
    Returns:
        TransformerBlockMQA module
    """
    return TransformerBlockMQA(
        d_model=d_model,
        num_query_heads=num_query_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        dropout=dropout,
        max_position=max_position,
        norm_type=norm_type
    )


class TransformerTowerMQA(nn.Module):
    """Stack of transformer blocks with Multi-Query Attention and RoPE."""
    def __init__(self, embed_dim, num_layers=9, num_query_heads=8, qk_head_dim=128,
                 v_head_dim=192, dropout=0.1, max_position=8192, norm_type='layer'):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlockMQA(
                d_model=embed_dim,
                num_query_heads=num_query_heads,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                dropout=dropout,
                max_position=max_position,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower_mqa(embed_dim, num_layers=9, num_query_heads=8, qk_head_dim=128,
                         v_head_dim=192, dropout=0.1, max_position=8192, 
                         norm_type='layer', **kwargs):
    """Stack of transformer blocks with Multi-Query Attention.
    
    Args:
        embed_dim: Embedding dimension (d_model)
        num_layers: Number of transformer layers (default 9)
        num_query_heads: Number of query heads in MQA (default 8)
        qk_head_dim: Dimension of each query/key head (default 128)
        v_head_dim: Dimension of value head (default 192)
        dropout: Dropout rate
        max_position: Maximum sequence length for RoPE
        norm_type: 'layer' for LayerNorm (default) or 'rms' for RMSNorm
    
    Returns:
        TransformerTowerMQA module
    
    Note:
        Features:
        - Multi-Query Attention (multiple query heads, 1 shared K/V)
        - RoPE for position encoding (no separate PE needed)
        - Soft-clipping of attention logits for stability
        - Optional RMSNorm instead of LayerNorm
        - FFN with 2x expansion and ReLU activation
    """
    return TransformerTowerMQA(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        dropout=dropout,
        max_position=max_position,
        norm_type=norm_type
    )


############################################################
# Convolution Blocks
############################################################

class ConvBlockCNA(nn.Module):
    """1D convolution block with CNA order (Conv -> Norm -> Activation), with optional pooling.
    
    Residual connection: input + conv_output (before norm and activation)
    """
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, stride=1, 
                 padding='same', activation='relu', dropout=0.1, norm_type='layer',
                 pooling_type=None, pool_size=2, pool_stride=None, residual=False):
        super().__init__()
        
        if padding == 'same':
            padding = kernel_size // 2
        
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Use lazy initialization if in_channels not specified
        if in_channels is None:
            self.conv = nn.LazyConv1d(out_channels, kernel_size, stride, padding)
            # For lazy initialization, we'll create residual_proj in forward if needed
            self.residual_proj = None
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Projection layer for residual if channels don't match or stride > 1
            if residual and (in_channels != out_channels or stride > 1):
                self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride, 0)
            else:
                self.residual_proj = None
        
        # Normalization
        if norm_type == 'layer':
            self.norm = nn.GroupNorm(1, out_channels)  # Layer norm for conv
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Pooling (optional)
        self.pooling_type = pooling_type
        if pooling_type is not None:
            pool_stride = pool_stride if pool_stride is not None else pool_size
            pooling_type_lower = pooling_type.lower()
            if pooling_type_lower in ['maxpool1d', 'max']:
                self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower in ['avgpool1d', 'avg', 'average']:
                self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)
            else:
                raise ValueError(f"Unsupported pooling_type: {pooling_type}. "
                               f"Supported types: MaxPooling: 'maxpool1d', 'max', AvgPooling: 'avgpool1d', 'avg', 'average'")
            self.pool_size = pool_size
            self.pool_stride = pool_stride
        else:
            self.pool = None
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, channels) -> need (batch, channels, seq_len)
        identity = x.transpose(1, 2) if self.residual else None
        
        x = x.transpose(1, 2)
        
        # Lazy initialization of residual_proj if needed
        if self.residual and self.in_channels is None and self.residual_proj is None:
            in_channels = x.size(1)
            out_channels = self.out_channels
            stride = self.stride
            if in_channels != out_channels or stride > 1:
                self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride, 0).to(x.device)
        
        x = self.conv(x)
        
        # CNA residual: input + conv_output (add before norm and activation)
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity
        
        x = self.norm(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
        x = layers.activate(x, self.activation)
        x = self.dropout(x)
        
        # Update mask after convolution if provided
        if mask is not None:
            new_seq_len = x.size(1)
            if mask.size(1) != new_seq_len:
                stride = self.conv.stride[0]
                kernel_size = self.conv.kernel_size[0]
                padding = self.conv.padding[0]
                
                mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                mask_pooled = torch.nn.functional.max_pool1d(
                    mask_float, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0
                )
                mask = mask_pooled.squeeze(1).bool()  # (batch, new_seq_len)
                
                if mask.size(1) > new_seq_len:
                    mask = mask[:, :new_seq_len]
                elif mask.size(1) < new_seq_len:
                    padding_size = new_seq_len - mask.size(1)
                    mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        # Apply pooling if specified
        if self.pool is not None:
            x = x.transpose(1, 2)  # (batch, channels, seq_len) for pooling
            x = self.pool(x)
            x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
            
            # Update mask after pooling
            if mask is not None:
                new_seq_len = x.size(1)
                if mask.size(1) != new_seq_len:
                    mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                    mask_pooled = torch.nn.functional.max_pool1d(
                        mask_float,
                        kernel_size=self.pool_size,
                        stride=self.pool_stride
                    )
                    mask = mask_pooled.squeeze(1).bool()
                    
                    if mask.size(1) > new_seq_len:
                        mask = mask[:, :new_seq_len]
                    elif mask.size(1) < new_seq_len:
                        padding_size = new_seq_len - mask.size(1)
                        mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        return x, mask


class ConvBlockNAC(nn.Module):
    """1D convolution block with NAC order (Norm -> Activation -> Conv), with optional pooling.
    
    Residual connection: identity + conv_output (after conv, so shapes match)
    """
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, stride=1, 
                 padding='same', activation='relu', dropout=0.1, norm_type='layer',
                 pooling_type=None, pool_size=2, pool_stride=None, residual=False):
        super().__init__()
        
        if padding == 'same':
            padding = kernel_size // 2
        
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Normalization (applied to input channels)
        if in_channels is not None:
            if norm_type == 'layer':
                self.norm = nn.GroupNorm(1, in_channels)  # Layer norm for conv
            elif norm_type == 'batch':
                self.norm = nn.BatchNorm1d(in_channels)
            else:
                self.norm = nn.Identity()
        else:
            # Use lazy normalization if in_channels not specified
            self.norm = None
            self.norm_type = norm_type
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Convolution
        if in_channels is None:
            self.conv = nn.LazyConv1d(out_channels, kernel_size, stride, padding)
            # For lazy initialization, we'll create residual_proj in forward if needed
            self.residual_proj = None
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Projection layer for residual if channels don't match or stride > 1
            if residual and (in_channels != out_channels or stride > 1):
                self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride, 0)
            else:
                self.residual_proj = None
        
        # Pooling (optional)
        self.pooling_type = pooling_type
        if pooling_type is not None:
            pool_stride = pool_stride if pool_stride is not None else pool_size
            pooling_type_lower = pooling_type.lower()
            if pooling_type_lower in ['maxpool1d', 'max']:
                self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower in ['avgpool1d', 'avg', 'average']:
                self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)
            else:
                raise ValueError(f"Unsupported pooling_type: {pooling_type}. "
                               f"Supported types: MaxPooling: 'maxpool1d', 'max', AvgPooling: 'avgpool1d', 'avg', 'average'")
            self.pool_size = pool_size
            self.pool_stride = pool_stride
        else:
            self.pool = None
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, channels) -> need (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Save identity before norm (for residual)
        identity = x if self.residual else None
        
        # Lazy initialization of norm if needed
        if self.norm is None:
            in_channels_actual = x.size(1)
            if self.norm_type == 'layer':
                self.norm = nn.GroupNorm(1, in_channels_actual).to(x.device)
            elif self.norm_type == 'batch':
                self.norm = nn.BatchNorm1d(in_channels_actual).to(x.device)
            else:
                self.norm = nn.Identity()
        
        # Lazy initialization of residual_proj if needed
        if self.residual and self.in_channels is None and self.residual_proj is None:
            in_channels = x.size(1)
            out_channels = self.out_channels
            stride = self.stride
            if in_channels != out_channels or stride > 1:
                self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride, 0).to(x.device)
        
        # NAC order: Norm -> Activation -> Conv
        x = self.norm(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels) for activation
        x = layers.activate(x, self.activation)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # Back to (batch, channels, seq_len) for conv
        x = self.conv(x)
        
        # NAC residual: identity + conv_output (add after conv so shapes match)
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            # Both identity and x are now (batch, out_channels, seq_len_out)
            x = x + identity
        
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
        
        # Update mask after convolution if provided
        if mask is not None:
            new_seq_len = x.size(1)
            if mask.size(1) != new_seq_len:
                stride = self.conv.stride[0]
                kernel_size = self.conv.kernel_size[0]
                padding = self.conv.padding[0]
                
                mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                mask_pooled = torch.nn.functional.max_pool1d(
                    mask_float, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0
                )
                mask = mask_pooled.squeeze(1).bool()  # (batch, new_seq_len)
                
                if mask.size(1) > new_seq_len:
                    mask = mask[:, :new_seq_len]
                elif mask.size(1) < new_seq_len:
                    padding_size = new_seq_len - mask.size(1)
                    mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        # Apply pooling if specified
        if self.pool is not None:
            x = x.transpose(1, 2)  # (batch, channels, seq_len) for pooling
            x = self.pool(x)
            x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
            
            # Update mask after pooling
            if mask is not None:
                new_seq_len = x.size(1)
                if mask.size(1) != new_seq_len:
                    mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                    mask_pooled = torch.nn.functional.max_pool1d(
                        mask_float,
                        kernel_size=self.pool_size,
                        stride=self.pool_stride
                    )
                    mask = mask_pooled.squeeze(1).bool()
                    
                    if mask.size(1) > new_seq_len:
                        mask = mask[:, :new_seq_len]
                    elif mask.size(1) < new_seq_len:
                        padding_size = new_seq_len - mask.size(1)
                        mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        return x, mask


def conv_block(in_channels=None, out_channels=None, kernel_size=3, stride=1, padding='same',
               activation='relu', dropout=0.1, norm_type='layer', order='nac', 
               pooling_type=None, pool_size=2, pool_stride=None, residual=False, **kwargs):
    """1D convolution block with configurable operation order and optional pooling.
    
    Args:
        in_channels: Input channels (None for automatic detection)
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding mode
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type ('layer', 'batch', or None)
        order: Operation order - 'nac' (Norm->Activation->Conv, default) or 'cna' (Conv->Norm->Activation)
        pooling_type: Pooling type - None (no pooling, default), 'maxpool1d'/'max', or 'avgpool1d'/'avg'
        pool_size: Pooling kernel size (default: 2)
        pool_stride: Pooling stride (default: same as pool_size)
        residual: Add residual connection (default: False)
                  - CNA: input + conv_output (after conv, before norm)
                  - NAC: identity + conv_output (after conv)
    
    Returns:
        ConvBlockNAC or ConvBlockCNA module depending on order
    """
    order = order.lower()
    if order == 'nac':
        return ConvBlockNAC(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           activation=activation, dropout=dropout, norm_type=norm_type,
                           pooling_type=pooling_type, pool_size=pool_size, pool_stride=pool_stride,
                           residual=residual)
    elif order == 'cna':
        return ConvBlockCNA(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           activation=activation, dropout=dropout, norm_type=norm_type,
                           pooling_type=pooling_type, pool_size=pool_size, pool_stride=pool_stride,
                           residual=residual)
    else:
        raise ValueError(f"order must be 'nac' or 'cna', got '{order}'")


class ConvTower(nn.Module):
    """Stack of convolution blocks."""
    def __init__(self, in_channels, filters_list, kernel_size=3, activation='relu',
                 dropout=0.1, norm_type='layer', order='nac', 
                 pooling_type=None, pool_size=2, pool_stride=None):
        super().__init__()
        layers_list = []
        current_channels = in_channels
        
        for out_channels in filters_list:
            layers_list.append(conv_block(
                current_channels, out_channels, kernel_size,
                activation=activation, dropout=dropout, norm_type=norm_type, order=order,
                pooling_type=pooling_type, pool_size=pool_size, pool_stride=pool_stride
            ))
            current_channels = out_channels
        
        self.layers = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.layers(x)


def conv_tower(in_channels, filters_list, kernel_size=3, activation='relu',
               dropout=0.1, norm_type='layer', order='nac', 
               pooling_type=None, pool_size=2, pool_stride=None, **kwargs):
    """Stack of convolution blocks with optional pooling.
    
    Args:
        in_channels: Input channels
        filters_list: List of output channels for each layer
        kernel_size: Convolution kernel size
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type
        order: Operation order - 'nac' (default) or 'cna'
        pooling_type: Pooling type - None (no pooling, default), 'maxpool1d'/'max', or 'avgpool1d'/'avg'
        pool_size: Pooling kernel size (default: 2)
        pool_stride: Pooling stride (default: same as pool_size)
    
    Returns:
        ConvTower module
    """
    return ConvTower(in_channels=in_channels, filters_list=filters_list,
                    kernel_size=kernel_size, activation=activation,
                    dropout=dropout, norm_type=norm_type, order=order,
                    pooling_type=pooling_type, pool_size=pool_size, pool_stride=pool_stride)


############################################################
# Dense Blocks (DenseNet)
############################################################

class DenseBlock(nn.Module):
    """Dense block with concatenation of all previous layer outputs.
    
    Each layer in the block receives all previous feature maps as input.
    Features grow by growth_rate at each layer.
    """
    def __init__(self, in_channels, num_layers=4, growth_rate=32, 
                 kernel_size=3, activation='relu', dropout=0.1, 
                 norm_type='layer', order='nac', bottleneck=True, 
                 bottleneck_channels=128):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.bottleneck = bottleneck
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            if bottleneck:
                # 1x1 conv bottleneck (reduces channels before 3x3 conv)
                # Use None for in_channels to enable lazy initialization
                layer = nn.ModuleList([
                    conv_block(
                        in_channels=None,  # Lazy init - will auto-detect from concatenated features
                        out_channels=bottleneck_channels,
                        kernel_size=1,
                        activation=activation,
                        dropout=dropout,
                        norm_type=norm_type,
                        order=order,
                        residual=False
                    ),
                    conv_block(
                        in_channels=bottleneck_channels,
                        out_channels=growth_rate,
                        kernel_size=kernel_size,
                        activation=activation,
                        dropout=dropout,
                        norm_type=norm_type,
                        order=order,
                        residual=False
                    )
                ])
            else:
                # Direct convolution - use lazy init
                layer = conv_block(
                    in_channels=None,  # Lazy init - will auto-detect from concatenated features
                    out_channels=growth_rate,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout,
                    norm_type=norm_type,
                    order=order,
                    residual=False
                )
            
            self.layers.append(layer)
            current_channels += growth_rate
        
        self.out_channels = current_channels
    
    def forward(self, x, mask=None):
        features = [x]
        
        for layer in self.layers:
            # Concatenate all previous features
            concat_features = torch.cat(features, dim=-1)
            
            if isinstance(layer, nn.ModuleList):
                # Bottleneck version
                out, mask = layer[0](concat_features, mask)
                out, mask = layer[1](out, mask)
            else:
                # Direct version
                out, mask = layer(concat_features, mask)
            
            features.append(out)
        
        # Final concatenation of all layers
        return torch.cat(features, dim=-1), mask


class TransitionBlock(nn.Module):
    """Transition block for compressing features between dense blocks.
    
    Uses 1x1 convolution for channel compression and optional pooling
    for spatial downsampling.
    """
    def __init__(self, in_channels, compression=0.5, activation='relu',
                 dropout=0.1, norm_type='layer', order='nac',
                 pooling_type=None, pool_size=2, pool_stride=None):
        super().__init__()
        
        out_channels = int(in_channels * compression)
        
        # 1x1 compression convolution
        self.conv = conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order,
            residual=False
        )
        
        # Optional pooling
        self.pooling_type = pooling_type
        if pooling_type is not None:
            pool_stride = pool_stride if pool_stride is not None else pool_size
            pooling_type_lower = pooling_type.lower()
            if pooling_type_lower in ['avg', 'average', 'avgpool1d']:
                self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower in ['max', 'maxpool1d']:
                self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower == 'softmax':
                self.pool = layers.SoftmaxPooling1d(kernel_size=pool_size, stride=pool_stride)
            else:
                raise ValueError(f"Unsupported pooling_type: {pooling_type}. "
                               f"Supported types: 'avg', 'max', 'softmax'")
            self.pool_size = pool_size
            self.pool_stride = pool_stride
        else:
            self.pool = None
        
        self.out_channels = out_channels
    
    def forward(self, x, mask=None):
        x, mask = self.conv(x, mask)
        
        if self.pool is not None:
            # Apply pooling
            x = x.transpose(1, 2)  # (batch, channels, seq_len)
            x = self.pool(x)
            x = x.transpose(1, 2)  # (batch, seq_len, channels)
            
            # Update mask
            if mask is not None:
                new_seq_len = x.size(1)
                if mask.size(1) != new_seq_len:
                    mask_float = mask.float().unsqueeze(1)
                    mask_pooled = torch.nn.functional.max_pool1d(
                        mask_float,
                        kernel_size=self.pool_size,
                        stride=self.pool_stride
                    )
                    mask = mask_pooled.squeeze(1).bool()
                    
                    if mask.size(1) > new_seq_len:
                        mask = mask[:, :new_seq_len]
                    elif mask.size(1) < new_seq_len:
                        padding_size = new_seq_len - mask.size(1)
                        mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        return x, mask


def dense_block(in_channels, num_layers=4, growth_rate=32, kernel_size=3,
                activation='relu', dropout=0.1, norm_type='layer', order='nac',
                bottleneck=True, bottleneck_channels=128, **kwargs):
    """Dense block with feature concatenation.
    
    Args:
        in_channels: Input channel dimension
        num_layers: Number of convolutional layers in the block
        growth_rate: Number of output channels per layer (k in DenseNet paper)
        kernel_size: Convolution kernel size (default 3)
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type ('layer', 'batch', or None)
        order: Conv block order - 'nac' or 'cna'
        bottleneck: Use 1x1 bottleneck before 3x3 conv (default True)
        bottleneck_channels: Number of channels in bottleneck layer (default 128)
    
    Returns:
        DenseBlock module
    
    Note:
        Output channels = in_channels + (num_layers * growth_rate)
    """
    return DenseBlock(
        in_channels=in_channels,
        num_layers=num_layers,
        growth_rate=growth_rate,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        order=order,
        bottleneck=bottleneck,
        bottleneck_channels=bottleneck_channels
    )


def transition_block(in_channels, compression=0.5, activation='relu',
                    dropout=0.1, norm_type='layer', order='nac',
                    pooling_type=None, pool_size=2, pool_stride=None, **kwargs):
    """Transition block for compression between dense blocks.
    
    Args:
        in_channels: Input channel dimension
        compression: Compression factor (0 < compression <= 1)
                    Output channels = in_channels * compression
        activation: Activation function
        dropout: Dropout rate
        norm_type: Normalization type
        order: Conv block order - 'nac' or 'cna'
        pooling_type: Pooling type - None (no pooling), 'avg', 'max', or 'softmax'
        pool_size: Pooling kernel size (default 2)
        pool_stride: Pooling stride (default same as pool_size)
    
    Returns:
        TransitionBlock module
    """
    return TransitionBlock(
        in_channels=in_channels,
        compression=compression,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        order=order,
        pooling_type=pooling_type,
        pool_size=pool_size,
        pool_stride=pool_stride
    )


############################################################
# U-Net Blocks
############################################################
class DownresBlockBAK(nn.Module):
    """Downsampling residual block for U-Net encoder.
    
    Two sequential NAC conv blocks with residual connections, followed by max pooling.
    First conv increases channels, second maintains dimension.
    Returns both pooled output and pre-pool output for skip connections.
    """
    def __init__(self, in_channels, channel_increase=128, kernel_size=5,
                 activation='gelu', dropout=0.1, norm_type='layer',
                 pool_size=2, pool_stride=None, order='nac'):
        super().__init__()
        
        self.in_channels = in_channels
        self.mid_channels = in_channels + channel_increase
        self.channel_increase = channel_increase
        
        # First conv block: increase channels by channel_increase
        self.conv1 = conv_block(
            in_channels=in_channels,
            out_channels=self.mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
        
        # Second conv block: maintain channel dimension
        self.conv2 = conv_block(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
        
        # Max pooling for downsampling
        pool_stride = pool_stride if pool_stride is not None else pool_size
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, channels)
            mask: Optional mask tensor
        
        Returns:
            pooled: Downsampled output for next stage (batch, seq_len/2, channels+channel_increase)
            skip: Pre-pool output for skip connection (batch, seq_len, channels+channel_increase)
            mask: Updated mask (if provided)
        """
        # First conv with zero-padded residual
        out1, mask = self.conv1(x, mask)
        # Pad input channels to match output channels
        x_padded = torch.nn.functional.pad(x, (0, self.channel_increase), mode='constant', value=0)
        out1 = out1 + x_padded
        
        # Second conv with residual (same dimensions)
        out2, mask = self.conv2(out1, mask)
        out2 = out2 + out1
        
        # Store skip connection before pooling
        skip = out2
        
        # Apply max pooling
        # Need to transpose for pooling: (batch, seq_len, channels) -> (batch, channels, seq_len)
        out2_transposed = out2.transpose(1, 2)
        pooled = self.pool(out2_transposed)
        pooled = pooled.transpose(1, 2)  # Back to (batch, seq_len, channels)
        
        # Update mask after pooling
        if mask is not None:
            new_seq_len = pooled.size(1)
            if mask.size(1) != new_seq_len:
                mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                mask_pooled = torch.nn.functional.max_pool1d(
                    mask_float,
                    kernel_size=self.pool.kernel_size,
                    stride=self.pool.stride
                )
                mask = mask_pooled.squeeze(1).bool()
                
                # Adjust mask size if needed
                if mask.size(1) > new_seq_len:
                    mask = mask[:, :new_seq_len]
                elif mask.size(1) < new_seq_len:
                    padding_size = new_seq_len - mask.size(1)
                    mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        return pooled, skip, mask

class DownresBlock(nn.Module):
    """Downsampling residual block for U-Net encoder.
    
    Two sequential NAC conv blocks with residual connections, followed by downsampling.
    First conv increases channels, second maintains dimension.
    Downsampling uses strided convolution (if pool_stride is provided) or pooling (default).
    Pooling types: 'max', 'avg', or 'softmax'.
    Returns both downsampled output and pre-downsample output for skip connections.
    """
    def __init__(self, in_channels, channel_increase=128, kernel_size=5,
                 activation='gelu', dropout=0.1, norm_type='layer',
                 pool_size=2, pool_stride=None, pool_type='max', order='nac'):
        super().__init__()
        
        self.in_channels = in_channels
        self.mid_channels = in_channels + channel_increase
        self.channel_increase = channel_increase
        
        # First conv block: increase channels by channel_increase
        self.conv1 = conv_block(
            in_channels=in_channels,
            out_channels=self.mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
        
        # Second conv block: maintain channel dimension
        self.conv2 = conv_block(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
        
        # Downsampling: use strided conv if pool_stride is provided, otherwise use pooling
        self.use_strided_conv = (pool_stride is not None)
        
        if self.use_strided_conv:
            # Use strided convolution for downsampling
            self.downsample_stride = pool_stride
            self.downsample_conv = conv_block(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=kernel_size,
                stride=pool_stride,
                padding='same',
                activation=activation,
                dropout=dropout,
                norm_type=norm_type,
                order=order
            )
        else:
            # Use pooling for downsampling
            self.downsample_stride = pool_size
            self.pool_type = pool_type.lower()
            
            if self.pool_type == 'max':
                self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            elif self.pool_type == 'avg' or self.pool_type == 'average':
                self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
            elif self.pool_type == 'softmax':
                self.pool = layers.SoftmaxPooling1d(kernel_size=pool_size, stride=pool_size)
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}. "
                               f"Supported types: 'max', 'avg'/'average', 'softmax'")
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, channels)
            mask: Optional mask tensor
        
        Returns:
            downsampled: Downsampled output for next stage (batch, seq_len/2, channels+channel_increase)
            skip: Pre-downsample output for skip connection (batch, seq_len, channels+channel_increase)
            mask: Updated mask (if provided)
        """
        # First conv with zero-padded residual
        out1, mask = self.conv1(x, mask)
        # Pad input channels to match output channels
        x_padded = torch.nn.functional.pad(x, (0, self.channel_increase), mode='constant', value=0)
        out1 = out1 + x_padded
        
        # Second conv with residual (same dimensions)
        out2, mask = self.conv2(out1, mask)
        out2 = out2 + out1
        
        # Store skip connection before downsampling
        skip = out2
        
        if self.use_strided_conv:
            # Apply strided convolution for downsampling
            downsampled, mask = self.downsample_conv(out2, mask)
            
            # Add residual connection from out2 (needs downsampling to match dimensions)
            # Downsample out2 using average pooling to match downsampled tensor size
            out2_for_residual = out2.transpose(1, 2)  # (batch, channels, seq_len)
            out2_downsampled = torch.nn.functional.avg_pool1d(
                out2_for_residual,
                kernel_size=self.downsample_stride,
                stride=self.downsample_stride
            )
            out2_downsampled = out2_downsampled.transpose(1, 2)  # (batch, seq_len/2, channels)
            
            # Match dimensions if needed
            if out2_downsampled.size(1) != downsampled.size(1):
                if out2_downsampled.size(1) > downsampled.size(1):
                    out2_downsampled = out2_downsampled[:, :downsampled.size(1), :]
                else:
                    padding_size = downsampled.size(1) - out2_downsampled.size(1)
                    out2_downsampled = torch.nn.functional.pad(out2_downsampled, (0, 0, 0, padding_size), mode='replicate')
            
            downsampled = downsampled + out2_downsampled
        else:
            # Apply max pooling for downsampling
            # Need to transpose for pooling: (batch, seq_len, channels) -> (batch, channels, seq_len)
            out2_transposed = out2.transpose(1, 2)
            downsampled = self.pool(out2_transposed)
            downsampled = downsampled.transpose(1, 2)  # Back to (batch, seq_len, channels)
            
            # Update mask after pooling
            if mask is not None:
                new_seq_len = downsampled.size(1)
                if mask.size(1) != new_seq_len:
                    mask_float = mask.float().unsqueeze(1)  # (batch, 1, seq_len)
                    mask_pooled = torch.nn.functional.max_pool1d(
                        mask_float,
                        kernel_size=self.pool.kernel_size,
                        stride=self.pool.stride
                    )
                    mask = mask_pooled.squeeze(1).bool()
                    
                    # Adjust mask size if needed
                    if mask.size(1) > new_seq_len:
                        mask = mask[:, :new_seq_len]
                    elif mask.size(1) < new_seq_len:
                        padding_size = new_seq_len - mask.size(1)
                        mask = torch.nn.functional.pad(mask, (0, padding_size), value=True)
        
        return downsampled, skip, mask


class UpresBlock(nn.Module):
    """Upsampling residual block for U-Net decoder.
    
    Reduces channels, upsamples by 2x, fuses with encoder skip connection,
    and applies final refinement.
    """
    def __init__(self, in_channels, skip_channels, kernel_size=5,
                 activation='gelu', dropout=0.1, norm_type='layer',
                 residual_scale_init=0.9, order='nac'):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        
        # Learnable residual scale parameter
        self.residual_scale = nn.Parameter(torch.tensor([residual_scale_init]))
        
        # First conv block (width=5): reduce channels from (c+128) to c
        self.conv1 = conv_block(
            in_channels=in_channels,
            out_channels=skip_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
        
        # Second conv block (width=1): process skip connection
        # Note: skip connection comes from encoder and has (skip_channels + channel_increase) channels
        # We need to reduce it to skip_channels to match the upsampled output
        # The skip will be provided by the encoder, so we use lazy initialization
        self.conv_skip = conv_block(
            in_channels=None,  # Will be auto-detected from skip connection
            out_channels=skip_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
        
        # Third conv block (width=5): final refinement
        self.conv2 = conv_block(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order
        )
    
    def forward(self, x, unet_skip, mask=None):
        """
        Args:
            x: Input tensor from previous layer (batch, seq_len, in_channels)
            unet_skip: Skip connection from encoder (batch, target_seq_len, skip_channels)
            mask: Optional mask tensor
        
        Returns:
            out: Output tensor (batch, target_seq_len, skip_channels)
            mask: Updated mask (if provided)
        """
        # First conv with cropped residual
        out, mask = self.conv1(x, mask)
        # Crop input to match skip_channels for residual
        x_cropped = x[:, :, :self.skip_channels]
        out = out + x_cropped
        
        # Upsample by repeating elements along sequence dimension
        # (batch, seq_len, channels) -> (batch, seq_len*2, channels)
        out = torch.repeat_interleave(out, repeats=2, dim=1)
        
        # Scale by learnable parameter
        out = out * self.residual_scale
        
        # Update mask after upsampling
        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=2, dim=1)
        
        # Match the upsampled output length to the skip connection length
        # This handles cases where max pooling caused length mismatch (e.g., odd lengths)
        target_len = unet_skip.size(1)
        current_len = out.size(1)
        
        if current_len != target_len:
            if current_len < target_len:
                # Pad if upsampled output is shorter
                padding_size = target_len - current_len
                out = torch.nn.functional.pad(out, (0, 0, 0, padding_size), mode='replicate')
                if mask is not None:
                    # Convert bool mask to float for padding, then back to bool
                    mask_float = mask.float()
                    mask_float = torch.nn.functional.pad(mask_float, (0, padding_size), mode='replicate')
                    mask = mask_float.bool()
            else:
                # Crop if upsampled output is longer
                out = out[:, :target_len, :]
                if mask is not None:
                    mask = mask[:, :target_len]
        
        # Process skip connection and fuse
        skip_processed, _ = self.conv_skip(unet_skip, mask)
        out = out + skip_processed
        
        # Final refinement with residual
        out_refined, mask = self.conv2(out, mask)
        out = out + out_refined
        
        return out, mask


class UNetTower(nn.Module):
    """U-Net tower with encoder-decoder architecture and skip connections.
    
    Orchestrates downsampling blocks (encoder), optional bottleneck module,
    and upsampling blocks (decoder) with skip connections.
    """
    def __init__(self, in_channels, num_downres=6, channel_increase=128,
                 bottleneck_module=None, bottleneck_params=None,
                 bottleneck_positional_encoding=False, bottleneck_pe_params=None,
                 downres_params=None, upres_params=None,
                 kernel_size=5, activation='gelu',
                 dropout=0.1, norm_type='layer', pool_size=2, 
                 residual_scale_init=0.9, order='nac'):
        super().__init__()
        
        self.num_downres = num_downres
        self.in_channels = in_channels
        self.channel_increase = channel_increase
        self.bottleneck_positional_encoding = bottleneck_positional_encoding
        
        # Prepare downres block parameters
        # Use nested params if provided, otherwise use top-level params
        downres_config = downres_params if downres_params is not None else {}
        downres_defaults = {
            'kernel_size': kernel_size,
            'activation': activation,
            'dropout': dropout,
            'norm_type': norm_type,
            'pool_size': pool_size,
            'order': order
        }
        # Merge: nested params override defaults
        final_downres_params = {**downres_defaults, **downres_config}
        
        # Prepare upres block parameters
        upres_config = upres_params if upres_params is not None else {}
        upres_defaults = {
            'kernel_size': kernel_size,
            'activation': activation,
            'dropout': dropout,
            'norm_type': norm_type,
            'residual_scale_init': residual_scale_init,
            'order': order
        }
        # Merge: nested params override defaults
        final_upres_params = {**upres_defaults, **upres_config}
        
        # Build encoder (downsampling path)
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_downres):
            self.encoder.append(DownresBlock(
                in_channels=current_channels,
                channel_increase=channel_increase,
                **final_downres_params
            ))
            current_channels += channel_increase
        
        # Build positional encoding for bottleneck if specified
        if bottleneck_positional_encoding:
            if bottleneck_pe_params is None:
                bottleneck_pe_params = {}
            # Default d_model is the channel dimension after all downsampling
            if 'd_model' not in bottleneck_pe_params:
                bottleneck_pe_params['d_model'] = current_channels
            # Set dropout from global default if not specified
            if 'dropout' not in bottleneck_pe_params:
                bottleneck_pe_params['dropout'] = dropout
            
            self.bottleneck_pe = layers.PositionalEncoding(**bottleneck_pe_params)
        else:
            self.bottleneck_pe = None
        
        # Build bottleneck module if specified
        if bottleneck_module is not None:
            # If bottleneck_module is a string (block name), instantiate it
            if isinstance(bottleneck_module, str):
                from . import blocks as blocks_module
                if bottleneck_params is None:
                    bottleneck_params = {}
                # Get the factory function from name_func
                if bottleneck_module in name_func:
                    self.bottleneck_module = name_func[bottleneck_module](**bottleneck_params)
                else:
                    raise ValueError(f"Unknown bottleneck module: {bottleneck_module}")
            else:
                # Already instantiated module
                self.bottleneck_module = bottleneck_module
        else:
            self.bottleneck_module = None
        
        # Build decoder (upsampling path)
        self.decoder = nn.ModuleList()
        
        for i in range(num_downres):
            # Decoder processes in reverse order
            # Input channels for upres = current encoder output channels
            # Skip channels = encoder output channels from corresponding level
            skip_channels = current_channels - channel_increase
            
            self.decoder.append(UpresBlock(
                in_channels=current_channels,
                skip_channels=skip_channels,
                **final_upres_params
            ))
            current_channels = skip_channels
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, in_channels)
            mask: Optional mask tensor
        
        Returns:
            out: Output tensor (batch, seq_len, out_channels)
            mask: Updated mask (if provided)
        """
        # Encoder path - collect skip connections
        skip_connections = []
        
        for encoder_block in self.encoder:
            x, skip, mask = encoder_block(x, mask)
            skip_connections.append(skip)
        
        # Apply positional encoding before bottleneck (if enabled)
        if self.bottleneck_pe is not None:
            x = self.bottleneck_pe(x)
        
        # Bottleneck (optional)
        if self.bottleneck_module is not None:
            x = self.bottleneck_module(x, mask)
            if isinstance(x, tuple):
                x, mask = x
        
        # Decoder path - use skip connections in reverse order
        for i, decoder_block in enumerate(self.decoder):
            # Get corresponding skip connection (reverse order)
            skip_idx = self.num_downres - 1 - i
            skip = skip_connections[skip_idx]
            x, mask = decoder_block(x, skip, mask)
        
        return x, mask


def downres_block(in_channels, channel_increase=128, kernel_size=5,
                  activation='gelu', dropout=0.1, norm_type='layer',
                  pool_size=2, pool_stride=None, pool_type='max', order='nac', **kwargs):
    """Downsampling residual block for U-Net encoder.
    
    Args:
        in_channels: Input channel dimension
        channel_increase: Channel increment (default 128)
        kernel_size: Conv kernel size (default 5)
        activation: Activation function (default 'gelu')
        dropout: Dropout rate (default 0.1)
        norm_type: Normalization type (default 'layer')
        pool_size: Pooling kernel size (default 2)
        pool_stride: If provided, use strided convolution instead of pooling
        pool_type: Pooling type - 'max', 'avg', or 'softmax' (default 'max')
        order: Conv block order - 'nac' or 'cna' (default 'nac')
    
    Returns:
        DownresBlock module
    """
    return DownresBlock(
        in_channels=in_channels,
        channel_increase=channel_increase,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        pool_size=pool_size,
        pool_stride=pool_stride,
        pool_type=pool_type,
        order=order
    )


def upres_block(in_channels, skip_channels, kernel_size=5,
                activation='gelu', dropout=0.1, norm_type='layer',
                residual_scale_init=0.9, order='nac', **kwargs):
    """Upsampling residual block for U-Net decoder.
    
    Args:
        in_channels: Input channels (should be skip_channels + 128)
        skip_channels: Expected channels from skip connection
        kernel_size: Main conv kernel size (default 5)
        activation: Activation function (default 'gelu')
        dropout: Dropout rate (default 0.1)
        norm_type: Normalization type (default 'layer')
        residual_scale_init: Initial value for learnable scale (default 0.9)
        order: Conv block order - 'nac' or 'cna' (default 'nac')
    
    Returns:
        UpresBlock module
    """
    return UpresBlock(
        in_channels=in_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        residual_scale_init=residual_scale_init,
        order=order
    )


def unet_tower(in_channels, num_downres=6, channel_increase=128,
               bottleneck_module=None, bottleneck_params=None,
               bottleneck_positional_encoding=False, bottleneck_pe_params=None,
               downres_params=None, upres_params=None,
               kernel_size=5, activation='gelu',
               dropout=0.1, norm_type='layer', pool_size=2,
               residual_scale_init=0.9, order='nac', **kwargs):
    """U-Net tower with encoder-decoder architecture and skip connections.
    
    Args:
        in_channels: Initial channel dimension
        num_downres: Number of downsampling/upsampling stages (default 6)
        channel_increase: Channel increment per stage (default 128)
        bottleneck_module: Optional middle module name (e.g., 'transformer_tower') or instance
        bottleneck_params: Dict of parameters for bottleneck module (if bottleneck_module is a string)
        bottleneck_positional_encoding: Add positional encoding before bottleneck (default False)
        bottleneck_pe_params: Dict of parameters for positional encoding (optional)
        downres_params: Dict of parameters for downres blocks (overrides defaults)
        upres_params: Dict of parameters for upres blocks (overrides defaults)
        kernel_size: Conv kernel size (default 5, used if not in nested params)
        activation: Activation function (default 'gelu', used if not in nested params)
        dropout: Dropout rate (default 0.1, used if not in nested params)
        norm_type: Normalization type (default 'layer', used if not in nested params)
        pool_size: Max pool kernel size (default 2, used if not in nested params)
        residual_scale_init: Initial value for learnable scale (default 0.9, used if not in nested params)
        order: Conv block order - 'nac' or 'cna' (default 'nac', used if not in nested params)
    
    Returns:
        UNetTower module
    
    Example YAML configuration:
        - name: unet_tower
          in_channels: 128
          num_downres: 6
          channel_increase: 128
          downres_params:
            kernel_size: 5
            activation: gelu
            dropout: 0.1
            norm_type: layer
            pool_size: 2
            order: nac
          # Optional: Add positional encoding before transformer bottleneck
          bottleneck_positional_encoding: true
          bottleneck_pe_params:
            max_len: 50000  # Maximum sequence length
            dropout: 0.1    # Dropout after PE
          bottleneck_module: transformer_tower
          bottleneck_params:
            embed_dim: 128
            num_heads: 8
            d_ff: 512
            num_layers: 4
            dropout: 0.1
            activation: gelu
          upres_params:
            kernel_size: 5
            activation: gelu
            dropout: 0.1
            norm_type: layer
            residual_scale_init: 0.9
            order: nac
    """
    return UNetTower(
        in_channels=in_channels,
        num_downres=num_downres,
        channel_increase=channel_increase,
        bottleneck_module=bottleneck_module,
        bottleneck_params=bottleneck_params,
        bottleneck_positional_encoding=bottleneck_positional_encoding,
        bottleneck_pe_params=bottleneck_pe_params,
        downres_params=downres_params,
        upres_params=upres_params,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        pool_size=pool_size,
        residual_scale_init=residual_scale_init,
        order=order
    )


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
    def __init__(self, in_features=None, num_targets=None, hidden_features=None, 
                 dropout=0.1, activation='gelu'):
        super().__init__()
        layer_list = []
        
        if hidden_features is not None:
            # Use lazy initialization for first layer if in_features not specified
            if in_features is None:
                layer_list.append(nn.LazyLinear(hidden_features))
            else:
                layer_list.append(nn.Linear(in_features, hidden_features))
            
            layer_list.extend([
                nn.LayerNorm(hidden_features),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, num_targets)
            ])
        else:
            # Use lazy initialization if in_features not specified
            if in_features is None:
                layer_list.append(nn.LazyLinear(num_targets))
            else:
                layer_list.append(nn.Linear(in_features, num_targets))
        
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.network(x)


def regression_head(in_features=None, num_targets=None, hidden_features=None, 
                    dropout=0.1, activation='gelu', **kwargs):
    """Regression head block.
    
    Args:
        in_features: Input features (None for automatic detection)
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
    def __init__(self, in_features=None, num_tasks=None, num_classes=2, hidden_features=None,
                 dropout=0.1, activation='gelu'):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        
        layer_list = []
        if hidden_features is not None:
            # Use lazy initialization for first layer if in_features not specified
            if in_features is None:
                layer_list.append(nn.LazyLinear(hidden_features))
            else:
                layer_list.append(nn.Linear(in_features, hidden_features))
            
            layer_list.extend([
                nn.LayerNorm(hidden_features),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            final_dim = hidden_features
        else:
            final_dim = in_features
        
        # Output layer
        if num_classes == 2:
            if final_dim is None:
                layer_list.append(nn.LazyLinear(num_tasks))
            else:
                layer_list.append(nn.Linear(final_dim, num_tasks))
        else:
            if final_dim is None:
                layer_list.append(nn.LazyLinear(num_tasks * num_classes))
            else:
                layer_list.append(nn.Linear(final_dim, num_tasks * num_classes))
        
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        logits = self.network(x)
        if self.num_classes > 2:
            logits = logits.reshape(-1, self.num_tasks, self.num_classes)
        return logits


def classification_head(in_features=None, num_tasks=None, num_classes=2, hidden_features=None,
                       dropout=0.1, activation='gelu', **kwargs):
    """Classification head block.
    
    Args:
        in_features: Input features (None for automatic detection)
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
    
    # RoPE Transformers
    'transformer_rope': transformer_rope,
    'transformer_tower_rope': transformer_tower_rope,
    
    # Learnable PE Transformers
    'transformer_learnable': transformer_learnable,
    'transformer_tower_learnable': transformer_tower_learnable,
    
    # Multi-Query Attention Transformers
    'transformer_mqa': transformer_mqa,
    'transformer_tower_mqa': transformer_tower_mqa,
    
    # Convolutions
    'conv_block': conv_block,
    'conv_tower': conv_tower,
    
    # Dense blocks (DenseNet)
    'dense_block': dense_block,
    'transition_block': transition_block,
    
    # U-Net blocks
    'downres_block': downres_block,
    'upres_block': upres_block,
    'unet_tower': unet_tower,
    
    # Residual
    'res_block': res_block,
    'res_tower': res_tower,
    
    # MLP
    'mlp_block': mlp_block,
    
    # Pooling
    'global_pool': global_pool,
    
    # Heads
    'regression_head': regression_head,
    'classification_head': classification_head,
}


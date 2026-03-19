"""
Architectural building blocks for genomic neural networks.
These blocks can be composed via YAML configuration to build custom architectures.
"""

import torch
import torch.nn as nn
import numpy as np
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
# Input Gating Blocks (SGU / DGU)
############################################################

class SNPGatedUnit(nn.Module):
    """
    SNP-wise Gated Unit (SGU) - Feature-wise gating using GLU.

    Applied right after input, before backbone. Uses GLU-style projection
    to learn which features are important for each SNP position.

    Input:  (B, n_snps, 8) diploid one-hot encoding
    Output: (B, n_snps, out_dim)

    Architecture:
        Linear(8 → 2*out_dim) → split → value * sigmoid(gate) → LayerNorm → Dropout
    """

    def __init__(self, in_dim: int = 8, out_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            in_dim: Input dimension (default: 8 for diploid one-hot)
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # GLU projection: input → 2*out_dim (split into value and gate)
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n_snps, in_dim)

        Returns:
            Gated output of shape (B, n_snps, out_dim)
        """
        # Validate input: must be 3D (diploid one-hot encoding)
        if x.dim() != 3:
            raise ValueError(
                f"SGU requires 3D input (diploid one-hot encoding), got {x.dim()}D input "
                f"with shape {x.shape}. Please use encoding_type='diploid_onehot' in your config."
            )

        # Ensure float type (convert from Long/int if needed)
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()

        # Project to 2*out_dim
        proj = self.proj(x)  # (B, n_snps, 2*out_dim)

        # Split into value and gate
        value, gate = proj.chunk(2, dim=-1)  # Each: (B, n_snps, out_dim)

        # GLU: value * sigmoid(gate)
        out = value * torch.sigmoid(gate)

        # Normalize and dropout
        out = self.norm(out)
        out = self.dropout(out)

        return out


def sgu(in_dim=8, out_dim=64, dropout=0.1, **kwargs):
    """SNP-wise Gated Unit (SGU) block.

    Feature-wise gating using GLU for input encoding.

    Args:
        in_dim: Input dimension (default: 8 for diploid one-hot)
        out_dim: Output dimension
        dropout: Dropout rate

    Returns:
        SNPGatedUnit module
    """
    return SNPGatedUnit(in_dim=in_dim, out_dim=out_dim, dropout=dropout)


class DiploidGatedUnit(nn.Module):
    """
    Diploid Gated Unit (DGU) - Allele-combined gating.

    Combines the two alleles (first 4 dims + last 4 dims) before applying
    GLU gating. This exploits the diploid structure of the genotype encoding.

    Input:  (B, n_snps, 8) diploid one-hot encoding
    Output: (B, n_snps, out_dim)

    Architecture:
        Split → allele1 + allele2 → Linear(4 → 2*out_dim) → GLU → LayerNorm → Dropout
    """

    def __init__(self, out_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.out_dim = out_dim

        # GLU projection from combined alleles (4-dim) → 2*out_dim
        self.proj = nn.Linear(4, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n_snps, 8)
               First 4 dims: allele 1, Last 4 dims: allele 2

        Returns:
            Gated output of shape (B, n_snps, out_dim)
        """
        # Validate input: must be 3D (diploid one-hot encoding)
        if x.dim() != 3:
            raise ValueError(
                f"DGU requires 3D input (diploid one-hot encoding), got {x.dim()}D input "
                f"with shape {x.shape}. Please use encoding_type='diploid_onehot' in your config."
            )

        # Validate last dimension must be 8
        if x.shape[-1] != 8:
            raise ValueError(
                f"DGU requires input with last dimension = 8 (diploid one-hot), "
                f"got shape {x.shape}. Please use encoding_type='diploid_onehot' in your config."
            )

        # Ensure float type (convert from Long/int if needed)
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()

        # Split into two alleles
        allele1 = x[..., :4]  # (B, n_snps, 4)
        allele2 = x[..., 4:]  # (B, n_snps, 4)

        # Combine alleles (additive encoding)
        x4 = allele1 + allele2  # (B, n_snps, 4)

        # Project to 2*out_dim
        proj = self.proj(x4)  # (B, n_snps, 2*out_dim)

        # Split into value and gate
        value, gate = proj.chunk(2, dim=-1)  # Each: (B, n_snps, out_dim)

        # GLU: value * sigmoid(gate)
        out = value * torch.sigmoid(gate)

        # Normalize and dropout
        out = self.norm(out)
        out = self.dropout(out)

        return out


def dgu(out_dim=64, dropout=0.1, **kwargs):
    """Diploid Gated Unit (DGU) block.

    Allele-combined gating that exploits diploid genotype structure.

    Args:
        out_dim: Output dimension
        dropout: Dropout rate

    Returns:
        DiploidGatedUnit module
    """
    return DiploidGatedUnit(out_dim=out_dim, dropout=dropout)


############################################################
# Transformer Blocks
############################################################

class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu',
                 use_rope=False, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                 max_position=50000, **kwargs):
        super().__init__()

        # Positional encoding (applied before attention)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == 'sinusoidal':
                self.pos_encoding = layers.PositionalEncoding(
                    d_model=d_model,
                    max_len=max_position,
                    dropout=dropout
                )
            elif positional_encoding_type == 'learnable':
                self.pos_encoding = layers.LearnablePositionalEmbedding(
                    d_model=d_model,
                    max_positions=max_position,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {positional_encoding_type}. "
                                 f"Supported types: 'sinusoidal', 'learnable'")
        else:
            self.pos_encoding = None

        # Choose attention mechanism based on use_rope
        if use_rope:
            self.attention = layers.MultiHeadSelfAttentionRoPE(
                d_model, num_heads, dropout)
        else:
            self.attention = layers.MultiHeadSelfAttention(
                d_model, num_heads, dropout)
        self.ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply positional encoding before attention (if enabled)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


class TransformerBlockTranspose(nn.Module):
    """Transformer block with transpose mode: MHA operates on channel dimension.

    When transpose=True, the input is transposed so that MHA operates on the channel
    dimension instead of the sequence dimension. The feature dimension (seq_len) is
    reduced to d_model after transpose.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu',
                 use_rope=False, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                 max_position=50000, **kwargs):
        """
        Args:
            d_model: Model dimension (channel dimension)
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            use_rope: Whether to use RoPE (default False)
            use_positional_encoding: Whether to use positional encoding (default False)
            positional_encoding_type: Type of positional encoding - 'sinusoidal' or 'learnable' (default 'sinusoidal')
            max_position: Maximum sequence length for positional encoding (default 50000)
        """
        super().__init__()
        self.d_model = d_model

        # Positional encoding (applied before attention)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == 'sinusoidal':
                self.pos_encoding = layers.PositionalEncoding(
                    d_model=d_model,
                    max_len=max_position,
                    dropout=dropout
                )
            elif positional_encoding_type == 'learnable':
                self.pos_encoding = layers.LearnablePositionalEmbedding(
                    d_model=d_model,
                    max_positions=max_position,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {positional_encoding_type}. "
                                 f"Supported types: 'sinusoidal', 'learnable'")
        else:
            self.pos_encoding = None

        # Choose attention mechanism based on use_rope
        if use_rope:
            self.attention = layers.MultiHeadSelfAttentionRoPE(
                d_model, num_heads, dropout)
        else:
            self.attention = layers.MultiHeadSelfAttention(
                d_model, num_heads, dropout)
        self.ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Projection layers to reduce/expand seq_len to/from d_model after transpose
        # These will be initialized dynamically in forward based on actual seq_len
        self.seq_len_proj_down = None  # seq_len -> d_model
        self.seq_len_proj_up = None    # d_model -> seq_len

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask (not used in transpose mode)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Transpose: (batch, seq_len, C) -> (batch, C, seq_len)
        # Now C is the sequence dimension (T), seq_len is the feature dimension (C)
        x = x.transpose(1, 2)  # (batch, C, seq_len)

        # Project seq_len (feature dimension) to d_model if needed
        if seq_len != d_model:
            # Initialize projection layer if not exists or size changed
            if self.seq_len_proj_down is None or self.seq_len_proj_down.in_features != seq_len:
                self.seq_len_proj_down = nn.Linear(
                    seq_len, d_model).to(x.device)
                self.seq_len_proj_up = nn.Linear(d_model, seq_len).to(x.device)
            x = self.seq_len_proj_down(x)  # (batch, C, D)
        # Now x is (batch, C, D), where D is the d_model

        # Apply positional encoding after transpose (if enabled)
        # Positional encoding is applied on the channel dimension (d_model), which is now the sequence dimension
        if self.pos_encoding is not None:
            # Positional encoding expects (batch, seq_len, d_model) and operates on seq_len dimension
            # Current: (batch, d_model, d_model), where first d_model is the sequence dimension
            # We need to transpose so that d_model becomes the seq_len dimension for PositionalEncoding
            # (batch, d_model, d_model) -> (batch, d_model, d_model)
            x = x.transpose(1, 2)
            # Now x is (batch, d_model, d_model), where second d_model is the sequence dimension
            # PositionalEncoding will add pe to the second dimension (seq_len), which is what we want
            # Apply positional encoding on d_model (channel) dimension
            x = self.pos_encoding(x)
            x = x.transpose(1, 2)  # Transpose back: (batch, d_model, d_model)

        # Normalize along feature dimension (last dimension)
        # x is (batch, C, D), where:
        #   - C is channel dimension of the input
        #   - D is model dimension
        x_norm = self.norm1(x)

        attn_out = self.attention(x_norm, mask=None)

        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.norm2(x))

        x = x + self.dropout2(ffn_out)

        # Project back to original seq_len if needed
        if seq_len != d_model:
            x = self.seq_len_proj_up(x)  # (batch, C, seq_len)

        # Transpose back: (batch, C, seq_len) -> (batch, seq_len, C)
        x = x.transpose(1, 2)

        return x


def transformer(d_model, num_heads, d_ff, dropout=0.1, activation='gelu',
                use_rope=False, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                max_position=50000, **kwargs):
    """Single transformer encoder block.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        use_rope: Whether to use RoPE (default False)
        use_positional_encoding: Whether to use positional encoding (default False)
        positional_encoding_type: Type of positional encoding - 'sinusoidal' or 'learnable' (default 'sinusoidal')
        max_position: Maximum sequence length for positional encoding (default 50000)

    Returns:
        TransformerBlock module

    Note:
        Positional encoding is applied before attention in the forward pass.
        If use_rope=True, positional encoding is typically not needed (RoPE already encodes position),
        but both can be used together if desired.
    """
    return TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                            dropout=dropout, activation=activation, use_rope=use_rope,
                            use_positional_encoding=use_positional_encoding,
                            positional_encoding_type=positional_encoding_type,
                            max_position=max_position, **kwargs)


def transformerTranspose(d_model, num_heads, d_ff, dropout=0.1, activation='gelu',
                         use_rope=False, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                         max_position=50000, **kwargs):
    """Single transformer encoder block with transpose mode: MHA operates on channel dimension.

    Args:
        d_model: Model dimension (channel dimension)
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        use_rope: Whether to use RoPE (default False)
        use_positional_encoding: Whether to use positional encoding (default False)
        positional_encoding_type: Type of positional encoding - 'sinusoidal' or 'learnable' (default 'sinusoidal')
        max_position: Maximum sequence length for positional encoding (default 50000)

    Returns:
        TransformerBlockTranspose module

    Note:
        When transpose mode is enabled, the input is transposed so that MHA operates on the channel
        dimension instead of the sequence dimension. The feature dimension (seq_len) is reduced to
        d_model after transpose.
    """
    return TransformerBlockTranspose(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                     dropout=dropout, activation=activation, use_rope=use_rope,
                                     use_positional_encoding=use_positional_encoding,
                                     positional_encoding_type=positional_encoding_type,
                                     max_position=max_position, **kwargs)


class TransformerBlockMqaTranspose(nn.Module):
    """Transformer block with Multi-Query Attention and transpose mode: MQA operates on channel dimension.

    When transpose=True, the input is transposed so that MQA operates on the channel
    dimension instead of the sequence dimension. The feature dimension (seq_len) is
    reduced to d_model after transpose.
    """

    def __init__(self, d_model, num_query_heads=8, qk_head_dim=128, v_head_dim=192,
                 dropout=0.1, max_position=8192, norm_type='layer', activation='gelu',
                 use_rope=True, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                 **kwargs):
        """
        Args:
            d_model: Model dimension (channel dimension)
            num_query_heads: Number of query heads (default 8)
            qk_head_dim: Dimension of each query/key head (default 128)
            v_head_dim: Dimension of value head (default 192)
            dropout: Dropout rate
            max_position: Maximum position for RoPE
            norm_type: 'layer' for LayerNorm (default) or 'rms' for RMSNorm
            activation: Activation function
            use_rope: Whether to use RoPE (default True)
            use_positional_encoding: Whether to use positional encoding (default False)
            positional_encoding_type: Type of positional encoding - 'sinusoidal' or 'learnable' (default 'sinusoidal')
        """
        super().__init__()
        self.d_model = d_model

        # Positional encoding (applied after transpose)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == 'sinusoidal':
                self.pos_encoding = layers.PositionalEncoding(
                    d_model=d_model,
                    max_len=max_position,
                    dropout=dropout
                )
            elif positional_encoding_type == 'learnable':
                self.pos_encoding = layers.LearnablePositionalEmbedding(
                    d_model=d_model,
                    max_positions=max_position,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {positional_encoding_type}. "
                                 f"Supported types: 'sinusoidal', 'learnable'")
        else:
            self.pos_encoding = None

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

        # Multi-Query Attention with RoPE (optional)
        self.attention = layers.MultiQueryAttentionRoPE(
            d_model=d_model,
            num_query_heads=num_query_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dropout=dropout,
            max_position=max_position,
            use_rope=use_rope
        )

        # Feed-forward network (2x expansion with activation, AlphaGenome style)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            layers.Activation(activation),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Projection layers to reduce/expand seq_len to/from d_model after transpose
        # These will be initialized dynamically in forward based on actual seq_len
        self.seq_len_proj_down = None  # seq_len -> d_model
        self.seq_len_proj_up = None    # d_model -> seq_len

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask (not used in transpose mode)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Transpose: (batch, seq_len, C) -> (batch, C, seq_len)
        # Now C is the sequence dimension (T), seq_len is the feature dimension (C)
        x = x.transpose(1, 2)  # (batch, C, seq_len)

        # Project seq_len (feature dimension) to d_model if needed
        if seq_len != d_model:
            # Initialize projection layer if not exists or size changed
            if self.seq_len_proj_down is None or self.seq_len_proj_down.in_features != seq_len:
                self.seq_len_proj_down = nn.Linear(
                    seq_len, d_model).to(x.device)
                self.seq_len_proj_up = nn.Linear(d_model, seq_len).to(x.device)
            x = self.seq_len_proj_down(x)  # (batch, C, D)
        # Now x is (batch, C, D), where D is the d_model

        # Apply positional encoding after transpose (if enabled)
        # Positional encoding is applied on the channel dimension (d_model), which is now the sequence dimension
        if self.pos_encoding is not None:
            # Positional encoding expects (batch, seq_len, d_model) and operates on seq_len dimension
            # Current: (batch, d_model, d_model), where first d_model is the sequence dimension
            # We need to transpose so that d_model becomes the seq_len dimension for PositionalEncoding
            x = x.transpose(1, 2)
            # Now x is (batch, d_model, d_model), where second d_model is the sequence dimension
            # PositionalEncoding will add pe to the second dimension (seq_len), which is what we want
            # Apply positional encoding on d_model (channel) dimension
            x = self.pos_encoding(x)
            x = x.transpose(1, 2)  # Transpose back: (batch, d_model, d_model)

        # Normalize along feature dimension (last dimension)
        # x is (batch, C, D), where:
        #   - C is channel dimension of the input
        #   - D is model dimension
        x_norm = self.norm1(x)

        # Attention block with residual
        attn_out = self.attention(x_norm, mask=None)
        attn_out = self.norm2(attn_out)
        x = x + self.dropout1(attn_out)

        # FFN block with residual
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.norm4(ffn_out)
        x = x + self.dropout2(ffn_out)

        # Project back to original seq_len if needed
        if seq_len != d_model:
            x = self.seq_len_proj_up(x)  # (batch, C, seq_len)

        # Transpose back: (batch, C, seq_len) -> (batch, seq_len, C)
        x = x.transpose(1, 2)

        return x


def transformerMqaTranspose(d_model, num_query_heads=8, qk_head_dim=128, v_head_dim=192,
                            dropout=0.1, max_position=8192, norm_type='layer', activation='gelu',
                            use_rope=True, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                            **kwargs):
    """Single transformer block with Multi-Query Attention and transpose mode: MQA operates on channel dimension.

    Args:
        d_model: Model dimension (channel dimension)
        num_query_heads: Number of query heads (default 8)
        qk_head_dim: Dimension of each query/key head (default 128)
        v_head_dim: Dimension of value head (default 192)
        dropout: Dropout rate
        max_position: Maximum position for RoPE
        norm_type: 'layer' for LayerNorm (default) or 'rms' for RMSNorm
        activation: Activation function
        use_rope: Whether to use RoPE (default True)
        use_positional_encoding: Whether to use positional encoding (default False)
        positional_encoding_type: Type of positional encoding - 'sinusoidal' or 'learnable' (default 'sinusoidal')

    Returns:
        TransformerBlockMqaTranspose module

    Note:
        When transpose mode is enabled, the input is transposed so that MQA operates on the channel
        dimension instead of the sequence dimension. The feature dimension (seq_len) is reduced to
        d_model after transpose.
    """
    return TransformerBlockMqaTranspose(
        d_model=d_model,
        num_query_heads=num_query_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        dropout=dropout,
        max_position=max_position,
        norm_type=norm_type,
        activation=activation,
        use_rope=use_rope,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_type=positional_encoding_type,
        **kwargs
    )


class TransformerTower(nn.Module):
    """Stack of transformer encoder blocks."""

    def __init__(self, embed_dim, num_heads, d_ff, repeat, dropout=0.1, activation='gelu',
                 use_rope=False, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                 max_position=50000, **kwargs):
        super().__init__()

        # Positional encoding (applied before transformer blocks)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == 'sinusoidal':
                self.pos_encoding = layers.PositionalEncoding(
                    d_model=embed_dim,
                    max_len=max_position,
                    dropout=dropout
                )
            elif positional_encoding_type == 'learnable':
                self.pos_encoding = layers.LearnablePositionalEmbedding(
                    d_model=embed_dim,
                    max_positions=max_position,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {positional_encoding_type}. "
                                 f"Supported types: 'sinusoidal', 'learnable'")
        else:
            self.pos_encoding = None

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, d_ff,
                             dropout, activation, use_rope=use_rope)
            for _ in range(repeat)
        ])

    def forward(self, x, mask=None):
        # Apply positional encoding before transformer blocks
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower(embed_dim, num_heads, d_ff, repeat, dropout=0.1, activation='gelu',
                      use_rope=False, use_positional_encoding=False, positional_encoding_type='sinusoidal',
                      max_position=50000, **kwargs):
    """Stack of transformer blocks.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        repeat: Number of times to repeat the transformer block
        dropout: Dropout rate
        activation: Activation function
        use_rope: Whether to use RoPE (default False)
        use_positional_encoding: Whether to use positional encoding (default False)
        positional_encoding_type: Type of positional encoding - 'sinusoidal' or 'learnable' (default 'sinusoidal')
        max_position: Maximum sequence length for positional encoding (default 50000)

    Returns:
        TransformerTower module

    Note:
        Positional encoding is applied before all transformer blocks.
        If use_rope=True, positional encoding is typically not needed (RoPE already encodes position),
        but both can be used together if desired.
    """
    return TransformerTower(embed_dim=embed_dim, num_heads=num_heads, d_ff=d_ff,
                            repeat=repeat, dropout=dropout, activation=activation,
                            use_rope=use_rope, use_positional_encoding=use_positional_encoding,
                            positional_encoding_type=positional_encoding_type,
                            max_position=max_position, **kwargs)


############################################################
# RoPE Transformer Blocks
############################################################

class TransformerBlockRoPE(nn.Module):
    """Single transformer encoder block with Rotary Position Embedding (RoPE)."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = layers.MultiHeadSelfAttentionRoPE(
            d_model, num_heads, dropout)
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

    def __init__(self, embed_dim, num_heads, d_ff, repeat, dropout=0.1, activation='gelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockRoPE(embed_dim, num_heads,
                                 d_ff, dropout, activation)
            for _ in range(repeat)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower_rope(embed_dim, num_heads, d_ff, repeat, dropout=0.1, activation='gelu', **kwargs):
    """Stack of transformer blocks with RoPE.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        repeat: Number of times to repeat the transformer block
        dropout: Dropout rate
        activation: Activation function

    Returns:
        TransformerTowerRoPE module

    Note:
        RoPE encodes position information directly in the attention mechanism,
        so no separate positional encoding layer is needed.
    """
    return TransformerTowerRoPE(embed_dim=embed_dim, num_heads=num_heads, d_ff=d_ff,
                                repeat=repeat, dropout=dropout, activation=activation)


############################################################
# Learnable PE Transformer Blocks
############################################################

class TransformerBlockLearnable(nn.Module):
    """Single transformer encoder block (for use with learnable positional embeddings)."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = layers.MultiHeadSelfAttention(
            d_model, num_heads, dropout)
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

    def __init__(self, embed_dim, num_heads, d_ff, repeat, max_positions=50000,
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
            TransformerBlockLearnable(
                embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(repeat)
        ])

    def forward(self, x, mask=None):
        # Apply learnable positional embeddings
        x = self.pos_embedding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower_learnable(embed_dim, num_heads, d_ff, repeat, max_positions=50000,
                                dropout=0.1, activation='gelu', **kwargs):
    """Stack of transformer blocks with learnable positional embeddings.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        repeat: Number of times to repeat the transformer block
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
                                     repeat=repeat, max_positions=max_positions,
                                     dropout=dropout, activation=activation)


############################################################
# Multi-Query Attention Transformer Blocks
############################################################

class TransformerBlockMQA(nn.Module):
    """Transformer block with Multi-Query Attention and RoPE."""

    def __init__(self, d_model, num_query_heads=8, qk_head_dim=128, v_head_dim=192,
                 dropout=0.1, max_position=8192, norm_type='layer', activation='gelu', use_rope=True, **kwargs):
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

        # Multi-Query Attention with RoPE (optional)
        self.attention = layers.MultiQueryAttentionRoPE(
            d_model=d_model,
            num_query_heads=num_query_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dropout=dropout,
            max_position=max_position,
            use_rope=use_rope
        )

        # Feed-forward network (2x expansion with activation, AlphaGenome style)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            layers.Activation(activation),
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
                    dropout=0.1, max_position=8192, norm_type='layer', activation='gelu', **kwargs):
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
        norm_type=norm_type,
        activation=activation,
        **kwargs
    )


class TransformerTowerMQA(nn.Module):
    """Stack of transformer blocks with Multi-Query Attention and RoPE."""

    def __init__(self, embed_dim, repeat=9, num_query_heads=8, qk_head_dim=128,
                 v_head_dim=192, dropout=0.1, max_position=8192, norm_type='layer', activation='gelu', use_rope=True, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlockMQA(
                d_model=embed_dim,
                num_query_heads=num_query_heads,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                dropout=dropout,
                max_position=max_position,
                norm_type=norm_type,
                activation=activation,
                use_rope=use_rope
            )
            for _ in range(repeat)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_tower_mqa(embed_dim, repeat=9, num_query_heads=8, qk_head_dim=128,
                          v_head_dim=192, dropout=0.1, max_position=8192,
                          norm_type='layer', activation='gelu', use_rope=True, **kwargs):
    """Stack of transformer blocks with Multi-Query Attention.

    Args:
        embed_dim: Embedding dimension (d_model)
        repeat: Number of times to repeat the transformer block (default 9)
        num_query_heads: Number of query heads in MQA (default 8)
        qk_head_dim: Dimension of each query/key head (default 128)
        v_head_dim: Dimension of value head (default 192)
        dropout: Dropout rate
        max_position: Maximum sequence length for RoPE
        norm_type: 'layer' for LayerNorm (default) or 'rms' for RMSNorm
        use_rope: Whether to use RoPE (default True)

    Returns:
        TransformerTowerMQA module

    Note:
        Features:
        - Multi-Query Attention (multiple query heads, 1 shared K/V)
        - RoPE for position encoding (optional, default enabled)
        - Soft-clipping of attention logits for stability
        - Optional RMSNorm instead of LayerNorm
        - FFN with 2x expansion and activation
    """
    return TransformerTowerMQA(
        embed_dim=embed_dim,
        repeat=repeat,
        num_query_heads=num_query_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        dropout=dropout,
        max_position=max_position,
        norm_type=norm_type,
        activation=activation,
        use_rope=use_rope,
        **kwargs
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
            self.conv = nn.LazyConv1d(
                out_channels, kernel_size, stride, padding)
            # For lazy initialization, we'll create residual_proj in forward if needed
            self.residual_proj = None
        else:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size, stride, padding)

            # Projection layer for residual if channels don't match or stride > 1
            if residual and (in_channels != out_channels or stride > 1):
                self.residual_proj = nn.Conv1d(
                    in_channels, out_channels, 1, stride, 0)
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
                self.pool = nn.MaxPool1d(
                    kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower in ['avgpool1d', 'avg', 'average']:
                self.pool = nn.AvgPool1d(
                    kernel_size=pool_size, stride=pool_stride)
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
                self.residual_proj = nn.Conv1d(
                    in_channels, out_channels, 1, stride, 0).to(x.device)

        x = self.conv(x)

        # CNA residual: input + conv_output (add before norm and activation)
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)

            # Ensure identity and x have the same sequence length
            # This handles cases where padding='same' doesn't perfectly preserve length
            # (e.g., with even kernel sizes or when residual_proj changes stride)
            # dimension 2 is sequence length in (batch, channels, seq_len)
            seq_len_x = x.size(2)
            seq_len_identity = identity.size(2)

            if seq_len_identity != seq_len_x:
                if seq_len_identity > seq_len_x:
                    # Crop identity to match x
                    identity = identity[:, :, :seq_len_x]
                else:
                    # Pad identity to match x
                    padding_size = seq_len_x - seq_len_identity
                    identity = torch.nn.functional.pad(
                        identity, (0, padding_size), mode='replicate')

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
                    mask = torch.nn.functional.pad(
                        mask, (0, padding_size), value=True)

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
                        mask = torch.nn.functional.pad(
                            mask, (0, padding_size), value=True)

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
            self.conv = nn.LazyConv1d(
                out_channels, kernel_size, stride, padding)
            # For lazy initialization, we'll create residual_proj in forward if needed
            self.residual_proj = None
        else:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size, stride, padding)

            # Projection layer for residual if channels don't match or stride > 1
            if residual and (in_channels != out_channels or stride > 1):
                self.residual_proj = nn.Conv1d(
                    in_channels, out_channels, 1, stride, 0)
            else:
                self.residual_proj = None

        # Pooling (optional)
        self.pooling_type = pooling_type
        if pooling_type is not None:
            pool_stride = pool_stride if pool_stride is not None else pool_size
            pooling_type_lower = pooling_type.lower()
            if pooling_type_lower in ['maxpool1d', 'max']:
                self.pool = nn.MaxPool1d(
                    kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower in ['avgpool1d', 'avg', 'average']:
                self.pool = nn.AvgPool1d(
                    kernel_size=pool_size, stride=pool_stride)
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

        # Ensure consistent dtype (convert to float32 if needed)
        if x.dtype != torch.float32:
            x = x.float()
            if identity is not None:
                identity = identity.float()

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
                self.residual_proj = nn.Conv1d(
                    in_channels, out_channels, 1, stride, 0).to(x.device)

        # NAC order: Norm -> Activation -> Conv
        x = self.norm(x)
        # Back to (batch, seq_len, channels) for activation
        x = x.transpose(1, 2)
        x = layers.activate(x, self.activation)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # Back to (batch, channels, seq_len) for conv
        x = self.conv(x)

        # NAC residual: identity + conv_output (add after conv so shapes match)
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)

            # Ensure identity and x have the same sequence length
            # This handles cases where padding='same' doesn't perfectly preserve length
            # (e.g., with even kernel sizes or when residual_proj changes stride)
            # dimension 2 is sequence length in (batch, channels, seq_len)
            seq_len_x = x.size(2)
            seq_len_identity = identity.size(2)

            if seq_len_identity != seq_len_x:
                if seq_len_identity > seq_len_x:
                    # Crop identity to match x
                    identity = identity[:, :, :seq_len_x]
                else:
                    # Pad identity to match x
                    padding_size = seq_len_x - seq_len_identity
                    identity = torch.nn.functional.pad(
                        identity, (0, padding_size), mode='replicate')

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
                    mask = torch.nn.functional.pad(
                        mask, (0, padding_size), value=True)

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
                        mask = torch.nn.functional.pad(
                            mask, (0, padding_size), value=True)

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


class DownConvBlock(nn.Module):
    """High-level convolution block that combines feature extraction and downsampling.

    This block consists of two conv_block layers:
    1. Feature extraction conv_block (stride=1): extracts features
    2. Downsampling conv_block (stride=2): reduces sequence dimension by half

    Together, this implements a common pattern: extract features then downsample.
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=5,
                 padding='same', activation='gelu', dropout=0.1, norm_type='layer',
                 order='nac', residual=True):
        """
        Args:
            in_channels: Input channels (None for automatic detection)
            out_channels: Output channels (same for both conv blocks)
            kernel_size: Convolution kernel size (default: 5)
            padding: Padding mode (default: 'same')
            activation: Activation function (default: 'gelu')
            dropout: Dropout rate (default: 0.1)
            norm_type: Normalization type (default: 'layer')
            order: Operation order - 'nac' (default) or 'cna'
            residual: Add residual connections (default: True)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv_block: feature extraction (stride=1)
        self.conv_feat = conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order,
            residual=residual
        )

        # Second conv_block: downsampling (stride=2)
        self.conv_down = conv_block(
            in_channels=out_channels,  # Output of first conv becomes input of second
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            order=order,
            residual=residual
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, channels)
            mask: Optional mask tensor (batch, seq_len)

        Returns:
            out: Output tensor (batch, seq_len/2, channels)
            mask: Updated mask (if provided)
        """
        # Feature extraction
        x, mask = self.conv_feat(x, mask)

        # Downsampling
        x, mask = self.conv_down(x, mask)

        return x, mask


class DownConvTower(nn.Module):
    """Tower of DownConvBlocks that automatically downsamples sequence dimension.

    Stacks multiple DownConvBlocks. Each block reduces the sequence length by half
    (stride=2 in the downsampling conv). 

    If repeat=None (default), automatically determines the number of blocks needed
    based on the actual input sequence length and seq_len_threshold. If repeat is
    specified, uses that fixed number of blocks instead.

    """

    def __init__(self, in_channels=None, out_channels=256, kernel_size=5,
                 padding='same', activation='gelu', dropout=0.1, norm_type='layer',
                 order='nac', residual=True, repeat=None, seq_len_threshold=1500):
        """
        Args:
            in_channels: Input channels (None for automatic detection)
            out_channels: Output channels for all conv blocks (default: 256)
            kernel_size: Convolution kernel size (default: 5)
            padding: Padding mode (default: 'same')
            activation: Activation function (default: 'gelu')
            dropout: Dropout rate (default: 0.1)
            norm_type: Normalization type (default: 'layer')
            order: Operation order - 'nac' (default) or 'cna'
            residual: Add residual connections (default: True)
            repeat: Number of DownConvBlocks to stack (default: None)
                   If None, automatically determines based on input seq_len and seq_len_threshold
                   If specified, uses that fixed number (disables auto calculation)
            seq_len_threshold: Sequence length threshold (default: 1500)
                             Used for auto calculation when repeat=None
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = repeat
        self.seq_len_threshold = seq_len_threshold

        # Store parameters for dynamic block creation
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.norm_type = norm_type
        self.order = order
        self.residual = residual

        # Build blocks
        # If repeat is specified, create that many blocks upfront
        # Otherwise, create a reasonable default number (20 blocks can handle sequences up to ~20M length)
        if repeat is not None:
            num_blocks = repeat
        else:
            # Auto mode: create a reasonable default number
            # 20 blocks can handle sequences up to ~20M length (2^20 * 1500)
            num_blocks = 20

        self.blocks = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_blocks):
            block = DownConvBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                dropout=dropout,
                norm_type=norm_type,
                order=order,
                residual=residual
            )
            self.blocks.append(block)
            current_channels = out_channels

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, channels)
            mask: Optional mask tensor (batch, seq_len)

        Returns:
            out: Output tensor (batch, final_seq_len, channels)
            mask: Updated mask (if provided)

        Note:
            Each block reduces sequence length by half. If repeat=None (auto mode),
            blocks are created dynamically as needed based on the input sequence
            length and seq_len_threshold.
        """
        if self.repeat is not None:
            # Fixed mode: use all blocks up to repeat
            for i in range(self.repeat):
                x, mask = self.blocks[i](x, mask)
        else:
            # Auto mode: use blocks until seq_len <= seq_len_threshold
            current_seq_len = x.size(1)
            block_idx = 0

            while current_seq_len > self.seq_len_threshold and block_idx < len(self.blocks):
                x, mask = self.blocks[block_idx](x, mask)
                current_seq_len = x.size(1)
                block_idx += 1

            # Warn if we ran out of blocks before reaching threshold
            if current_seq_len > self.seq_len_threshold:
                import warnings
                warnings.warn(
                    f"DownConvTower: Sequence length ({current_seq_len}) still exceeds "
                    f"threshold ({self.seq_len_threshold}) after using all {len(self.blocks)} blocks. "
                    f"Consider specifying a larger repeat value."
                )

        return x, mask


def down_conv_block(in_channels=None, out_channels=256, kernel_size=5,
                    padding='same', activation='gelu', dropout=0.1,
                    norm_type='layer', order='nac', residual=True, **kwargs):
    """High-level convolution block combining feature extraction and downsampling.

    Args:
        in_channels: Input channels (None for automatic detection)
        out_channels: Output channels (default: 256)
        kernel_size: Convolution kernel size (default: 5)
        padding: Padding mode (default: 'same')
        activation: Activation function (default: 'gelu')
        dropout: Dropout rate (default: 0.1)
        norm_type: Normalization type (default: 'layer')
        order: Operation order - 'nac' (default) or 'cna'
        residual: Add residual connections (default: True)

    Returns:
        DownConvBlock module

    Note:
        This block consists of two conv_block layers:
        1. Feature extraction (stride=1)
        2. Downsampling (stride=2, reduces sequence length by half)
    """
    return DownConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        order=order,
        residual=residual
    )


def down_conv_tower(in_channels=None, out_channels=256, kernel_size=5,
                    padding='same', activation='gelu', dropout=0.1,
                    norm_type='layer', order='nac', residual=True,
                    repeat=None, seq_len_threshold=1500, **kwargs):
    """Tower of DownConvBlocks that automatically downsamples sequence dimension.

    Args:
        in_channels: Input channels (None for automatic detection)
        out_channels: Output channels for all conv blocks (default: 256)
        kernel_size: Convolution kernel size (default: 5)
        padding: Padding mode (default: 'same')
        activation: Activation function (default: 'gelu')
        dropout: Dropout rate (default: 0.1)
        norm_type: Normalization type (default: 'layer')
        order: Operation order - 'nac' (default) or 'cna'
        residual: Add residual connections (default: True)
        repeat: Number of DownConvBlocks to stack (default: None)
               If None, automatically determines based on input seq_len and seq_len_threshold
               Blocks are created dynamically as needed in forward()
               If specified, uses that fixed number (disables auto calculation)
        seq_len_threshold: Sequence length threshold (default: 1500)
                         Used for auto calculation when repeat=None

    Returns:
        DownConvTower module

    Example:
        If repeat=None, input seq_len=12000, seq_len_threshold=1500:
        - Creates and uses blocks until seq_len <= 1500
        - Block 1: 12000 -> 6000
        - Block 2: 6000 -> 3000
        - Block 3: 3000 -> 1500
        - Stops (1500 <= threshold)
        Final seq_len = 1500

    Note:
        Each DownConvBlock consists of:
        1. Feature extraction conv_block (stride=1)
        2. Downsampling conv_block (stride=2)

        When repeat=None (default), blocks are created dynamically in forward()
        based on the actual input sequence length and seq_len_threshold.
        When repeat is specified, uses that fixed number of blocks.
    """
    return DownConvTower(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        dropout=dropout,
        norm_type=norm_type,
        order=order,
        residual=residual,
        repeat=repeat,
        seq_len_threshold=seq_len_threshold
    )


############################################################
# Dense Blocks (DenseNet)
############################################################

class DenseBlock(nn.Module):
    """Dense block with concatenation of all previous layer outputs.

    Each layer in the block receives all previous feature maps as input.
    Features grow by growth_rate at each layer.
    """

    def __init__(self, in_channels=None, num_layers=4, growth_rate=32,
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
                result = layer[0](concat_features, mask)
                if isinstance(result, tuple):
                    out, mask = result
                else:
                    out = result

                result = layer[1](out, mask)
                if isinstance(result, tuple):
                    out, mask = result
                else:
                    out = result
            else:
                # Direct version
                result = layer(concat_features, mask)
                if isinstance(result, tuple):
                    out, mask = result
                else:
                    out = result

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
                self.pool = nn.AvgPool1d(
                    kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower in ['max', 'maxpool1d']:
                self.pool = nn.MaxPool1d(
                    kernel_size=pool_size, stride=pool_stride)
            elif pooling_type_lower == 'softmax':
                self.pool = layers.SoftmaxPooling1d(
                    kernel_size=pool_size, stride=pool_stride)
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
                        mask = torch.nn.functional.pad(
                            mask, (0, padding_size), value=True)

        return x, mask


def dense_block(in_channels=None, num_layers=4, growth_rate=32, kernel_size=3,
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
        x_padded = torch.nn.functional.pad(
            x, (0, self.channel_increase), mode='constant', value=0)
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
                    mask = torch.nn.functional.pad(
                        mask, (0, padding_size), value=True)

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
                self.pool = nn.MaxPool1d(
                    kernel_size=pool_size, stride=pool_size)
            elif self.pool_type == 'avg' or self.pool_type == 'average':
                self.pool = nn.AvgPool1d(
                    kernel_size=pool_size, stride=pool_size)
            elif self.pool_type == 'softmax':
                self.pool = layers.SoftmaxPooling1d(
                    kernel_size=pool_size, stride=pool_size)
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
        x_padded = torch.nn.functional.pad(
            x, (0, self.channel_increase), mode='constant', value=0)
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
            out2_for_residual = out2.transpose(
                1, 2)  # (batch, channels, seq_len)
            out2_downsampled = torch.nn.functional.avg_pool1d(
                out2_for_residual,
                kernel_size=self.downsample_stride,
                stride=self.downsample_stride
            )
            out2_downsampled = out2_downsampled.transpose(
                1, 2)  # (batch, seq_len/2, channels)

            # Match dimensions if needed
            if out2_downsampled.size(1) != downsampled.size(1):
                if out2_downsampled.size(1) > downsampled.size(1):
                    out2_downsampled = out2_downsampled[:, :downsampled.size(
                        1), :]
                else:
                    padding_size = downsampled.size(
                        1) - out2_downsampled.size(1)
                    out2_downsampled = torch.nn.functional.pad(
                        out2_downsampled, (0, 0, 0, padding_size), mode='replicate')

            downsampled = downsampled + out2_downsampled
        else:
            # Apply max pooling for downsampling
            # Need to transpose for pooling: (batch, seq_len, channels) -> (batch, channels, seq_len)
            out2_transposed = out2.transpose(1, 2)
            downsampled = self.pool(out2_transposed)
            # Back to (batch, seq_len, channels)
            downsampled = downsampled.transpose(1, 2)

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
                        mask = torch.nn.functional.pad(
                            mask, (0, padding_size), value=True)

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
                out = torch.nn.functional.pad(
                    out, (0, 0, 0, padding_size), mode='replicate')
                if mask is not None:
                    # Convert bool mask to float for padding, then back to bool
                    mask_float = mask.float()
                    mask_float = torch.nn.functional.pad(
                        mask_float, (0, padding_size), mode='replicate')
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

            self.bottleneck_pe = layers.PositionalEncoding(
                **bottleneck_pe_params)
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
                    self.bottleneck_module = name_func[bottleneck_module](
                        **bottleneck_params)
                else:
                    raise ValueError(
                        f"Unknown bottleneck module: {bottleneck_module}")
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

    def __init__(self, d_model, d_ff, repeat, dropout=0.1, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_ff, dropout, activation)
            for _ in range(repeat)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def res_tower(d_model, d_ff, repeat, dropout=0.1, activation='relu', **kwargs):
    """Stack of residual blocks.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        repeat: Number of times to repeat the residual block
        dropout: Dropout rate
        activation: Activation function

    Returns:
        ResidualTower module
    """
    return ResidualTower(d_model=d_model, d_ff=d_ff, repeat=repeat,
                         dropout=dropout, activation=activation)


############################################################
# MLP Blocks
############################################################

class MLPBlock(nn.Module):
    """MLP block with multiple layers.

    Args:
        d_model: Input dimension
        d_ff: Hidden dimension
        num_layers: Number of layers (default 3)
        dropout: Dropout rate (default 0.1)
        activation: Activation function (default 'gelu')
        out_features: Output dimension (default: same as d_ff)
        norm_type: Normalization type - 'layer' for LayerNorm (default), None for no norm
    """

    def __init__(self, d_model, d_ff, num_layers=3, dropout=0.1, activation='gelu',
                 out_features=None, norm_type='layer'):
        super().__init__()
        layer_list = []
        current_dim = d_model
        output_dim = out_features if out_features is not None else d_ff

        for i in range(num_layers):
            # Last layer uses output_dim, others use d_ff
            target_dim = output_dim if i == num_layers - 1 else d_ff

            layer_list.append(nn.Linear(current_dim, target_dim))

            # Add normalization after linear (except for last layer if no norm)
            if norm_type == 'layer':
                layer_list.append(nn.LayerNorm(target_dim))

            # Add activation (except for last layer)
            if i < num_layers - 1 and activation:
                layer_list.append(layers.Activation(activation))

            # Add dropout (except for last layer)
            if i < num_layers - 1 and dropout > 0:
                layer_list.append(nn.Dropout(dropout))

            current_dim = target_dim

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)


def mlp_block(in_features=None, hidden_features=None, out_features=None, num_layers=3,
              dropout=0.1, activation='gelu', norm_type='layer', **kwargs):
    """MLP block.

    Args:
        in_features: Input dimension (required)
        hidden_features: Hidden dimension (required)
        out_features: Output dimension (default: same as hidden_features)
        num_layers: Number of layers (default 3)
        dropout: Dropout rate (default 0.1)
        activation: Activation function (default 'gelu')
        norm_type: Normalization type - 'layer' for LayerNorm (default), None for no norm

    Returns:
        MLPBlock module

    Example:
        - name: mlp_block
          in_features: 128
          hidden_features: 256
          out_features: 128
          num_layers: 2
          dropout: 0.1
          activation: gelu
          norm_type: layer
    """
    if in_features is None:
        raise ValueError("mlp_block requires 'in_features' parameter")
    if hidden_features is None:
        raise ValueError("mlp_block requires 'hidden_features' parameter")

    return MLPBlock(d_model=in_features, d_ff=hidden_features, num_layers=num_layers,
                    dropout=dropout, activation=activation, out_features=out_features,
                    norm_type=norm_type)


############################################################
# Pooling Blocks
############################################################

def global_pool(pool_type='mean', d_model=None, pool_axis=1, dropout=0.1, **kwargs):
    """Global pooling block.

    Args:
        pool_type: Pooling type ('mean', 'max', 'attention')
        d_model: Model dimension (required for attention pooling)
        pool_axis: Pooling axis - 1 for sequence dimension (default), 2 for feature dimension
        dropout: Dropout rate (default 0.1)

    Returns:
        GlobalPooling module
    """
    if pool_type == 'attention' and d_model is None:
        raise ValueError("d_model required for attention pooling")
    return layers.GlobalPooling(d_model=d_model or 128, pool_type=pool_type, pool_axis=pool_axis, dropout=dropout)


def multi_head_pool(d_model, num_heads=4, dropout=0.1, pool_axis=1, **kwargs):
    """Multi-head pooling block.

    Combines mean, max, attention, and std pooling strategies in parallel,
    then fuses them to capture different statistical properties.

    Args:
        d_model: Model dimension
        num_heads: Number of pooling heads (default 4: mean, max, attention, std)
        dropout: Dropout rate
        pool_axis: Pooling axis - 1 for sequence dimension (default), 2 for feature dimension

    Returns:
        MultiHeadPooling module
    """
    return layers.MultiHeadPooling(d_model=d_model, num_heads=num_heads, dropout=dropout, pool_axis=pool_axis)


def learnable_query_pool(d_model, num_queries=1, num_layers=2, dropout=0.1, pool_axis=1, **kwargs):
    """Learnable query pooling block (Set2Set style).

    Uses learnable query vectors to attend to the sequence or features, allowing
    the model to learn task-specific aggregation patterns.

    Args:
        d_model: Model dimension
        num_queries: Number of learnable query vectors (default 1, only used when pool_axis=1)
        num_layers: Number of LSTM layers for iterative refinement (default 2)
                   Set to 0 to disable LSTM (use_lstm will be False)
                   When num_layers > 0, LSTM is enabled (use_lstm will be True)
        dropout: Dropout rate
        pool_axis: Pooling axis - 1 for sequence dimension (default), 2 for feature dimension

    Returns:
        LearnableQueryPooling module

    Note:
        The use_lstm flag is automatically determined from num_layers:
        - num_layers > 0: use_lstm = True, applies LSTM refinement
        - num_layers = 0: use_lstm = False, no LSTM refinement
    """
    return layers.LearnableQueryPooling(
        d_model=d_model,
        num_queries=num_queries,
        num_layers=num_layers,
        dropout=dropout,
        pool_axis=pool_axis
    )


def hierarchical_pool(d_model, window_size=64, stride=32, dropout=0.1, **kwargs):
    """Hierarchical pooling block.

    First performs local pooling over sliding windows, then applies
    global attention pooling to capture multi-scale information.

    Args:
        d_model: Model dimension
        window_size: Size of sliding window for local pooling (default 64)
        stride: Stride for sliding window (default 32)
        dropout: Dropout rate

    Returns:
        HierarchicalPooling module
    """
    return layers.HierarchicalPooling(
        d_model=d_model,
        window_size=window_size,
        stride=stride,
        dropout=dropout
    )


def transformer_pool(d_model, num_layers=2, num_heads=8, dim_feedforward=None, dropout=0.1, pool_axis=1, **kwargs):
    """Transformer-based pooling block (BERT CLS style).

    Uses a learnable pooling token that attends to all sequence positions
    or feature dimensions through transformer layers, similar to BERT's CLS token.

    Args:
        d_model: Model dimension
        num_layers: Number of transformer encoder layers (default 2)
        num_heads: Number of attention heads (default 8, only used when pool_axis=1)
        dim_feedforward: Feed-forward dimension (default: d_model * 4 for pool_axis=1, 
                        feature_embed_dim * 2 for pool_axis=2)
        dropout: Dropout rate
        pool_axis: Pooling axis - 1 for sequence dimension (default), 2 for feature dimension

    Returns:
        TransformerPooling module

    Note:
        When pool_axis=2, a smaller transformer is used for efficiency:
        - Features are projected to a smaller embedding space (min(64, d_model))
        - Number of attention heads is adjusted based on embedding dimension
        - CLS token aggregates information across feature dimensions for each position
    """
    return layers.TransformerPooling(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pool_axis=pool_axis
    )


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
                layer_list.append(
                    nn.Linear(final_dim, num_tasks * num_classes))

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
# Fusion Blocks (for Multi-Branch Architectures)
############################################################

class GatedFusionBlock(nn.Module):
    """
    Gated fusion block for combining multiple branch outputs.

    Each branch is projected to a common dimension, then combined using
    learned gating weights with optional residual connections.
    """

    def __init__(self, fusion_dim, num_branches, dropout=0.1, which_branch_to_add=None, branch_dims=None):
        """
        Args:
            fusion_dim: Common dimension for fusion
            num_branches: Number of branches to fuse
            dropout: Dropout rate
            which_branch_to_add: Which branch to add as residual.
                                - None (default): no residual connection
                                - 0: add first branch (snp)
                                - 1: add second branch (indel)
                                - 2: add third branch (sv)
                                - etc.
            branch_dims: Optional list of input dimensions for each branch.
                        If provided, branch_projections will be created at init time.
                        If None, branch_projections will be created dynamically on first forward.
        """
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_branches = num_branches
        self.which_branch_to_add = which_branch_to_add

        # Per-branch projection layers
        self.branch_projections = nn.ModuleList()
        
        # If branch_dims is provided, create projections at init time
        if branch_dims is not None:
            for in_dim in branch_dims:
                self.branch_projections.append(nn.Linear(in_dim, fusion_dim))

        # Gating mechanism: learn importance weights for each branch
        self.gate_fc = nn.Linear(fusion_dim * num_branches, num_branches)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, branch_outputs):
        """
        Args:
            branch_outputs: List of tensors from different branches
                           Each can have shape (batch, seq_len_i, dim_i)

        Returns:
            Fused output: (batch, fusion_dim) or (batch, seq_len, fusion_dim)
        """
        batch_size = branch_outputs[0].shape[0]

        # Ensure we have the right number of projection layers
        if len(self.branch_projections) == 0:
            for branch_out in branch_outputs:
                in_dim = branch_out.shape[-1]
                self.branch_projections.append(
                    nn.Linear(in_dim, self.fusion_dim).to(branch_out.device)
                )

        # Project each branch to common dimension and pool to (batch, fusion_dim)
        projected = []
        for i, branch_out in enumerate(branch_outputs):
            # branch_out: (batch, seq_len, dim) or (batch, dim)
            if branch_out.dim() == 3:
                # Pool sequence dimension: use mean pooling
                pooled = branch_out.mean(dim=1)  # (batch, dim)
            else:
                pooled = branch_out

            # Project to fusion dimension
            proj = self.branch_projections[i](pooled)  # (batch, fusion_dim)
            projected.append(proj)

        # Stack projections
        # (batch, num_branches, fusion_dim)
        stacked = torch.stack(projected, dim=1)

        # Compute gating weights
        # (batch, num_branches * fusion_dim)
        concat_features = stacked.view(batch_size, -1)
        gate_weights = torch.softmax(self.gate_fc(
            concat_features), dim=-1)  # (batch, num_branches)

        # Apply gating: weighted sum
        gate_weights = gate_weights.unsqueeze(-1)  # (batch, num_branches, 1)
        fused = (stacked * gate_weights).sum(dim=1)  # (batch, fusion_dim)

        # Optional residual: add specified branch
        if self.which_branch_to_add is not None:
            branch_idx = self.which_branch_to_add
            if branch_idx < len(projected):
                fused = fused + projected[branch_idx]
            else:
                raise ValueError(f"which_branch_to_add={branch_idx} but only {len(projected)} branches available")

        # Normalize and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class CrossAttentionFusionBlock(nn.Module):
    """
    Cross-attention fusion block for inter-branch communication.

    Allows one branch (query) to attend to patterns in other branches (key/value).
    Useful for letting SV branch sense local SNP/INDEL patterns.

    By default, branch 2 (SV) is used as query to attend to SNP and INDEL branches.
    """

    def __init__(self, d_model, num_heads=4, dropout=0.1, which_branch_as_query=2, num_branches=3, branch_dims=None):
        """
        Args:
            d_model: Common dimension for query/key/value
            num_heads: Number of attention heads
            dropout: Dropout rate
            which_branch_as_query: Which branch index (0-based) to use as query.
                                  Default: 2 (SV branch)
                                  - 0: use first branch (snp) as query
                                  - 1: use second branch (indel) as query
                                  - 2: use third branch (sv) as query
                                  - etc.
            num_branches: Total number of branches (default: 3)
            branch_dims: Optional list of input dimensions for each branch.
                        If provided, branch_projections will be created at init time.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.which_branch_as_query = which_branch_as_query
        self.num_branches = num_branches

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Per-branch projection layers
        self.branch_projections = nn.ModuleList()
        
        # If branch_dims is provided, create projections at init time
        if branch_dims is not None:
            for in_dim in branch_dims:
                self.branch_projections.append(nn.Linear(in_dim, d_model))

    def forward(self, *branches):
        """
        Args:
            branches: List of branch tensors [(batch, seq_len_i, d_model_i), ...]
                     Uses which_branch_as_query to select query branch automatically.

        Returns:
            Enhanced query: (batch, seq_len_q, d_model)
        """
        batch_size = branches[0].shape[0]

        # Ensure we have the right number of projection layers
        if len(self.branch_projections) == 0:
            for branch_out in branches:
                in_dim = branch_out.shape[-1]
                self.branch_projections.append(
                    nn.Linear(in_dim, self.d_model).to(branch_out.device)
                )

        # Project each branch to common d_model
        projected = []
        for i, branch_out in enumerate(branches):
            proj = self.branch_projections[i](branch_out)  # (batch, ..., d_model)
            projected.append(proj)

        # Select query branch
        query_idx = self.which_branch_as_query
        query_branch = projected[query_idx]
        key_value_branches = [projected[i] for i in range(len(projected)) if i != query_idx]

        # Check if input is 2D or 3D
        is_2d = query_branch.dim() == 2

        # For 2D input: pool to (batch, d_model) before attention
        if is_2d:
            query_branch = query_branch.unsqueeze(1)  # (batch, 1, d_model)
            key_value_branches = [kv.unsqueeze(1) for kv in key_value_branches]

        # Concatenate all key/value branches along sequence dimension
        # (batch, sum(seq_len_i), d_model)
        kv = torch.cat(key_value_branches, dim=1)

        # Apply cross-attention
        attn_out, _ = self.cross_attention(
            query=query_branch,
            key=kv,
            value=kv
        )

        # For 2D input: pool back to (batch, d_model)
        if is_2d:
            attn_out = attn_out.squeeze(1)  # (batch, d_model)
            query_branch = query_branch.squeeze(1)

        # Residual connection and normalization
        output = self.norm(query_branch + self.dropout(attn_out))

        return output


def gated_fusion(fusion_dim, num_branches, dropout=0.1, which_branch_to_add=None, **kwargs):
    """
    Factory function for GatedFusionBlock.

    Args:
        fusion_dim: Common dimension for fusion
        num_branches: Number of branches to fuse
        dropout: Dropout rate
        which_branch_to_add: Which branch to add as residual.
                            - None (default): no residual connection
                            - 0: add first branch (snp)
                            - 1: add second branch (indel)
                            - 2: add third branch (sv)
                            - etc.

    Returns:
        GatedFusionBlock module

    Example YAML:
        - name: gated_fusion
          fusion_dim: 128
          num_branches: 3
          dropout: 0.1
          which_branch_to_add: 0  # Add first branch (SNP) as residual
    """
    return GatedFusionBlock(
        fusion_dim=fusion_dim,
        num_branches=num_branches,
        dropout=dropout,
        which_branch_to_add=which_branch_to_add
    )


def cross_attention_fusion(d_model, num_heads=4, dropout=0.1, which_branch_as_query=2, num_branches=3, **kwargs):
    """
    Factory function for CrossAttentionFusionBlock.

    By default, branch 2 (SV) is used as query to attend to SNP and INDEL branches.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        which_branch_as_query: Which branch index (0-based) to use as query.
                              Default: 2 (SV branch)
                              - 0: use first branch (snp) as query
                              - 1: use second branch (indel) as query
                              - 2: use third branch (sv) as query
                              - etc.
        num_branches: Total number of branches (default: 3)

    Returns:
        CrossAttentionFusionBlock module

    Example YAML:
        # Default: SV branch (2) attends to SNP (0) and INDEL (1)
        - name: cross_attention_fusion
          d_model: 128
          num_heads: 4

        # Custom: SNP branch (0) attends to INDEL (1) and SV (2)
        - name: cross_attention_fusion
          d_model: 128
          num_heads: 4
          which_branch_as_query: 0
          num_branches: 3
    """
    return CrossAttentionFusionBlock(
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        which_branch_as_query=which_branch_as_query,
        num_branches=num_branches
    )


class PerceiverCrossFusionBlock(nn.Module):
    """
    Perceiver-style cross-attention fusion block for inter-branch communication.

    Fusion process:
      1. Project each branch tokens to fusion_dim
      2. Concat tokens across branches
      3. K learnable latent tokens cross-attend to all tokens
      4. Pool latents to a single fused vector

    This approach allows all branches to interact through learnable latent queries,
    rather than having one branch attend to others. More symmetric and expressive
    than CrossAttentionFusionBlock.

    Args:
        d_model: Common dimension for fusion (and latent tokens)
        num_heads: Number of attention heads
        dropout: Dropout rate
        latent_k: Number of learnable latent tokens (default: 2)
        pool_latents: How to pool latent tokens - 'mean', 'max', 'first' (default: 'mean')
    """

    def __init__(self, d_model, num_heads=4, dropout=0.1, latent_k=2, pool_latents='mean', branch_dims=None):
        """
        Args:
            d_model: Common dimension for fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
            latent_k: Number of learnable latent tokens (default: 2)
            pool_latents: Pooling strategy for latents - 'mean', 'max', 'first' (default: 'mean')
            branch_dims: Optional list of input dimensions for each branch.
                        If provided, branch_projections will be created at init time.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_k = latent_k
        self.pool_latents = pool_latents

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(latent_k, d_model) * 0.02)

        # Cross-attention: query=latents, key/value=branch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalization layers (pre-norm architecture)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

        # Feed-forward network on latents
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln_out = nn.LayerNorm(d_model)

        # Per-branch projection layers
        self.branch_projections = nn.ModuleList()
        
        # If branch_dims is provided, create projections at init time
        if branch_dims is not None:
            for in_dim in branch_dims:
                self.branch_projections.append(nn.Linear(in_dim, d_model))

    def _build_projs_if_needed(self, branch_outputs):
        """Build projection layers if not already built."""
        if len(self.branch_projections) == 0:
            for x in branch_outputs:
                in_dim = x.shape[-1]
                self.branch_projections.append(
                    nn.Linear(in_dim, self.d_model).to(x.device)
                )

    def forward(self, *branches):
        """
        Args:
            branches: List of branch tensors [(batch, seq_len_i, d_model_i), ...]
                     Each tensor can be 3D (batch, seq_len, dim) or 2D (batch, dim).

        Returns:
            Fused output: (batch, d_model)
        """
        batch_size = branches[0].shape[0]

        # Ensure projection layers are built
        self._build_projs_if_needed(branches)

        # Project each branch to common d_model
        proj_tokens = []
        for i, branch_out in enumerate(branches):
            # Handle 2D input (already pooled): (batch, d_model) -> (batch, 1, d_model)
            if branch_out.dim() == 2:
                x = branch_out.unsqueeze(1)  # (batch, 1, d_model_i)
            else:
                x = branch_out  # (batch, seq_len_i, d_model_i)
            x = self.branch_projections[i](x)  # (batch, ..., d_model)
            proj_tokens.append(x)

        # Concatenate all branch tokens along sequence dimension
        # (batch, sum(seq_len_i), d_model)
        tokens = torch.cat(proj_tokens, dim=1)

        # Prepare latents for batch: (K, D) -> (B, K, D)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Pre-norm architecture
        q = self.ln_q(latents)  # (B, K, D)
        kv = self.ln_kv(tokens)  # (B, L_total, D)

        # Cross-attention: latents attend to all branch tokens
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        latents = latents + attn_out  # Residual connection

        # Feed-forward network on latents
        latents = latents + self.ffn(self.ln_out(latents))

        # Pool latent tokens to single fused vector
        if self.pool_latents == 'mean':
            fused = latents.mean(dim=1)  # (B, D)
        elif self.pool_latents == 'max':
            fused = latents.max(dim=1)[0]  # (B, D)
        elif self.pool_latents == 'first':
            fused = latents[:, 0, :]  # (B, D)
        else:
            raise ValueError(f"Unknown pool_latents: {self.pool_latents}")

        return fused


def perceiver_cross_fusion(d_model, num_heads=4, dropout=0.1, latent_k=2, pool_latents='mean', **kwargs):
    """
    Factory function for PerceiverCrossFusionBlock.

    Perceiver-style cross-attention fusion:
      1. Project each branch tokens to fusion_dim
      2. Concat tokens across branches
      3. K learnable latent tokens cross-attend to all tokens
      4. Pool latents to a single fused vector

    This approach allows all branches to interact through learnable latent queries,
    rather than having one branch attend to others. More symmetric and expressive.

    Args:
        d_model: Model dimension (common dimension for fusion)
        num_heads: Number of attention heads
        dropout: Dropout rate
        latent_k: Number of learnable latent tokens (default: 2)
        pool_latents: Pooling strategy for latents (default: 'mean')
            - 'mean': mean pooling across latent tokens
            - 'max': max pooling across latent tokens
            - 'first': use first latent token only

    Returns:
        PerceiverCrossFusionBlock module

    Example YAML:
        # Default settings
        - name: perceiver_cross_fusion
          d_model: 256
          num_heads: 4
          latent_k: 2
          pool_latents: mean

        # More latent tokens for more expressive fusion
        - name: perceiver_cross_fusion
          d_model: 256
          num_heads: 4
          latent_k: 4
          pool_latents: mean
    """
    return PerceiverCrossFusionBlock(
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        latent_k=latent_k,
        pool_latents=pool_latents
    )


############################################################
# Sample Structure Blocks
############################################################

def sample_structure_encoder(d_model, structure_dim=64, use_cross_sample_attention=True,
                             pooling_type='attention', pool_axis=1, dropout=0.1, **kwargs):
    """
    Factory function for SampleStructureEncoder.

    Learnable sample structure encoding that captures kinship/population structure.
    Instead of positional encoding, this learns sample-level embeddings based on
    genotype patterns and injects them into the feature space.

    Args:
        d_model: Model dimension (must match input feature dimension)
        structure_dim: Dimension for structure embedding (default: 64)
        use_cross_sample_attention: Whether to use cross-sample attention for kinship (default: True)
        pooling_type: Type of pooling for sample representation:
            - 'attention': Attention-based pooling using GlobalPooling (default, most expressive)
            - 'multi_head': MultiHeadPooling combining mean, max, attention, std (best expressiveness)
            - 'mean_max': Combination of mean and max pooling
            - 'mean': Simple mean pooling (fallback)
        pool_axis: Pooling axis (1=sequence, 2=feature, default: 1)
        dropout: Dropout rate (default: 0.1)

    Returns:
        SampleStructureEncoder module
    """
    return layers.SampleStructureEncoder(
        d_model=d_model,
        structure_dim=structure_dim,
        use_cross_sample_attention=use_cross_sample_attention,
        pooling_type=pooling_type,
        pool_axis=pool_axis,
        dropout=dropout
    )


def sample_cls_token(d_model, num_layers=2, num_heads=8, dim_feedforward=None,
                     pooling_type='attention', pool_axis=1, dropout=0.1, **kwargs):
    """
    Factory function for SampleCLSToken.

    Sample-level CLS token that learns population structure.
    Similar to sequence-level CLS token (like BERT), but operates at sample level.
    Each sample gets a learnable CLS token that aggregates information across
    the sequence and interacts with other samples' CLS tokens to learn kinship.

    Args:
        d_model: Model dimension (must match input feature dimension)
        num_layers: Number of transformer layers (default: 2)
        num_heads: Number of attention heads (default: 8)
        dim_feedforward: Feed-forward dimension (default: d_model * 4)
        pooling_type: Type of pooling for sample representation:
            - 'attention': Attention-based pooling using GlobalPooling (default, most expressive)
            - 'multi_head': MultiHeadPooling combining mean, max, attention, std (best expressiveness)
            - 'mean_max': Combination of mean and max pooling
            - 'mean': Simple mean pooling (fallback)
        pool_axis: Pooling axis (1=sequence, 2=feature, default: 1)
        dropout: Dropout rate (default: 0.1)

    Returns:
        SampleCLSToken module
    """
    return layers.SampleCLSToken(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        pooling_type=pooling_type,
        pool_axis=pool_axis,
        dropout=dropout
    )


def linear_block(out_features, bias=True, transpose=False, **kwargs):
    """Linear layer block with optional transpose support.

    Args:
        out_features: Output feature dimension (required)
        bias: Whether to use bias (default True)
        transpose: If True, transpose input from (batch, seq_len, C) to (batch, C, seq_len)
                   before applying linear layer, then transpose back. Default False.

    Returns:
        LinearBlock module

    Note:
        Uses LazyLinear to automatically infer in_features from input shape.
    """
    if transpose:
        class TransposeLinear(nn.Module):
            def __init__(self, out_features, bias=True):
                super().__init__()
                self.linear = nn.LazyLinear(out_features, bias=bias)

            def forward(self, x):
                # x: (batch, seq_len, C)
                # Transpose: (batch, seq_len, C) -> (batch, C, seq_len)
                x = x.transpose(1, 2)  # (batch, C, seq_len)
                # Apply linear: Linear operates on last dimension
                # (batch, C, seq_len) -> (batch, C, out_features)
                x = self.linear(x)  # (batch, C, out_features)
                # Transpose back: (batch, C, out_features) -> (batch, out_features, C)
                x = x.transpose(1, 2)  # (batch, out_features, C)
                return x

        return TransposeLinear(out_features, bias=bias)
    else:
        return nn.LazyLinear(out_features, bias=bias)

############################################################
# Block Dictionary
############################################################


name_func = {
    # Embeddings
    'snp_embedding': snp_embedding,
    'positional_encoding': positional_encoding,

    # Input Gating
    'sgu': sgu,
    'dgu': dgu,

    # Transformers
    'transformer': transformer,
    'transformer_tower': transformer_tower,

    # Transpose Transformers
    'transformerTranspose': transformerTranspose,

    # RoPE Transformers
    'transformer_rope': transformer_rope,
    'transformer_tower_rope': transformer_tower_rope,

    # Learnable PE Transformers
    'transformer_learnable': transformer_learnable,
    'transformer_tower_learnable': transformer_tower_learnable,

    # Multi-Query Attention Transformers
    'transformer_mqa': transformer_mqa,
    'transformer_tower_mqa': transformer_tower_mqa,
    'transformerMqaTranspose': transformerMqaTranspose,

    # Convolutions
    'conv_block': conv_block,
    'conv_tower': conv_tower,
    'down_conv_block': down_conv_block,
    'down_conv_tower': down_conv_tower,

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
    'multi_head_pool': multi_head_pool,
    'learnable_query_pool': learnable_query_pool,
    'hierarchical_pool': hierarchical_pool,
    'transformer_pool': transformer_pool,

    # Heads
    'regression_head': regression_head,
    'classification_head': classification_head,

    # Fusion (Multi-Branch)
    'gated_fusion': gated_fusion,
    'cross_attention_fusion': cross_attention_fusion,
    'perceiver_cross_fusion': perceiver_cross_fusion,

    # Sample Structure Blocks
    'sample_structure_encoder': sample_structure_encoder,
    'sample_cls_token': sample_cls_token,

    'linear_block': linear_block,
}

"""
Atomic neural network layers for genomic data processing.
Provides reusable layer components for building complex architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


############################################################
# Activation Functions
############################################################

def activate(x, activation='relu'):
    """Apply activation function.
    
    Args:
        x: Input tensor
        activation: Activation type ('relu', 'gelu', 'sigmoid', 'tanh', 'linear', None)
    
    Returns:
        Activated tensor
    """
    if activation == 'relu':
        return F.relu(x)
    elif activation == 'gelu':
        return F.gelu(x)
    elif activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'tanh':
        return torch.tanh(x)
    elif activation == 'linear' or activation is None:
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")


############################################################
# Genomic-Specific Embeddings
############################################################

class SNPEmbedding(nn.Module):
    """
    Embedding layer for SNP data with special handling for missing values.
    Maps {0, 1, 2, 3} to dense embeddings.
    """
    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 4 possible values: 0(AA), 1(Aa), 2(aa), 3(missing)
        self.embedding = nn.Embedding(4, embed_dim, padding_idx=3)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) with values in {0, 1, 2, 3}
        Returns:
            (batch, seq_len, embed_dim)
        """
        embedded = self.embedding(x)
        return self.dropout(embedded)


class IndelEmbedding(nn.Module):
    """Placeholder for INDEL-specific embedding (future extension)."""
    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Implement INDEL-specific encoding
        raise NotImplementedError("INDEL embedding not yet implemented")


class SVEmbedding(nn.Module):
    """Placeholder for structural variant embedding (future extension)."""
    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Implement SV-specific encoding
        raise NotImplementedError("SV embedding not yet implemented")


class CombinedGenomicEmbedding(nn.Module):
    """Combined embedding for SNP + INDEL + SV (future extension)."""
    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Implement combined genomic variant encoding
        raise NotImplementedError("Combined genomic embedding not yet implemented")


############################################################
# Positional Encoding
############################################################

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model, max_len=50000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


############################################################
# Attention Mechanisms
############################################################

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE)."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # RoPE module
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE to queries and keys
        q, k = self.rope(q, k)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


############################################################
# Normalization Layers
############################################################

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Similar to LayerNorm but uses RMS instead of mean and variance.
    More efficient and used in modern architectures like LLaMA and GPT variants.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of any shape with last dim = dim
        Returns:
            Normalized tensor with same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x / rms


class MultiQueryAttentionRoPE(nn.Module):
    """Multi-Query Attention with Rotary Position Embedding.
    
    Efficient attention mechanism with:
    - Multiple query heads (default 8)
    - Single shared key head
    - Single shared value head
    - RoPE on queries and keys
    - Soft-clipping of attention logits for stability
    """
    def __init__(self, d_model, num_query_heads=8, qk_head_dim=128, 
                 v_head_dim=192, dropout=0.1, max_position=8192):
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        
        # Projections
        self.q_proj = nn.Linear(d_model, num_query_heads * qk_head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, v_head_dim, bias=False)
        
        # LayerNorm for Q, K, V after projection (AlphaGenome style)
        self.q_norm = nn.LayerNorm(qk_head_dim)
        self.k_norm = nn.LayerNorm(qk_head_dim)
        self.v_norm = nn.LayerNorm(v_head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(num_query_heads * v_head_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = qk_head_dim ** -0.5
        
        # RoPE
        self.rope = RotaryPositionEmbedding(qk_head_dim, max_seq_len=max_position)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, num_query_heads * qk_head_dim)
        k = self.k_proj(x)  # (batch, seq_len, qk_head_dim)
        v = self.v_proj(x)  # (batch, seq_len, v_head_dim)
        
        # Reshape Q for multiple heads
        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.qk_head_dim)
        # Apply LayerNorm to Q
        q = self.q_norm(q)
        q = q.permute(0, 2, 1, 3)  # (batch, num_query_heads, seq_len, qk_head_dim)
        
        # Reshape K for single head (shared across queries)
        k = k.reshape(batch_size, seq_len, 1, self.qk_head_dim)
        # Apply LayerNorm to K
        k = self.k_norm(k)
        k = k.permute(0, 2, 1, 3)  # (batch, 1, seq_len, qk_head_dim)
        
        # Reshape V for single head (shared across queries)
        v = v.reshape(batch_size, seq_len, 1, self.v_head_dim)
        # Apply LayerNorm to V
        v = self.v_norm(v)
        v = v.permute(0, 2, 1, 3)  # (batch, 1, seq_len, v_head_dim)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Compute attention scores
        # q: (batch, num_query_heads, seq_len, qk_head_dim)
        # k: (batch, 1, seq_len, qk_head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_query_heads, seq_len, seq_len)
        
        # Soft-clip attention logits (AlphaGenome style)
        attn = torch.tanh(attn / 5.0) * 5.0
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # v: (batch, 1, seq_len, v_head_dim)
        out = attn_weights @ v  # (batch, num_query_heads, seq_len, v_head_dim)
        
        # Reshape and project back
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.num_query_heads * self.v_head_dim)
        out = self.out_proj(out)
        
        return out


############################################################
# Feed Forward Networks
############################################################

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
    def forward(self, x):
        return self.linear2(self.dropout(activate(self.linear1(x), self.activation)))


############################################################
# Pooling Layers
############################################################

class GlobalPooling(nn.Module):
    """Global pooling layer to aggregate sequence information."""
    def __init__(self, d_model, pool_type='mean'):
        super().__init__()
        self.pool_type = pool_type
        if pool_type == 'attention':
            self.attention_weights = nn.Linear(d_model, 1)
            
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, d_model)
        """
        if self.pool_type == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                return x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            return x.mean(dim=1)
        
        elif self.pool_type == 'max':
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                return x_masked.max(dim=1)[0]
            return x.max(dim=1)[0]
        
        elif self.pool_type == 'attention':
            # Attention-based pooling
            attn_logits = self.attention_weights(x).squeeze(-1)  # (batch, seq_len)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            return (x * attn_weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")


############################################################
# Normalization Layers
############################################################

class LayerNorm(nn.Module):
    """Layer normalization wrapper."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    
    def forward(self, x):
        return self.norm(x)


class BatchNorm(nn.Module):
    """Batch normalization wrapper for 1D data."""
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, momentum=momentum)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # BatchNorm1d expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


############################################################
# Utility Layers
############################################################

class MaskHandler(nn.Module):
    """Utility for handling missing/masked positions in genomic data."""
    def __init__(self):
        super().__init__()
    
    def create_mask(self, x, missing_value=3):
        """Create boolean mask where True indicates valid positions.
        
        Args:
            x: (batch, seq_len) tensor
            missing_value: Value indicating missing data (default: 3 for SNPs)
        
        Returns:
            (batch, seq_len) boolean mask
        """
        return x != missing_value
    
    def forward(self, x, missing_value=3):
        return self.create_mask(x, missing_value)


############################################################
# Pooling Layers
############################################################

class SoftmaxPooling1d(nn.Module):
    """Softmax-weighted pooling for 1D sequences.
    
    Uses learnable attention weights computed via softmax to pool features.
    This is a differentiable alternative to max/average pooling.
    """
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        # Learnable weight for computing attention scores
        self.weight = nn.Parameter(torch.randn(1, 1, kernel_size))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
        
        Returns:
            Pooled tensor (batch, channels, seq_len // stride)
        """
        batch_size, channels, seq_len = x.shape
        
        # Calculate output length
        out_len = (seq_len - self.kernel_size) // self.stride + 1
        
        # Unfold to get sliding windows: (batch, channels, out_len, kernel_size)
        x_unfold = x.unfold(2, self.kernel_size, self.stride)
        
        # Compute attention scores using learnable weight
        # (batch, channels, out_len, kernel_size)
        scores = x_unfold * self.weight
        
        # Apply softmax over kernel dimension
        # (batch, channels, out_len, kernel_size)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum: (batch, channels, out_len)
        pooled = (x_unfold * attn_weights).sum(dim=-1)
        
        return pooled


############################################################
# Rotary Position Embedding (RoPE)
############################################################

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformer attention.
    
    Applies rotary embeddings to queries and keys to encode position information.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021)
    """
    def __init__(self, dim, max_seq_len=50000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequency for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device):
        """Update cached cos/sin values if sequence length changed."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim//2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
            self._cos_cached = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
            self._sin_cached = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k):
        """Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
        
        Returns:
            Rotated q and k tensors with same shape
        """
        seq_len = q.shape[2]
        self._update_cos_sin_cache(seq_len, q.device)
        
        # Apply rotation
        q_rot = (q * self._cos_cached) + (self._rotate_half(q) * self._sin_cached)
        k_rot = (k * self._cos_cached) + (self._rotate_half(k) * self._sin_cached)
        
        return q_rot, k_rot


############################################################
# Learnable Positional Embedding
############################################################

class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings for sequences.
    
    Uses trainable embeddings instead of fixed sinusoidal patterns.
    Suitable for sparse variant positions in genomic data.
    """
    def __init__(self, d_model, max_positions=50000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        
        # Learnable position embeddings
        self.pos_embedding = nn.Embedding(max_positions, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
    
    def forward(self, x):
        """Add learnable positional embeddings to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional embeddings added (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model, max_len=50000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)




class RegressionHead(nn.Module):
    """Task-specific head for regression tasks."""
    def __init__(self, d_model, num_targets, hidden_dim=None, dropout=0.1):
        super().__init__()
        layers = []
        
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_targets)
            ])
        else:
            layers.append(nn.Linear(d_model, num_targets))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, d_model)
        Returns:
            (batch, num_targets)
        """
        return self.network(x)


class ClassificationHead(nn.Module):
    """Task-specific head for binary/multi-class classification."""
    def __init__(self, d_model, num_tasks, num_classes=2, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        
        layers = []
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            final_dim = hidden_dim
        else:
            final_dim = d_model
        
        # Output layer for classification
        if num_classes == 2:
            # Binary classification: single logit per task
            layers.append(nn.Linear(final_dim, num_tasks))
        else:
            # Multi-class: num_classes logits per task
            layers.append(nn.Linear(final_dim, num_tasks * num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, d_model)
        Returns:
            (batch, num_tasks) for binary or (batch, num_tasks, num_classes) for multi-class
        """
        logits = self.network(x)
        if self.num_classes > 2:
            logits = logits.reshape(-1, self.num_tasks, self.num_classes)
        return logits


class MLPBlock(nn.Module):
    """Simple MLP block as alternative to transformer."""
    def __init__(self, d_model, d_ff, dropout=0.1, num_layers=3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(d_model, d_ff),
                nn.LayerNorm(d_ff),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            d_model = d_ff
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_ff)
        """
        return self.network(x)


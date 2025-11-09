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
    def __init__(self, d_model, pool_type='mean', pool_axis=1):
        super().__init__()
        self.pool_type = pool_type
        self.pool_axis = pool_axis
        if pool_type == 'attention':
            if pool_axis == 1:
                # Pool over sequence dimension: attention over seq_len
                self.attention_weights = nn.Linear(d_model, 1)
            else:
                # Pool over feature dimension: attention over d_model
                # Linear layer to compute attention logits for each feature dimension
                self.attention_weights = nn.Linear(1, 1)
            
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask (only used when pool_axis=1)
        Returns:
            If pool_axis=1: (batch, d_model) - pooled over sequence dimension
            If pool_axis=2: (batch, seq_len) - pooled over feature dimension
        """
        if self.pool_axis == 1:
            # Pool over sequence dimension (default)
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
                # Attention-based pooling over sequence dimension
                attn_logits = self.attention_weights(x).squeeze(-1)  # (batch, seq_len)
                if mask is not None:
                    attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
                attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
                return (x * attn_weights).sum(dim=1)
            
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        elif self.pool_axis == 2:
            # Pool over feature dimension
            if self.pool_type == 'mean':
                return x.mean(dim=2)  # (batch, seq_len)
            
            elif self.pool_type == 'max':
                return x.max(dim=2)[0]  # (batch, seq_len)
            
            elif self.pool_type == 'attention':
                # Attention-based pooling over feature dimension
                # x: (batch, seq_len, d_model)
                # For each sequence position, compute attention weights over feature dimensions
                # Reshape to apply linear layer to each feature value
                x_reshaped = x.unsqueeze(-1)  # (batch, seq_len, d_model, 1)
                attn_logits = self.attention_weights(x_reshaped).squeeze(-1)  # (batch, seq_len, d_model)
                attn_weights = F.softmax(attn_logits, dim=2)  # (batch, seq_len, d_model)
                # Weighted sum over feature dimension
                return (x * attn_weights).sum(dim=2)  # (batch, seq_len)
            
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        else:
            raise ValueError(f"pool_axis must be 1 (sequence) or 2 (feature), got {self.pool_axis}")


class MultiHeadPooling(nn.Module):
    """Multi-head pooling: parallel use of multiple pooling strategies.
    
    Combines mean, max, attention, and std pooling to capture different
    statistical properties of the sequence.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, pool_axis=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pool_axis = pool_axis
        
        # Attention pooling head
        if pool_axis == 1:
            # Pool over sequence dimension
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1)
            )
        else:
            # Pool over feature dimension
            self.attention_pool = nn.Linear(1, 1)
        
        # Fusion layer to combine all pooling heads
        if pool_axis == 1:
            # Output shape: (batch, d_model)
            self.fusion = nn.Linear(d_model * num_heads, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            # Output shape: (batch, seq_len)
            # Fusion layer: (batch, seq_len, num_heads) -> (batch, seq_len, 1)
            # This doesn't depend on seq_len, so we can create it in __init__
            self.fusion = nn.Linear(num_heads, 1)
            # LayerNorm requires fixed size, so we'll skip it for variable-length sequences
            self.layer_norm = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask (only used when pool_axis=1)
        Returns:
            If pool_axis=1: (batch, d_model) - pooled over sequence dimension
            If pool_axis=2: (batch, seq_len) - pooled over feature dimension
        """
        if self.pool_axis == 1:
            # Pool over sequence dimension (default)
            pooled = []
            
            # 1. Mean pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                mean_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                mean_pooled = x.mean(dim=1)
            pooled.append(mean_pooled)
            
            # 2. Max pooling
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                max_pooled = x_masked.max(dim=1)[0]
            else:
                max_pooled = x.max(dim=1)[0]
            pooled.append(max_pooled)
            
            # 3. Attention pooling
            attn_logits = self.attention_pool(x).squeeze(-1)  # (batch, seq_len)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            attn_pooled = (x * attn_weights).sum(dim=1)
            pooled.append(attn_pooled)
            
            # 4. Std pooling (capture variability)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                mean = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                std_pooled = torch.sqrt(((x - mean.unsqueeze(1)) ** 2 * mask_expanded).sum(dim=1) / 
                                       mask_expanded.sum(dim=1).clamp(min=1))
            else:
                std_pooled = x.std(dim=1)
            pooled.append(std_pooled)
            
            # Concatenate all pooling results
            concat = torch.cat(pooled, dim=-1)  # (batch, d_model * num_heads)
            
            # Fuse and normalize
            fused = self.fusion(concat)
            fused = self.layer_norm(fused)
            fused = self.dropout(fused)
            
            return fused
        
        elif self.pool_axis == 2:
            # Pool over feature dimension
            pooled = []
            
            # 1. Mean pooling over feature dimension
            mean_pooled = x.mean(dim=2)  # (batch, seq_len)
            pooled.append(mean_pooled)
            
            # 2. Max pooling over feature dimension
            max_pooled = x.max(dim=2)[0]  # (batch, seq_len)
            pooled.append(max_pooled)
            
            # 3. Attention pooling over feature dimension
            x_reshaped = x.unsqueeze(-1)  # (batch, seq_len, d_model, 1)
            attn_logits = self.attention_pool(x_reshaped).squeeze(-1)  # (batch, seq_len, d_model)
            attn_weights = F.softmax(attn_logits, dim=2)  # (batch, seq_len, d_model)
            attn_pooled = (x * attn_weights).sum(dim=2)  # (batch, seq_len)
            pooled.append(attn_pooled)
            
            # 4. Std pooling over feature dimension
            std_pooled = x.std(dim=2)  # (batch, seq_len)
            pooled.append(std_pooled)
            
            # Stack all pooling results
            concat = torch.stack(pooled, dim=-1)  # (batch, seq_len, num_heads)
            
            # Fuse: (batch, seq_len, num_heads) -> (batch, seq_len, 1) -> (batch, seq_len)
            fused = self.fusion(concat).squeeze(-1)  # (batch, seq_len)
            
            # Apply dropout (skip LayerNorm since seq_len can vary)
            fused = self.dropout(fused)
            
            return fused
        
        else:
            raise ValueError(f"pool_axis must be 1 (sequence) or 2 (feature), got {self.pool_axis}")


class LearnableQueryPooling(nn.Module):
    """Learnable query pooling using learnable query vectors (Set2Set style).
    
    Uses learnable query vectors to attend to the sequence or features, allowing
    the model to learn task-specific aggregation patterns.
    """
    def __init__(self, d_model, num_queries=1, num_layers=2, dropout=0.1, pool_axis=1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.pool_axis = pool_axis
        
        if pool_axis == 1:
            # Pool over sequence dimension
            # Learnable query vectors for attending to sequence positions
            self.query = nn.Parameter(torch.randn(num_queries, d_model))
            
            # Cross-attention mechanism
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            
            # Optional: LSTM for iterative refinement (Set2Set style)
            self.use_lstm = num_layers > 0
            if self.use_lstm:
                self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
            
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            # Pool over feature dimension
            # For feature dimension pooling, we use a learnable weight vector
            # to compute attention scores for each feature dimension
            # This is more memory-efficient than using a Sequential network
            self.feature_weight = nn.Parameter(torch.randn(d_model))
            
            # Optional: LSTM for iterative refinement (applied per sequence position)
            self.use_lstm = num_layers > 0
            if self.use_lstm:
                self.lstm = nn.LSTM(1, 1, num_layers, batch_first=True)
            
            # Query parameter not used for feature dimension pooling
            self.query = None
            # LayerNorm not used for variable-length sequences
            self.layer_norm = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask (only used when pool_axis=1)
        Returns:
            If pool_axis=1: (batch, d_model) - pooled over sequence dimension
            If pool_axis=2: (batch, seq_len) - pooled over feature dimension
        """
        if self.pool_axis == 1:
            # Pool over sequence dimension (default)
            batch_size = x.size(0)
            
            # Expand learnable query to batch size
            query = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, d_model)
            
            # Cross-attention: query attends to sequence
            attn_out, _ = self.cross_attention(
                query, x, x,
                key_padding_mask=~mask if mask is not None else None
            )  # (batch, num_queries, d_model)
            
            # Optional LSTM refinement
            if self.use_lstm:
                attn_out, _ = self.lstm(attn_out)
            
            # Normalize and dropout
            output = self.layer_norm(attn_out)
            output = self.dropout(output)
            
            # If single query, squeeze; otherwise return all queries
            if self.num_queries == 1:
                return output.squeeze(1)  # (batch, d_model)
            else:
                # Average over queries if multiple
                return output.mean(dim=1)  # (batch, d_model)
        
        elif self.pool_axis == 2:
            # Pool over feature dimension
            # x: (batch, seq_len, d_model)
            
            # Compute attention logits for each feature dimension
            # Use learnable weight vector to compute attention scores
            # Memory-efficient: element-wise multiplication (no large intermediate tensors)
            attn_logits = x * self.feature_weight.unsqueeze(0).unsqueeze(0)  # (batch, seq_len, d_model)
            
            # Apply softmax to get attention weights over feature dimension
            attn_weights = F.softmax(attn_logits, dim=2)  # (batch, seq_len, d_model)
            
            # Weighted sum over feature dimension
            pooled = (x * attn_weights).sum(dim=2)  # (batch, seq_len)
            
            # Optional LSTM refinement (applied per sequence position)
            if self.use_lstm:
                # Reshape for LSTM: (batch, seq_len) -> (batch, seq_len, 1)
                pooled = pooled.unsqueeze(-1)  # (batch, seq_len, 1)
                pooled, _ = self.lstm(pooled)  # (batch, seq_len, 1)
                pooled = pooled.squeeze(-1)  # (batch, seq_len)
            
            # Apply dropout
            output = self.dropout(pooled)
            
            return output
        
        else:
            raise ValueError(f"pool_axis must be 1 (sequence) or 2 (feature), got {self.pool_axis}")


class HierarchicalPooling(nn.Module):
    """Hierarchical pooling: local pooling followed by global pooling.
    
    First performs local pooling over sliding windows, then applies
    global attention pooling to capture multi-scale information.
    """
    def __init__(self, d_model, window_size=64, stride=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.stride = stride
        
        # Local pooling (adaptive average pooling)
        self.local_pool = nn.AdaptiveAvgPool1d(1)
        
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Fusion layer to combine local and global features
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Local pooling: sliding window approach
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        local_features = []
        
        # Apply sliding window pooling
        for i in range(0, seq_len, self.stride):
            end_idx = min(i + self.window_size, seq_len)
            if end_idx - i < self.window_size // 2:  # Skip too small windows
                break
            window = x_t[:, :, i:end_idx]
            pooled = self.local_pool(window).squeeze(-1)  # (batch, d_model)
            local_features.append(pooled)
        
        if len(local_features) == 0:
            # Fallback to global pooling if no windows
            local_features = [x.mean(dim=1)]
        
        local_features = torch.stack(local_features, dim=1)  # (batch, num_windows, d_model)
        
        # Global attention pooling
        # Use mean of local features as query
        query = local_features.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        global_feat, _ = self.global_attention(
            query, x, x,
            key_padding_mask=~mask if mask is not None else None
        )  # (batch, 1, d_model)
        global_feat = global_feat.squeeze(1)  # (batch, d_model)
        
        # Aggregate local features
        local_agg = local_features.mean(dim=1)  # (batch, d_model)
        
        # Fuse local and global features
        concat = torch.cat([local_agg, global_feat], dim=-1)  # (batch, d_model * 2)
        fused = self.fusion(concat)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused


class TransformerPooling(nn.Module):
    """Transformer-based pooling using special pooling token (BERT CLS style).
    
    Uses a learnable pooling token that attends to all sequence positions
    or feature dimensions through transformer layers, similar to BERT's CLS token.
    """
    def __init__(self, d_model, num_layers=2, num_heads=8, dim_feedforward=None, dropout=0.1, pool_axis=1):
        super().__init__()
        self.d_model = d_model
        self.pool_axis = pool_axis
        
        if pool_axis == 1:
            # Pool over sequence dimension
            # Learnable pooling token
            self.pool_token = nn.Parameter(torch.randn(1, 1, d_model))
            
            # Transformer encoder
            if dim_feedforward is None:
                dim_feedforward = d_model * 4
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            # Pool over feature dimension
            # For each sequence position, use a CLS token to aggregate features
            # Learnable pooling token for feature dimension
            self.pool_token = nn.Parameter(torch.randn(1, 1, 1))
            
            # Smaller transformer for feature dimension
            # Each "position" in this transformer corresponds to a feature dimension
            # We need embedding_dim that works for the transformer
            feature_embed_dim = min(64, d_model)  # Use smaller embedding for efficiency
            if dim_feedforward is None:
                dim_feedforward = feature_embed_dim * 2
            
            # Project each feature value to embedding space
            self.feature_projection = nn.Linear(1, feature_embed_dim)
            
            # Transformer encoder for features
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_embed_dim,
                nhead=min(4, feature_embed_dim // 16),  # Adjust heads based on embedding dim
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Project CLS token embedding back to scalar
            self.output_projection = nn.Linear(feature_embed_dim, 1)
            
            # No LayerNorm for variable-length sequences
            self.layer_norm = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask (only used when pool_axis=1)
        Returns:
            If pool_axis=1: (batch, d_model) - pooled over sequence dimension
            If pool_axis=2: (batch, seq_len) - pooled over feature dimension
        """
        if self.pool_axis == 1:
            # Pool over sequence dimension (default)
            batch_size = x.size(0)
            
            # Add pooling token at the beginning
            pool_token = self.pool_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            x_with_pool = torch.cat([pool_token, x], dim=1)  # (batch, seq_len+1, d_model)
            
            # Extend mask to include pooling token (always valid)
            if mask is not None:
                pool_mask = torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool)
                extended_mask = torch.cat([pool_mask, mask], dim=1)
                # Convert to key_padding_mask format (True = ignore)
                key_padding_mask = ~extended_mask
            else:
                key_padding_mask = None
            
            # Transformer encoding
            encoded = self.transformer(x_with_pool, src_key_padding_mask=key_padding_mask)
            
            # Extract pooling token representation
            pool_output = encoded[:, 0, :]  # (batch, d_model)
            
            # Normalize and dropout
            output = self.layer_norm(pool_output)
            output = self.dropout(output)
            
            return output
        
        elif self.pool_axis == 2:
            # Pool over feature dimension
            # x: (batch, seq_len, d_model)
            batch_size, seq_len, d_model = x.size()
            
            # Reshape to process each position independently
            # (batch, seq_len, d_model) -> (batch * seq_len, d_model)
            x_flat = x.reshape(-1, d_model)  # (batch * seq_len, d_model)
            
            # Project each feature value to embedding space
            # (batch * seq_len, d_model) -> (batch * seq_len, d_model, 1) -> (batch * seq_len, d_model, feature_embed_dim)
            x_reshaped = x_flat.unsqueeze(-1)  # (batch * seq_len, d_model, 1)
            x_embedded = self.feature_projection(x_reshaped)  # (batch * seq_len, d_model, feature_embed_dim)
            
            # Add CLS token for each position
            pool_token = self.pool_token.expand(batch_size * seq_len, -1, -1)  # (batch * seq_len, 1, 1)
            pool_token_embedded = self.feature_projection(pool_token)  # (batch * seq_len, 1, feature_embed_dim)
            
            # Concatenate CLS token with feature embeddings
            x_with_pool = torch.cat([pool_token_embedded, x_embedded], dim=1)  # (batch * seq_len, d_model+1, feature_embed_dim)
            
            # Apply transformer to aggregate features
            encoded = self.transformer(x_with_pool)  # (batch * seq_len, d_model+1, feature_embed_dim)
            
            # Extract CLS token representation
            cls_output = encoded[:, 0, :]  # (batch * seq_len, feature_embed_dim)
            
            # Project back to scalar
            pooled = self.output_projection(cls_output).squeeze(-1)  # (batch * seq_len,)
            
            # Reshape back to (batch, seq_len)
            output = pooled.reshape(batch_size, seq_len)
            
            # Apply dropout
            output = self.dropout(output)
            
            return output
        
        else:
            raise ValueError(f"pool_axis must be 1 (sequence) or 2 (feature), got {self.pool_axis}")


class SampleStructureEncoder(nn.Module):
    """
    Learnable sample structure encoding that captures kinship/population structure.
    
    Instead of positional encoding (which didn't help), this learns sample-level
    embeddings based on genotype patterns and injects them into the feature space.
    
    The module:
    1. Extracts sample-level statistics from genotype patterns (using existing pooling)
    2. Learns sample structure embeddings
    3. Uses cross-sample attention to model kinship relationships
    4. Injects structure information into sequence features
    """
    def __init__(self, d_model, structure_dim=64, use_cross_sample_attention=True, 
                 pooling_type='attention', pool_axis=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.structure_dim = structure_dim
        self.use_cross_sample_attention = use_cross_sample_attention
        self.pooling_type = pooling_type
        self.pool_axis = pool_axis
        
        # Sample-level pooling: use existing GlobalPooling or MultiHeadPooling
        if pooling_type == 'attention':
            # Use existing GlobalPooling with attention
            self.sample_pooling = GlobalPooling(d_model, pool_type='attention', pool_axis=pool_axis)
        elif pooling_type == 'multi_head':
            # Use existing MultiHeadPooling (combines mean, max, attention, std)
            self.sample_pooling = MultiHeadPooling(d_model, num_heads=4, dropout=dropout, pool_axis=pool_axis)
        elif pooling_type == 'mean_max':
            # Combine mean and max pooling manually
            self.sample_pooling = None  # Will handle in forward
        else:
            # Default: simple mean pooling using GlobalPooling
            self.sample_pooling = GlobalPooling(d_model, pool_type='mean', pool_axis=pool_axis)
        
        # Statistics extractor: compute sample-level features
        # If pool_axis=1: Input is (batch, d_model) -> Output: (batch, structure_dim)
        # If pool_axis=2: Input is (batch, seq_len) -> Output: (batch, structure_dim)
        # We'll create it dynamically based on the actual input shape
        self.stat_extractor = None  # Will be created on first forward pass
        self.structure_dim = structure_dim
        self.dropout_rate = dropout
        
        # Project structure embedding to match d_model
        self.structure_projection = nn.Linear(structure_dim, d_model)
        
        # Cross-sample attention: learn kinship relationships
        if use_cross_sample_attention:
            self.cross_sample_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) - sequence features
            mask: (batch, seq_len) - sequence mask (optional)
        Returns:
            x_enhanced: (batch, seq_len, d_model) - features with structure info
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute sample-level representation from sequence
        if self.sample_pooling is not None:
            # Use existing pooling module (GlobalPooling or MultiHeadPooling)
            sample_repr = self.sample_pooling(x, mask=mask)  # (batch, d_model) or (batch, seq_len)
        elif self.pooling_type == 'mean_max':
            # Combine mean and max pooling
            if mask is not None:
                x_masked = x * mask.unsqueeze(-1)
                mean_pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                # Max pooling with mask
                x_masked_for_max = x.clone()
                x_masked_for_max[~mask.unsqueeze(-1)] = float('-inf')
                max_pooled = x_masked_for_max.max(dim=1)[0]
            else:
                mean_pooled = x.mean(dim=1)
                max_pooled = x.max(dim=1)[0]
            # Concatenate and project
            sample_repr = (mean_pooled + max_pooled) / 2  # Average of mean and max
        else:
            # Fallback to mean pooling
            if mask is not None:
                x_masked = x * mask.unsqueeze(-1)
                sample_repr = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                sample_repr = x.mean(dim=1)
        
        # Extract sample structure features
        # Create stat_extractor dynamically based on pool_axis
        if self.stat_extractor is None:
            # Determine input dimension based on pool_axis
            if self.pool_axis == 2:
                input_dim = seq_len  # pool_axis=2: input is (batch, seq_len)
            else:
                input_dim = d_model  # pool_axis=1: input is (batch, d_model)
            
            self.stat_extractor = nn.Sequential(
                nn.Linear(input_dim, self.structure_dim),
                nn.LayerNorm(self.structure_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.structure_dim, self.structure_dim),
                nn.LayerNorm(self.structure_dim)
            ).to(sample_repr.device)
        
        structure_features = self.stat_extractor(sample_repr)  # (batch, structure_dim)
        
        # Project to model dimension
        structure_emb = self.structure_projection(structure_features)  # (batch, d_model)
        
        # Option 1: Additive injection (like positional encoding)
        x_enhanced = x + structure_emb.unsqueeze(1)  # (batch, seq_len, d_model)
        
        # Option 2: Cross-sample attention (learn kinship relationships)
        if self.use_cross_sample_attention:
            # Use structure embeddings for cross-sample attention
            structure_emb_expanded = structure_emb.unsqueeze(1)  # (batch, 1, d_model)
            
            # Cross-sample attention: each sample attends to all samples' structure
            attended, _ = self.cross_sample_attention(
                query=x_enhanced,
                key=structure_emb_expanded,
                value=structure_emb_expanded
            )
            
            # Residual connection with gating
            x_enhanced = x_enhanced + 0.3 * self.attention_norm(attended)
        
        # Final normalization and dropout
        x_enhanced = self.layer_norm(x_enhanced)
        x_enhanced = self.dropout(x_enhanced)
        
        return x_enhanced


class SampleCLSToken(nn.Module):
    """
    Sample-level CLS token that learns population structure.
    
    Similar to sequence-level CLS token (like BERT), but operates at sample level.
    Each sample gets a learnable CLS token that aggregates information across
    the sequence and interacts with other samples' CLS tokens to learn kinship.
    
    This allows the model to:
    1. Learn sample-level representations
    2. Model relationships between samples
    3. Capture population structure automatically
    """
    def __init__(self, d_model, num_layers=2, num_heads=8, dim_feedforward=None, 
                 pooling_type='attention', pool_axis=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pooling_type = pooling_type
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        
        # Sample-level pooling: use existing GlobalPooling or MultiHeadPooling
        # pool_axis=1: pool over sequence -> (batch, d_model) directly
        # pool_axis=2: pool over features -> (batch, seq_len), then map to (batch, d_model) with Linear
        if pooling_type == 'attention':
            # Use existing GlobalPooling with attention
            self.sample_pooling = GlobalPooling(d_model, pool_type='attention', pool_axis=pool_axis)
        elif pooling_type == 'multi_head':
            # Use existing MultiHeadPooling (combines mean, max, attention, std)
            self.sample_pooling = MultiHeadPooling(d_model, num_heads=4, dropout=dropout, pool_axis=pool_axis)
        elif pooling_type == 'mean_max':
            # Combine mean and max pooling manually
            self.sample_pooling = None  # Will handle in forward
        else:
            # Default: simple mean pooling using GlobalPooling
            self.sample_pooling = GlobalPooling(d_model, pool_type='mean', pool_axis=pool_axis)
        
        # Projection to create sample CLS token from sequence summary
        # Will be created dynamically based on input shape (d_model or seq_len)
        self.sample_cls_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Transformer to aggregate sample structure and enable cross-sample interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.sample_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to inject CLS information back into sequence
        self.cls_injection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) - sequence features
            mask: (batch, seq_len) - sequence mask (optional)
        Returns:
            x_enhanced: (batch, seq_len, d_model) - features enhanced with sample CLS
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create sample CLS token from sequence summary
        if self.sample_pooling is not None:
            # Use existing pooling module (GlobalPooling or MultiHeadPooling)
            sample_summary = self.sample_pooling(x, mask=mask)  # (batch, d_model) or (batch, seq_len)
        elif self.pooling_type == 'mean_max':
            # Combine mean and max pooling
            if mask is not None:
                x_masked = x * mask.unsqueeze(-1)
                mean_pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                # Max pooling with mask
                x_masked_for_max = x.clone()
                x_masked_for_max[~mask.unsqueeze(-1)] = float('-inf')
                max_pooled = x_masked_for_max.max(dim=1)[0]
            else:
                mean_pooled = x.mean(dim=1)
                max_pooled = x.max(dim=1)[0]
            # Average of mean and max
            sample_summary = (mean_pooled + max_pooled) / 2
        else:
            # Fallback to mean pooling
            if mask is not None:
                x_masked = x * mask.unsqueeze(-1)
                sample_summary = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                sample_summary = x.mean(dim=1)
        
        # Project to create sample CLS token
        # Create sample_cls_proj dynamically based on pool_axis
        if self.sample_cls_proj is None:
            # Determine input dimension based on pool_axis
            if self.pool_axis == 2:
                input_dim = seq_len  # pool_axis=2: input is (batch, seq_len)
            else:
                input_dim = d_model  # pool_axis=1: input is (batch, d_model)
            
            self.sample_cls_proj = nn.Sequential(
                nn.Linear(input_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout_rate_cls),
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model)
            ).to(sample_summary.device)
        
        sample_cls = self.sample_cls_proj(sample_summary)  # (batch, d_model)
        
        # Prepare input for transformer: [sample_cls, sequence]
        # This allows the CLS token to attend to the sequence and vice versa
        sample_cls_expanded = sample_cls.unsqueeze(1)  # (batch, 1, d_model)
        x_with_cls = torch.cat([sample_cls_expanded, x], dim=1)  # (batch, seq_len+1, d_model)
        
        # Extend mask to include CLS token (always valid)
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool)
            extended_mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = ~extended_mask
        else:
            key_padding_mask = None
        
        # Apply transformer (enables cross-sample interaction through CLS tokens)
        encoded = self.sample_transformer(x_with_cls, src_key_padding_mask=key_padding_mask)
        
        # Extract enhanced CLS and sequence
        enhanced_cls = encoded[:, 0, :]  # (batch, d_model)
        enhanced_seq = encoded[:, 1:, :]  # (batch, seq_len, d_model)
        
        # Inject CLS information into sequence
        cls_injected = self.cls_injection(enhanced_cls)  # (batch, d_model)
        x_enhanced = enhanced_seq + cls_injected.unsqueeze(1)  # (batch, seq_len, d_model)
        
        # Final normalization
        x_enhanced = self.layer_norm(x_enhanced)
        x_enhanced = self.dropout(x_enhanced)
        
        return x_enhanced


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


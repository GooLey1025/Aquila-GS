from __future__ import annotations
"""
不同类型的注意力机制实现 - 使用高效的库
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, Tuple
import math
import warnings
import logging  # Add import

# 尝试导入 flash attention
try:
    # 导入 flash_attn_func 和 varlen 版本
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    # 定义占位符以便类型检查通过
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    warnings.warn("未安装flash-attn。FlashAttention将回退到标准注意力。安装方法: pip install flash-attn")


class BaseCrossAttention(nn.Module):
    """注意力机制基类"""
    def __init__(self,
                 encoder_dim: int,
                 num_heads: int,
                 head_dims: int,
                 dropout_rate: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.dropout_rate = dropout_rate
        # 存储温度值，确保不为零
        self.temperature = max(temperature, 1e-6)
        self.total_head_dim = num_heads * head_dims
        # 计算基础的 softmax scale (1 / sqrt(d_head))
        self.scale = self.head_dims ** -0.5

        # 注意：这里可能需要调整投影矩阵的输出尺寸，使其与注意力头数和每个头维度兼容
        self.q_proj = nn.Linear(encoder_dim, self.total_head_dim)
        self.k_proj = nn.Linear(encoder_dim, self.total_head_dim)
        self.v_proj = nn.Linear(encoder_dim, self.total_head_dim)
        # 添加输出投影层
        self.out_proj = nn.Linear(self.total_head_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播函数

        参数:
            query: [batch_size, seq_len_q, encoder_dim]
            key: [batch_size, seq_len_kv, encoder_dim]
            value: [batch_size, seq_len_kv, encoder_dim]
            attn_mask: 注意力掩码 [batch_size, seq_len_kv] (True表示保留, False表示屏蔽)

        返回:
            output: [batch_size, seq_len_q, encoder_dim]
            attention_weights: 注意力权重 (Optional)
        """
        raise NotImplementedError("子类必须实现forward方法")


class CrossAttention(BaseCrossAttention):
    """优化的交叉注意力实现"""
    def __init__(self,
                 encoder_dim: int,
                 num_heads: int,
                 head_dims: int,
                 dropout_rate: float = 0.1,
                 temperature: float = 1.0):
        """初始化标准交叉注意力模块"""
        super().__init__(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            head_dims=head_dims,
            dropout_rate=dropout_rate,
            temperature=temperature
        )

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dims).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dims).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dims).transpose(1, 2)

        final_scale = self.scale / self.temperature

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * final_scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # 假设 mask 中 True 表示保留, False 表示屏蔽
                mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
                mask = mask_bool[:, None, None, :] # [B, 1, 1, Nk]
            else:
                 raise ValueError(f"CrossAttention expects 2D attn_mask [B, Nk], got dim={attn_mask.dim()}")
            attn_weights = attn_weights.masked_fill(~mask, float('-inf')) # 使用 ~mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)

        output = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.total_head_dim
        )
        output = self.out_proj(output)

        avg_attn_probs = attn_probs.mean(dim=1) if attn_probs is not None else None

        return output, avg_attn_probs


class FlashCrossAttention(BaseCrossAttention):
    """优化的交叉注意力实现 - 使用FlashAttention"""
    def __init__(self,
                 encoder_dim: int,
                 num_heads: int,
                 head_dims: int,
                 dropout_rate: float = 0.1,
                 temperature: float = 1.0):
        super().__init__(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            head_dims=head_dims,
            dropout_rate=dropout_rate,
            temperature=temperature
        )
        if not HAS_FLASH_ATTN:
            warnings.warn("未安装flash-attn，FlashCrossAttention将回退到标准交叉注意力实现")

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        final_softmax_scale = self.scale / self.temperature

        if HAS_FLASH_ATTN:
            q_flash = q.view(batch_size, seq_len_q, self.num_heads, self.head_dims)
            k_flash = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dims)
            v_flash = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dims)

            if attn_mask is not None:
                if attn_mask.dim() == 2 and attn_mask.shape[0] == batch_size and attn_mask.shape[1] == seq_len_kv:
                    try:
                        mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask

                        if seq_len_q > seq_len_kv:
                             warnings.warn(f"查询序列长度 ({seq_len_q}) 大于掩码/键序列长度 ({seq_len_kv})，FlashAttention变长模式将只考虑掩码覆盖的键。")
                        mask_q = mask_bool[:, :seq_len_q]

                        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q_ = unpad_input(q_flash, mask_q)
                        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k_ = unpad_input(k_flash, mask_bool)
                        v_unpad, _, _, _ = unpad_input(v_flash, mask_bool)

                        if q_unpad is None or k_unpad is None or v_unpad is None:
                             warnings.warn("FlashCrossAttention unpad_input 返回 None (可能由于全零掩码)，回退到标准实现。")
                             raise RuntimeError("Unpad resulted in None tensor")

                        dropout_p = self.dropout_rate if self.training else 0.0

                        output_unpad = flash_attn_varlen_func(
                            q_unpad, k_unpad, v_unpad,
                            cu_seqlens_q, cu_seqlens_k,
                            max_seqlen_q_, max_seqlen_k_,
                            dropout_p=dropout_p,
                            softmax_scale=final_softmax_scale,
                            causal=False,
                            return_attn_probs=False
                        )

                        output = pad_input(output_unpad, indices_q, batch_size, seq_len_q)
                        output = output.reshape(batch_size, seq_len_q, self.total_head_dim)
                        output = self.out_proj(output)

                        return output, None

                    except Exception as e:
                        if not self.training:
                            logging.warning(f"Flash Attention (varlen) fallback during eval: {e}")
                        warnings.warn(f"Flash Attention变长版本失败，回退到标准实现: {e}")
                        return self._fallback_attention(q, k, v, attn_mask)
                else:
                    warnings.warn(f"FlashCrossAttention接收到不支持的掩码形状 (dim={attn_mask.dim()}, shape={attn_mask.shape}) 或与输入不匹配，期望 [B={batch_size}, Nk={seq_len_kv}]。回退到标准实现。")
                    return self._fallback_attention(q, k, v, attn_mask)
            else:
                try:
                    dropout_p = self.dropout_rate if self.training else 0.0
                    output = flash_attn_func(
                        q_flash, k_flash, v_flash,
                        dropout_p=dropout_p,
                        softmax_scale=final_softmax_scale,
                        causal=False
                    )
                    output = output.reshape(batch_size, seq_len_q, self.total_head_dim)
                    output = self.out_proj(output)
                    return output, None

                except Exception as e:
                    if not self.training:
                        logging.warning(f"Flash Attention (standard) fallback during eval: {e}")
                    warnings.warn(f"标准Flash Attention失败，回退到标准实现: {e}")
                    return self._fallback_attention(q, k, v, attn_mask)

        if not self.training:
            logging.warning(f"Flash Attention not available/failed, using fallback during eval.")
        return self._fallback_attention(q, k, v, attn_mask)

    def _fallback_attention(self, q, k, v, attn_mask):
        """标准注意力的回退实现"""
        if not self.training:
            logging.info(f"Executing _fallback_attention during eval.")
        batch_size, seq_len_q = q.shape[:2]
        _, seq_len_kv = k.shape[:2]

        if q.dim() == 3:
            q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dims).transpose(1, 2)
            k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dims).transpose(1, 2)
            v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dims).transpose(1, 2)
        elif q.dim() == 4:
             q = q.transpose(1, 2)
             k = k.transpose(1, 2)
             v = v.transpose(1, 2)
        else:
             raise ValueError(f"Unsupported input dimension for fallback: {q.dim()}")

        final_scale = self.scale / self.temperature

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * final_scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
                mask = mask_bool[:, None, None, :] # [B, 1, 1, Nk]
            else:
                 warnings.warn(f"Fallback attention received unexpected mask dimension: {attn_mask.dim()}. Trying to adapt.")
                 if attn_mask.dim() == 3:
                     mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
                     mask = mask_bool[:, None, :, :]
                 elif attn_mask.dim() == 4:
                     mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
                     mask = mask_bool
                 else:
                     raise ValueError(f"Unsupported attn_mask dimension in fallback: {attn_mask.dim()}")
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)

        output = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.total_head_dim
        )
        output = self.out_proj(output)

        avg_attn_probs = attn_probs.mean(dim=1) if attn_probs is not None else None

        return output, avg_attn_probs


class ProbabilisticCrossAttention(BaseCrossAttention):
    """概率稀疏交叉注意力实现"""
    def __init__(self,
                 encoder_dim: int,
                 num_heads: int,
                 head_dims: int,
                 dropout_rate: float = 0.1,
                 temperature: float = 1.0,
                 sparsity: float = 0.9):
        super().__init__(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            head_dims=head_dims,
            dropout_rate=dropout_rate,
            temperature=temperature
        )
        self.sparsity = sparsity

        # 添加额外的FFN层用于生成稀疏掩码
        self.sparsity_ffn = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dims).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dims).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dims).transpose(1, 2)

        final_scale = self.scale / self.temperature

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * final_scale

        combined_mask_inf = torch.zeros_like(attn_weights)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask_bool = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
                pad_mask = mask_bool[:, None, None, :] # [B, 1, 1, Nk]
                combined_mask_inf.masked_fill_(~pad_mask, float('-inf'))
            else:
                 raise ValueError(f"ProbabilisticCrossAttention expects 2D attn_mask [B, Nk], got dim={attn_mask.dim()}")

        sparsity_logits = self.sparsity_ffn(key)
        sparsity_scores = torch.sigmoid(sparsity_logits).squeeze(-1)

        k_to_keep = max(1, int(seq_len_kv * (1 - self.sparsity)))
        _, top_indices = torch.topk(sparsity_scores, k=k_to_keep, dim=-1)

        sparse_mask_bool = torch.zeros((batch_size, seq_len_kv), device=q.device, dtype=torch.bool)
        sparse_mask_bool.scatter_(1, top_indices, True)
        sparse_mask = sparse_mask_bool[:, None, None, :]

        combined_mask_inf.masked_fill_(~sparse_mask, float('-inf'))

        attn_weights = attn_weights + combined_mask_inf

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)

        output = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.total_head_dim
        )
        output = self.out_proj(output)

        avg_attn_probs = attn_probs.mean(dim=1) if attn_probs is not None else None

        return output, avg_attn_probs


def create_crossattention_from_config(config: Dict[str, Any], encoder_dim: int) -> BaseCrossAttention:
    """
    根据配置创建对应的注意力机制实例
    """
    attention_type = config.get("type", "standard").lower()
    num_heads = config.get("num_heads", 8)
    head_dims = config.get("head_dims", 64)
    dropout_rate = config.get("dropout_rate", 0.1)
    temperature = config.get("temperature", 1.0)

    total_head_dim = num_heads * head_dims

    if attention_type == "flash_attention":
        return FlashCrossAttention(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            head_dims=head_dims,
            dropout_rate=dropout_rate,
            temperature=temperature
        )
    elif attention_type == "probabilistic":
        sparsity = config.get("sparsity", 0.9)
        return ProbabilisticCrossAttention(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            head_dims=head_dims,
            dropout_rate=dropout_rate,
            temperature=temperature,
            sparsity=sparsity
        )
    else:
        return CrossAttention(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            head_dims=head_dims,
            dropout_rate=dropout_rate,
            temperature=temperature
        )

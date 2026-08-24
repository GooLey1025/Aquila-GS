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

# 尝试导入 xformers
try:
    import xformers.ops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    warnings.warn("未安装xformers。高效注意力变体将回退到标准实现。安装方法: pip install xformers")

# 基类 - 保持API兼容性
class BaseAttention(nn.Module):
    """注意力机制基类"""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError(f"维度 {dim} 无法被头数 {num_heads} 整除")
        self.dropout = nn.Dropout(dropout_rate)
        # 基础缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        """子类必须实现此方法"""
        raise NotImplementedError

    def get_proj_queries_keys(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """子类应实现此方法以支持预计算投影"""
        raise NotImplementedError("get_proj_queries_keys 未在此类中实现")

class StandardAttention(BaseAttention):
    """标准多头注意力实现 - 使用PyTorch原生MultiheadAttention优化"""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1,
                 temperature: float = 1.0):  # <--- 添加 temperature 参数
        super().__init__(dim, num_heads, dropout_rate)
        # 使用 PyTorch 的 MultiheadAttention 层
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout_rate, batch_first=True)
        # 存储温度值
        self.temperature = max(temperature, 1e-6)  # <--- 存储 temperature

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        """
        使用 PyTorch MHA 或自定义实现进行前向传播

        Args:
            q, k, v: 输入张量 [B, L, D]
            mask: 注意力掩码。可以是:
                  - [B, Lk] (key_padding_mask): True 表示屏蔽
                  - [Lq, Lk] (attn_mask): True 表示屏蔽
                  - [B, Nq, Lk] (attn_mask): True 表示屏蔽
            proj_q, proj_k: 预计算的投影 (未使用，因为 MHA 内部处理)

        Returns:
            输出张量和注意力权重 (如果可用)
        """
        B, Nq, D = q.shape
        Nk = k.shape[1]

        # 处理掩码以适应 MHA 的期望格式
        attn_mask_mha = None
        key_padding_mask_mha = None

        if mask is not None:
            if mask.dtype != torch.bool:
                 warnings.warn(f"StandardAttention 接收到非布尔掩码 (dtype={mask.dtype})，将尝试转换为布尔型。期望 True 表示屏蔽。")
                 mask = mask < 0.5  # 假设 0 表示保留, 1 表示屏蔽 -> < 0.5 为 True (屏蔽)

            if mask.dim() == 2 and mask.shape[0] == B and mask.shape[1] == Nk:
                # key_padding_mask: [B, Lk], True 表示屏蔽
                key_padding_mask_mha = mask
            elif mask.dim() == 2 and mask.shape[0] == Nq and mask.shape[1] == Nk:
                # attn_mask: [Lq, Lk], True 表示屏蔽
                attn_mask_mha = mask
            elif mask.dim() == 3 and mask.shape[0] == B and mask.shape[1] == Nq and mask.shape[2] == Nk:
                 # attn_mask: [B, Nq, Lk], 需要转换为 [B*H, Nq, Lk]
                 # MHA 不直接支持批处理的 3D attn_mask，需要手动实现或回退
                 warnings.warn("StandardAttention (MHA) 不直接支持 [B, Nq, Lk] 掩码，将回退到手动计算。")
                 return self._forward_with_proj(q, k, v, mask=mask, proj_q=proj_q, proj_k=proj_k)
            else:
                 warnings.warn(f"StandardAttention 接收到不支持的掩码形状: {mask.shape}，将忽略掩码。")

        # 如果 temperature 不为 1.0，MHA 不支持，需要回退
        if self.temperature != 1.0:
            warnings.warn("StandardAttention (MHA) 不支持 temperature != 1.0，将回退到手动计算。")
            return self._forward_with_proj(q, k, v, mask=mask, proj_q=proj_q, proj_k=proj_k)

        # 使用 MHA 计算 (仅当 temperature=1.0 且掩码兼容时)
        # MHA 期望 need_weights=True 才返回权重
        try:
            # MHA 内部处理 Q, K, V 投影
            output, attn_weights = self.mha(q, k, v,
                                            key_padding_mask=key_padding_mask_mha,
                                            attn_mask=attn_mask_mha,
                                            need_weights=True,  # 请求权重
                                            average_attn_weights=False)  # 获取每个头的权重 [B, H, Nq, Nk]
            # MHA 返回的 attn_weights 是 [B, Nq, Nk] (如果 average_attn_weights=True)
            # 或 [B, H, Nq, Nk] (如果 average_attn_weights=False)
            # 我们需要平均权重 [B, Nq, Nk]
            if attn_weights is not None and attn_weights.dim() == 4:
                 avg_attn_weights = attn_weights.mean(dim=1)
            else:
                 avg_attn_weights = attn_weights  # 已经是平均值或 None

            # 忽略API不一致，返回计算出的权重
            return output, avg_attn_weights

        except Exception as e:
            warnings.warn(f"PyTorch MHA 计算失败 ({e})，回退到手动计算。")
            return self._forward_with_proj(q, k, v, mask=mask, proj_q=proj_q, proj_k=proj_k)

    def _forward_with_proj(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                          proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                          batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        """使用手动计算或预计算投影的回退/手动实现"""
        B, Nq, D = q.shape
        Nk = k.shape[1]
        Nv = v.shape[1]
        H = self.num_heads
        D_head = self.head_dim

        # 获取 Q, K, V 投影
        if proj_q is None or proj_k is None:
            # 手动计算 Q, K 投影 (如果未提供)
            q_weight = self.mha.in_proj_weight[:D, :]
            q_bias = self.mha.in_proj_bias[:D] if self.mha.in_proj_bias is not None else None
            q_proj = F.linear(q, q_weight, q_bias).view(B, Nq, H, D_head).transpose(1, 2)  # [B, H, Nq, D_head]

            k_weight = self.mha.in_proj_weight[D:2*D, :]
            k_bias = self.mha.in_proj_bias[D:2*D] if self.mha.in_proj_bias is not None else None
            k_proj = F.linear(k, k_weight, k_bias).view(B, Nk, H, D_head).transpose(1, 2)  # [B, H, Nk, D_head]
        else:
            # 使用预计算的投影
            q_proj = proj_q  # 假设形状为 [B, H, Nq, D_head]
            k_proj = proj_k  # 假设形状为 [B, H, Nk, D_head]

        # 手动计算 V 投影
        v_weight = self.mha.in_proj_weight[2*D:, :]
        v_bias = self.mha.in_proj_bias[2*D:] if self.mha.in_proj_bias is not None else None
        v_proj = F.linear(v, v_weight, v_bias).view(B, Nv, H, D_head).transpose(1, 2)  # [B, H, Nv, D_head]

        # 计算包含温度的缩放因子
        final_softmax_scale = self.scale / self.temperature  # <--- 使用 temperature

        # 计算注意力分数
        # [B, H, Nq, D_head] @ [B, H, D_head, Nk] -> [B, H, Nq, Nk]
        attn = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * final_softmax_scale

        # 应用掩码 - 假设 mask 中 1 表示保留, 0 表示屏蔽
        if mask is not None:
            attn_mask = mask
            # 确保掩码是布尔型
            if attn_mask.dtype != torch.bool:
                 attn_mask = attn_mask < 0.5  # 假设 0 保留, 1 屏蔽 -> < 0.5 为 True (屏蔽)

            # 调整掩码维度以进行广播
            if attn_mask.dim() == 2:  # [B, Lk] -> [B, 1, 1, Lk]
                attn_mask = attn_mask[:, None, None, :Nk]  # 截取或填充到 Nk
            elif attn_mask.dim() == 3:  # [B, Nq, Lk] -> [B, 1, Nq, Lk]
                 attn_mask = attn_mask[:, None, :, :Nk]  # 截取或填充到 Nk
            # ~attn_mask 为 True 的位置需要屏蔽
            attn = attn.masked_fill(~attn_mask, float('-inf'))  # 使用 ~mask.bool()

        attn_weights = F.softmax(attn, dim=-1)  # [B, H, Nq, Nk]
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        out = torch.matmul(attn_weights, v_proj)  # [B, H, Nq, D_head]
        out = out.transpose(1, 2).reshape(B, Nq, D)  # [B, Nq, D]

        # 应用输出投影
        out_weight = self.mha.out_proj.weight
        out_bias = self.mha.out_proj.bias
        final_out = F.linear(out, out_weight, out_bias)

        # 返回平均权重
        avg_attn_weights = attn_weights.mean(dim=1)  # [B, Nq, Nk]

        # 忽略API不一致，返回计算出的权重
        return final_out, avg_attn_weights

    def get_proj_queries_keys(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """获取投影后的查询和键矩阵"""
        B, Nq, D = q.shape
        Nk = k.shape[1]
        H = self.num_heads
        D_head = self.head_dim

        # 手动计算 Q 投影
        q_weight = self.mha.in_proj_weight[:D, :]
        q_bias = self.mha.in_proj_bias[:D] if self.mha.in_proj_bias is not None else None
        proj_q = F.linear(q, q_weight, q_bias)  # [B, Nq, D]
        proj_q = proj_q.view(B, Nq, H, D_head).transpose(1, 2)  # [B, H, Nq, D_head]

        # 手动计算 K 投影
        k_weight = self.mha.in_proj_weight[D:2*D, :]
        k_bias = self.mha.in_proj_bias[D:2*D] if self.mha.in_proj_bias is not None else None
        proj_k = F.linear(k, k_weight, k_bias)  # [B, Nk, D]
        proj_k = proj_k.view(B, Nk, H, D_head).transpose(1, 2)  # [B, H, Nk, D_head]

        return proj_q, proj_k

class FlashAttention(BaseAttention):
    """使用Flash Attention的高效注意力实现 (适配 flash-attn < 2.0)"""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1,
                 temperature: float = 1.0, causal: bool = False):  # temperature 已存在
        super().__init__(dim, num_heads, dropout_rate)

        if not HAS_FLASH_ATTN:
            warnings.warn(
                "未安装flash-attn，FlashAttention将回退到标准注意力。安装方法: pip install flash-attn"
            )

        # 存储温度值，确保不为零以避免除零错误
        self.temperature = max(temperature, 1e-6)  # temperature 已存在
        self.causal = causal

        # 创建投影层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        """
        使用Flash Attention进行高效的注意力计算
        (之前已修正 temperature 和 scale)
        """
        B, Nq, _ = q.shape
        Nk = k.shape[1]  # 获取键的序列长度
        Nv = v.shape[1]  # 获取值的序列长度

        # 使用预投影的张量或手动投影
        if proj_q is not None:
            q_proj = proj_q.transpose(1, 2)  # [B, H, Nq, D_head] -> [B, Nq, H, D_head]
        else:
            q_proj = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim)

        if proj_k is not None:
            k_proj = proj_k.transpose(1, 2)  # [B, H, Nk, D_head] -> [B, Nk, H, D_head]
        else:
            k_proj = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim)

        v_proj = self.v_proj(v).reshape(B, Nv, self.num_heads, self.head_dim)

        q_input = q_proj

        final_softmax_scale = self.scale / self.temperature

        if HAS_FLASH_ATTN and mask is None:
            try:
                output = flash_attn_func(
                    q_input, k_proj, v_proj,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    softmax_scale=final_softmax_scale,
                    causal=self.causal
                )
                output = output.reshape(B, Nq, self.dim)
                # 忽略API不一致，返回 None
                return self.out_proj(output), None

            except Exception as e:
                if "Causal mask is only supported when seqlen_q == seqlen_k" in str(e) and self.causal:
                     warnings.warn(f"Flash Attention causal mask 要求 Nq ({Nq}) == Nk ({Nk})，回退到标准注意力。")
                else:
                     warnings.warn(f"Flash Attention失败，回退到标准注意力: {e}")

        # 回退到标准注意力
        q_fallback = q_input.transpose(1, 2)
        k_fallback = k_proj.transpose(1, 2)
        v_fallback = v_proj.transpose(1, 2)

        attn = torch.matmul(q_fallback, k_fallback.transpose(-2, -1)) * final_softmax_scale

        if mask is not None:
            attn_mask = mask
            if attn_mask.dtype != torch.bool:
                 attn_mask = attn_mask < 0.5  # 假设 0 保留, 1 屏蔽 -> < 0.5 为 True (屏蔽)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[:, None, None, :Nk]
            elif attn_mask.dim() == 3:
                 attn_mask = attn_mask[:, None, :, :Nk]
            attn = attn.masked_fill(~attn_mask, float('-inf'))  # 使用 ~mask.bool()

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(Nq, Nk, dtype=torch.bool, device=q.device), diagonal=1
            )
            attn = attn.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v_fallback)
        out = out.transpose(1, 2).reshape(B, Nq, self.dim)

        avg_attn_weights = attn_weights.mean(dim=1)

        # 忽略API不一致，返回计算出的权重
        return self.out_proj(out), avg_attn_weights

    def get_proj_queries_keys(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """获取投影后的查询和键矩阵"""
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        proj_q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        proj_k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        return proj_q, proj_k

class FlashAttention2(BaseAttention):
    """FlashAttention 2高效注意力实现 - 最新版本特别优化"""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1,
                 temperature: float = 1.0, causal: bool = False):  # temperature 已存在
        super().__init__(dim, num_heads, dropout_rate)

        if not HAS_FLASH_ATTN:
            warnings.warn(
                "未安装flash-attn 2，FlashAttention2将回退到标准注意力。安装方法: pip install flash-attn>=2.0.0"
            )

        self.temperature = max(temperature, 1e-6)  # temperature 已存在
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        """
        使用Flash Attention 2进行高效的注意力计算
        (之前已修正 temperature 和 scale)
        """
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        Nv = v.shape[1]

        if proj_q is not None:
            q_proj = proj_q.transpose(1, 2)
        else:
            q_proj = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim)

        if proj_k is not None:
            k_proj = proj_k.transpose(1, 2)
        else:
            k_proj = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim)

        v_proj = self.v_proj(v).reshape(B, Nv, self.num_heads, self.head_dim)

        q_input = q_proj
        final_softmax_scale = self.scale / self.temperature

        if HAS_FLASH_ATTN:
            try:
                if mask is None:
                    output = flash_attn_func(
                        q_input, k_proj, v_proj,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        softmax_scale=final_softmax_scale,
                        causal=self.causal,
                        return_attn_probs=False
                    )
                else:
                    if mask.dim() == 2 and mask.shape[0] == B and mask.shape[1] == Nk:
                        # 假设 mask 中 1 表示保留, 0 表示屏蔽
                        # unpad_input 需要布尔掩码，True 表示保留
                        mask_bool = mask.bool() if mask.dtype != torch.bool else mask

                        if Nq > Nk:
                             warnings.warn(f"查询序列长度 ({Nq}) 大于掩码/键序列长度 ({Nk})，FlashAttention2变长模式将只考虑掩码覆盖的键。")
                        mask_q = mask_bool[:, :Nq]

                        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q_ = unpad_input(q_input, mask_q)
                        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k_ = unpad_input(k_proj, mask_bool)
                        v_unpad, _, _, _ = unpad_input(v_proj, mask_bool)

                        if q_unpad is None or k_unpad is None or v_unpad is None:
                             warnings.warn("FlashAttention2 unpad_input 返回 None (可能由于全零掩码)，回退到标准实现。")
                             raise RuntimeError("Unpad resulted in None tensor")

                        output_unpad = flash_attn_varlen_func(
                            q_unpad, k_unpad, v_unpad,
                            cu_seqlens_q, cu_seqlens_k,
                            max_seqlen_q_, max_seqlen_k_,
                            dropout_p=self.dropout.p if self.training else 0.0,
                            softmax_scale=final_softmax_scale,
                            causal=self.causal,
                            return_attn_probs=False
                        )
                        output = pad_input(output_unpad, indices_q, B, Nq)
                    else:
                        warnings.warn(f"FlashAttention2接收到不支持的掩码形状 (dim={mask.dim()}, shape={mask.shape}) 或与输入不匹配，期望 [B={B}, Nk={Nk}]。回退到标准实现。")
                        raise NotImplementedError("FlashAttention2目前只支持[B, Nk]形状的2D掩码")

                output = output.reshape(B, Nq, self.dim)
                # 忽略API不一致，返回 None
                return self.out_proj(output), None

            except Exception as e:
                warnings.warn(f"Flash Attention 2失败，回退到标准实现: {e}")

        # 回退到标准注意力
        q_fallback = q_input.transpose(1, 2)
        k_fallback = k_proj.transpose(1, 2)
        v_fallback = v_proj.transpose(1, 2)

        attn = torch.matmul(q_fallback, k_fallback.transpose(-2, -1)) * final_softmax_scale

        if mask is not None:
            attn_mask = mask
            if attn_mask.dtype != torch.bool:
                 attn_mask = attn_mask < 0.5  # 假设 0 保留, 1 屏蔽 -> < 0.5 为 True (屏蔽)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[:, None, None, :Nk]
            elif attn_mask.dim() == 3:
                 attn_mask = attn_mask[:, None, :, :Nk]
            attn = attn.masked_fill(~attn_mask, float('-inf'))  # 使用 ~mask.bool()

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(Nq, Nk, dtype=torch.bool, device=q.device), diagonal=1
            )
            attn = attn.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v_fallback)
        out = out.transpose(1, 2).reshape(B, Nq, self.dim)

        avg_attn_weights = attn_weights.mean(dim=1)

        # 忽略API不一致，返回计算出的权重
        return self.out_proj(out), avg_attn_weights

    def get_proj_queries_keys(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """获取投影后的查询和键矩阵"""
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        proj_q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        proj_k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        return proj_q, proj_k

class XFormersAttention(BaseAttention):
    """使用xFormers库的高效注意力实现"""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1,
                 temperature: float = 1.0,  # <--- 添加 temperature 参数
                 attention_type: str = "scaled_dot_product"):  # attention_type 参数似乎未使用
        super().__init__(dim, num_heads, dropout_rate)

        self.has_xformers = HAS_XFORMERS
        if not self.has_xformers:
            warnings.warn(
                "未安装xformers，XFormersAttention将使用标准注意力回退。安装方法: pip install xformers"
            )

        # 存储温度值
        self.temperature = max(temperature, 1e-6)  # <--- 存储 temperature

        # 创建投影层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        Nv = v.shape[1]

        # 使用预投影或手动投影
        if proj_q is not None:
            q_proj = proj_q  # 假设形状 [B, H, Nq, D_head]
        else:
            q_proj = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)

        if proj_k is not None:
            k_proj = proj_k  # 假设形状 [B, H, Nk, D_head]
        else:
            k_proj = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        v_proj = self.v_proj(v).reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nv, D_head]

        # 计算包含温度的缩放因子
        final_softmax_scale = self.scale / self.temperature  # <--- 使用 temperature

        # 如果有xFormers可用
        if self.has_xformers:
            try:
                # 准备 attn_bias (xFormers 的掩码格式)
                # xFormers 期望 mask 中非零值表示保留，零表示屏蔽
                # 或者是一个加性偏置 (-inf 表示屏蔽)
                attn_bias = None
                if mask is not None:
                    # 假设 mask 中 1 表示保留, 0 表示屏蔽
                    if mask.dtype != torch.bool:
                         mask = mask > 0.5  # 转换为布尔型，True 表示保留
                    # xformers 需要加性偏置，True (保留) -> 0, False (屏蔽) -> -inf
                    attn_bias = xformers.ops.fmha.attn_bias.LowerTriangularMask.from_mask(mask, mask_value=float("-inf"))
                    # 注意: xformers 的掩码处理可能需要更复杂的转换，取决于具体掩码类型
                    # 这里使用了一个简单的转换，可能需要根据实际掩码格式调整
                    # 例如，对于 key_padding_mask [B, Lk]，可能需要 xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(...)
                    warnings.warn("XFormersAttention 的掩码处理可能不完整，请根据具体掩码类型验证。")

                # 使用 xFormers 的 memory_efficient_attention
                # 注意：xFormers 不直接接受 softmax_scale，它使用标准 scale
                # 如果需要 temperature，需要回退或修改 xFormers (不推荐)
                if self.temperature != 1.0:
                     warnings.warn("XFormersAttention 不直接支持 temperature != 1.0，将回退到标准计算。")
                     raise NotImplementedError("XFormers temperature scaling not implemented")

                output = xformers.ops.memory_efficient_attention(
                    q_proj.contiguous(),  # 需要 contiguous
                    k_proj.contiguous(),
                    v_proj.contiguous(),
                    attn_bias=attn_bias,
                    p=self.dropout.p if self.training else 0.0,
                    # scale 参数在 xFormers 内部处理，通常是 1/sqrt(d_head)
                )
                # xFormers 输出形状 [B, Nq, H, D_head]，需要调整
                output = output.transpose(1, 2).reshape(B, Nq, self.dim)

                # xFormers 不返回注意力权重
                # 忽略API不一致，返回 None
                return self.out_proj(output), None

            except Exception as e:
                warnings.warn(f"xFormers Attention失败，回退到标准实现: {e}")

        # 回退到标准注意力实现
        q_fallback = q_proj
        k_fallback = k_proj
        v_fallback = v_proj

        # 计算注意力分数
        attn = torch.matmul(q_fallback, k_fallback.transpose(-2, -1)) * final_softmax_scale  # <--- 使用 temperature

        # 应用掩码 - 假设 mask 中 1 表示保留, 0 表示屏蔽
        if mask is not None:
            attn_mask = mask
            if attn_mask.dtype != torch.bool:
                 attn_mask = attn_mask < 0.5  # 假设 0 保留, 1 屏蔽 -> < 0.5 为 True (屏蔽)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[:, None, None, :Nk]
            elif attn_mask.dim() == 3:
                 attn_mask = attn_mask[:, None, :, :Nk]
            # 修正掩码逻辑
            attn = attn.masked_fill(~attn_mask, float('-inf'))  # <--- 修正: 使用 ~mask.bool()

        # 注意：XFormersAttention 没有 causal 参数，如果需要因果掩码，需在此处添加
        # if self.causal: ...

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v_fallback)
        out = out.transpose(1, 2).reshape(B, Nq, self.dim)

        avg_attn_weights = attn_weights.mean(dim=1)

        # 忽略API不一致，返回计算出的权重
        return self.out_proj(out), avg_attn_weights

    def get_proj_queries_keys(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """获取投影后的查询和键矩阵"""
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        proj_q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        proj_k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        return proj_q, proj_k

class ProbSparseAttention(BaseAttention):
    """概率稀疏注意力实现 (自定义或使用xFormers)"""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1,
                 temperature: float = 1.0,  # <--- 添加 temperature 参数
                 sparsity_config: Optional[Dict] = None):  # 使用更通用的配置
        super().__init__(dim, num_heads, dropout_rate)

        self.has_xformers = HAS_XFORMERS  # 检查 xFormers 可用性
        # 存储温度值
        self.temperature = max(temperature, 1e-6)  # <--- 存储 temperature

        # 稀疏性相关配置 (如果使用自定义实现)
        self.sparsity_config = sparsity_config if sparsity_config else {}
        self.factor = self.sparsity_config.get('factor', 5)  # Informer 中的因子 c
        self.sampling_cache = {}  # 用于缓存采样索引

        # 创建投影层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _prob_QK(self, Q, K, sample_k, batch_idx):
        """计算概率稀疏 QK 分数和掩码 (自定义实现)"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 计算 K 的采样
        # 注意：原始Informer实现更复杂，这里是简化版本
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # 缓存采样索引以提高效率
        # 修正缓存性能问题：直接存储在设备上
        cache_key = f"{L_K}_{sample_k}"
        if cache_key not in self.sampling_cache or self.sampling_cache[cache_key].device != Q.device:
            # index_sample = torch.randint(0, L_K, (L_Q, sample_k)) # .cpu() # <--- 移除 .cpu()
            # 使用 torch.randperm 可能更好，避免重复采样
            perms = torch.stack([torch.randperm(L_K, device=Q.device) for _ in range(L_Q)], dim=0)
            index_sample = perms[:, :sample_k]
            self.sampling_cache[cache_key] = index_sample
        else:
            index_sample = self.sampling_cache[cache_key]  # .to(Q.device) # <--- 移除 .to(device)

        K_sample = K.gather(2, index_sample.unsqueeze(1).unsqueeze(-1).expand(B, H, L_Q, sample_k, E))

        # 计算 Q 和 K_sample 的点积
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [B, H, L_Q, sample_k]

        # 计算稀疏度量 M = max(Q) - mean(Q)
        M = Q_K_sample.max(dim=-1)[0] - torch.div(Q_K_sample.sum(dim=-1), sample_k)  # [B, H, L_Q]

        # 计算 Top-k 稀疏掩码
        M_top = M.topk(k=min(L_K, int(self.factor * math.log(L_K))), dim=-1)[1]  # [B, H, top_k]

        # 创建掩码 (True 表示保留)
        mask = torch.zeros((B, H, L_Q, L_K), dtype=torch.bool, device=Q.device)
        mask.scatter_(-1, M_top.unsqueeze(-1).expand(-1, -1, -1, L_K), True)  # 这里逻辑可能需要调整

        # 注意：原始Informer实现是选择Top-k的Q，然后这些Q与所有K计算。这里是选择Top-k的QK度量对应的Q。
        # 这个简化实现可能与原论文有差异。

        return mask, M_top  # 返回掩码和 top-k 索引 (可能用于调试)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                proj_q: Optional[Tensor] = None, proj_k: Optional[Tensor] = None,
                batch_idx: int = 0) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值类型改为 Optional[Tensor]
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        Nv = v.shape[1]

        # 投影
        if proj_q is not None:
            q_proj = proj_q  # [B, H, Nq, D_head]
        else:
            q_proj = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)

        if proj_k is not None:
            k_proj = proj_k  # [B, H, Nk, D_head]
        else:
            k_proj = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        v_proj = self.v_proj(v).reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nv, D_head]

        # 计算包含温度的缩放因子
        final_softmax_scale = self.scale / self.temperature  # <--- 使用 temperature

        # 尝试使用 xFormers 稀疏注意力 (如果可用且配置允许)
        # 注意: xFormers 的稀疏注意力 API 可能需要特定格式的 layout
        use_xformers_sparse = self.has_xformers and self.sparsity_config.get('use_xformers', False)
        if use_xformers_sparse:
            try:
                # 准备稀疏布局 (layout) - 这通常需要预先计算
                # layout = ... # 需要根据 xFormers API 和稀疏模式计算
                # raise NotImplementedError("xFormers 稀疏布局计算未实现")
                warnings.warn("xFormers 稀疏注意力需要预计算布局，当前未实现，将回退。")
                raise NotImplementedError("xFormers sparse layout not implemented")

                # output = xformers.ops.sparse_attention(...)
                # output = output.transpose(1, 2).reshape(B, Nq, self.dim)
                # return self.out_proj(output), None # xFormers 不返回权重

            except (AttributeError, NotImplementedError, Exception) as e:
                warnings.warn(f"xFormers稀疏注意力失败或未实现，使用自定义实现: {e}")

        # 自定义概率稀疏注意力实现
        sample_k = self.sparsity_config.get('sample_k', min(25, Nk))  # 采样超参数
        # 计算稀疏掩码 (True 表示保留)
        sparse_mask, _ = self._prob_QK(q_proj, k_proj, sample_k, batch_idx)  # [B, H, Nq, Nk]

        # 计算注意力分数
        attn = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * final_softmax_scale  # <--- 使用 temperature

        # 应用概率稀疏掩码 (False 的位置屏蔽)
        attn = attn.masked_fill(~sparse_mask, float('-inf'))

        # 应用外部传入的掩码 - 假设 mask 中 1 表示保留, 0 表示屏蔽
        if mask is not None:
            attn_mask = mask
            if attn_mask.dtype != torch.bool:
                 attn_mask = attn_mask < 0.5  # 假设 0 保留, 1 屏蔽 -> < 0.5 为 True (屏蔽)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[:, None, None, :Nk]
            elif attn_mask.dim() == 3:
                 attn_mask = attn_mask[:, None, :, :Nk]
            # 修正掩码逻辑
            attn = attn.masked_fill(~attn_mask, float('-inf'))  # <--- 修正: 使用 ~mask.bool()

        # 注意：ProbSparseAttention 没有 causal 参数，如果需要因果掩码，需在此处添加

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v_proj)
        out = out.transpose(1, 2).reshape(B, Nq, self.dim)

        avg_attn_weights = attn_weights.mean(dim=1)

        # 忽略API不一致，返回计算出的权重
        return self.out_proj(out), avg_attn_weights

    def get_proj_queries_keys(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """获取投影后的查询和键矩阵"""
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        proj_q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        proj_k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        return proj_q, proj_k


def create_attention_from_config(config: Dict[str, Any], random_seed: Optional[int] = None) -> BaseAttention:
    """
    根据配置创建对应的注意力机制实例

    参数:
        config: 配置字典, 包含 type, hidden_dims, num_heads 等
        random_seed: 随机种子 (当前未使用)

    返回:
        注意力机制实例
    """
    attention_type = config.get("type", "standard").lower()
    # 注意: 配置中使用 hidden_dims，但基类使用 dim
    dim = config.get("hidden_dims")
    num_heads = config.get("num_heads")
    dropout_rate = config.get("dropout_rate", 0.1)
    temperature = config.get("temperature", 1.0)  # 统一获取 temperature
    causal = config.get("causal", False)          # 统一获取 causal
    sparsity_config = config.get("sparsity_config", {}) # 获取 prob_sparse 配置

    if dim is None or num_heads is None:
        raise ValueError("注意力配置必须包含 'hidden_dims' 和 'num_heads'")

    # 根据注意力类型创建实例
    if attention_type == 'standard':
        # 传递 temperature
        return StandardAttention(dim, num_heads, dropout_rate, temperature)

    elif attention_type == 'flash_attention': # 对应 FlashAttention2
        if HAS_FLASH_ATTN:
            logging.info("Using FlashAttention2 as requested by config.")
            return FlashAttention2(dim, num_heads, dropout_rate, temperature, causal)
        else:
             warnings.warn("FlashAttention (v2) configured but flash-attn not installed. Falling back to StandardAttention.")
             return StandardAttention(dim, num_heads, dropout_rate, temperature)

    elif attention_type in ['flash', 'flash1']: # 对应 FlashAttention (v1)
        if HAS_FLASH_ATTN:
             logging.info("Using FlashAttention (v1) as requested by config.")
             return FlashAttention(dim, num_heads, dropout_rate, temperature, causal)
        else:
             warnings.warn("FlashAttention (v1) configured but flash-attn not installed. Falling back to StandardAttention.")
             return StandardAttention(dim, num_heads, dropout_rate, temperature)

    elif attention_type in ['prob_sparse', 'prob_sp']:
        # 传递 temperature 和 sparsity_config
        # factor 参数现在由 sparsity_config 内部处理
        return ProbSparseAttention(dim, num_heads, dropout_rate, temperature, sparsity_config)

    elif attention_type in ['xformers', 'xformer']:
        # 传递 temperature
        # attention_flavor 参数在当前 XFormersAttention 类中未使用
        # attention_flavor = config.get('attn_flavor', 'scaled_dot_product') # 如果需要可以获取
        return XFormersAttention(dim, num_heads, dropout_rate, temperature)

    else:
        raise ValueError(f"不支持的注意力类型: {attention_type}")

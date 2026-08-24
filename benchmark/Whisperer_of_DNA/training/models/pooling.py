from __future__ import annotations
"""
池化层的实现
输入: ExpertChoiceMoE 的输出 [B, S*e, D]
目标: 对每个专家 e 独立地沿序列维度 S 进行池化，输出 [B, e, D]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Any, Tuple, Union, Literal
import math
import warnings
from entmax import entmax_bisect as entmax_bisect_lib  # 导入库函数并重命名以避免冲突

class BasePooling(nn.Module):
    """池化基类"""
    def __init__(self, num_experts: int, expert_dim: int):
        super().__init__()
        self.num_experts = num_experts  # 专家数量 (e)
        self.expert_dim = expert_dim    # 每个专家的维度 (D)

    def forward(self,
                expert_output: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        前向传播函数

        参数:
            expert_output: ExpertChoiceMoE 的输出 [batch_size, seq_len * num_experts, expert_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]，True表示有效位置，False表示需要掩盖的位置

        返回:
            pooled_output: 池化后的输出 [batch_size, num_experts, expert_dim]
            pooling_weights: 池化权重 [batch_size, num_experts, seq_len]
        """
        raise NotImplementedError("子类必须实现forward方法")

    def _reshape_input(self, expert_output: Tensor) -> Tuple[Tensor, int, int, int, int]:
        """将输入 [B, S*e, D] 变形为 [B, e, S, D]"""
        batch_size, seq_len_times_experts, expert_dim = expert_output.shape
        if seq_len_times_experts % self.num_experts != 0:
            raise ValueError(f"输入张量的第二维 ({seq_len_times_experts}) 不能被专家数 ({self.num_experts}) 整除。")
        seq_len = seq_len_times_experts // self.num_experts

        # [B, S*e, D] -> [B, S, e, D]
        x_reshaped = expert_output.view(batch_size, seq_len, self.num_experts, self.expert_dim)
        # [B, S, e, D] -> [B, e, S, D]
        x_permuted = x_reshaped.permute(0, 2, 1, 3)
        return x_permuted, batch_size, seq_len, self.num_experts, self.expert_dim

    def _apply_mask_to_scores(self, scores: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        """将掩码应用于注意力或门控分数 [B, e, S, ...]"""
        if attention_mask is not None:
            # attention_mask: [B, S]
            # mask: [B, 1, S, 1] for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            # Ensure mask is boolean
            mask = mask.bool()
            # Expand mask to match score dimensions if necessary (e.g., for SelfAttentionPooling [B, e, 1, S])
            while mask.dim() < scores.dim():
                 mask = mask.unsqueeze(-1)
            # Align dimensions, assuming the sequence dimension is the second to last (-2 for [B,e,S,1], -1 for [B,e,1,S])
            if scores.shape[-1] == mask.shape[-2]: # Scores are [B, e, 1, S]
                 mask = mask.transpose(-2,-1) # -> [B, 1, 1, S]
            elif scores.shape[-2] != mask.shape[-2]: # Scores are [B, e, S, 1]
                 pass # Mask is already [B, 1, S, 1]
            else:
                 # Fallback or error if dimensions don't match as expected
                 warnings.warn(f"Mask shape {attention_mask.shape} could not be aligned with score shape {scores.shape}. Skipping masking.")
                 return scores

            scores = scores.masked_fill(~mask, float('-inf'))
        return scores

    def _apply_mask_to_values(self, values: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        """将掩码应用于值张量 [B, e, S, D]"""
        if attention_mask is not None:
            # attention_mask: [B, S]
            # mask: [B, 1, S, 1] for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).bool()
            values = values.masked_fill(~mask, 0.0) # Fill masked values with 0
        return values


class SelfAttentionPooling(BasePooling):
    """
    自注意力池化实现。
    使用 e 个可学习的查询向量，每个向量关注对应专家的序列信息。
    """
    def __init__(self, num_experts: int, expert_dim: int,
                 dropout_rate: float = 0.1, alpha: float = 1.5):
        super().__init__(num_experts, expert_dim)

        self.query_vectors = nn.Parameter(torch.empty(1, num_experts, 1, expert_dim))
        nn.init.xavier_uniform_(self.query_vectors) # 使用 Xavier 初始化

        # K 和 V 从变形后的输入 [B, e, S, D] 投影
        self.k_proj = nn.Linear(expert_dim, expert_dim)
        self.v_proj = nn.Linear(expert_dim, expert_dim)

        self.scale = expert_dim ** -0.5
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                expert_output: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        x, batch_size, seq_len, _, _ = self._reshape_input(expert_output) # x: [B, e, S, D]

        # 计算 K 和 V
        k = self.k_proj(x) # [B, e, S, D]
        v = self.v_proj(x) # [B, e, S, D]

        # 准备 Q
        q = self.query_vectors.expand(batch_size, -1, -1, -1) # [B, e, 1, D]

        # 计算注意力分数 [B, e, 1, S]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用掩码 [B, 1, 1, S]
        attn_scores = self._apply_mask_to_scores(attn_scores, attention_mask)

        # 计算池化权重 [B, e, 1, S]
        pooling_weights = entmax_bisect_lib(attn_scores, alpha=self.alpha, dim=-1)
        # pooling_weights = self.dropout(pooling_weights)

        # 加权求和 V: [B, e, 1, S] @ [B, e, S, D] -> [B, e, 1, D]
        pooled_output = torch.matmul(pooling_weights, v)

        # 去除多余维度
        pooled_output = pooled_output.squeeze(2) # [B, e, D]
        final_pooling_weights = pooling_weights.squeeze(2) # [B, e, S]

        return pooled_output, final_pooling_weights

class FFNGatedPooling(BasePooling):
    """
    前馈网络门控池化实现。
    为每个专家学习一个独立的门控网络（Linear或FFN），计算序列token的重要性，
    然后使用 entmax 归一化得到权重进行加权池化。
    """
    def __init__(self, num_experts: int, expert_dim: int,
                 ffn_expansion_factor: int = 1, # 默认=1表示线性层
                 dropout_rate: float = 0.1, alpha: float = 1.5):
        super().__init__(num_experts, expert_dim)

        self.alpha = alpha
        self.gate_networks = nn.ModuleList()
        for _ in range(num_experts):
            if ffn_expansion_factor <= 1:
                # 简单线性门控
                gate_module = nn.Linear(expert_dim, 1)
            else:
                # FFN 门控
                hidden_dim = expert_dim * ffn_expansion_factor
                gate_module = nn.Sequential(
                    nn.Linear(expert_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, 1)
                )
            self.gate_networks.append(gate_module)

    def forward(self,
                expert_output: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        x, batch_size, seq_len, _, _ = self._reshape_input(expert_output) # x: [B, e, S, D]

        gate_scores_list = []
        for i in range(self.num_experts):
            # 对每个专家的 [B, S, D] 切片应用对应的门控网络
            expert_slice = x[:, i, :, :] # [B, S, D]
            scores = self.gate_networks[i](expert_slice) # [B, S, 1]
            gate_scores_list.append(scores)

        # 合并分数 [B, e, S, 1]
        gate_scores = torch.stack(gate_scores_list, dim=1)

        # 应用掩码 [B, 1, S, 1]
        gate_scores = self._apply_mask_to_scores(gate_scores, attention_mask)

        # 沿序列维度 S (dim=2) 归一化得到池化权重 [B, e, S, 1]
        pooling_weights = entmax_bisect_lib(gate_scores, alpha=self.alpha, dim=2)

        # 加权求和: [B, e, S, D] * [B, e, S, 1] -> sum over S -> [B, e, D]
        pooled_output = torch.sum(x * pooling_weights, dim=2)

        # 或者使用 matmul:
        # pooling_weights_matmul = pooling_weights.transpose(-2, -1) # [B, e, 1, S]
        # pooled_output = torch.matmul(pooling_weights_matmul, x).squeeze(2) # [B, e, 1, S] @ [B, e, S, D] -> [B, e, 1, D] -> [B, e, D]

        final_pooling_weights = pooling_weights.squeeze(-1) # [B, e, S]

        return pooled_output, final_pooling_weights


class MeanPooling(BasePooling):
    """平均池化实现 (使用 entmax 进行权重归一化)"""
    def __init__(self, num_experts: int, expert_dim: int, alpha: float = 1.0): # Default alpha=1.0 for softmax-like mean
        super().__init__(num_experts, expert_dim)
        self.alpha = alpha

    def forward(self,
                expert_output: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        x, batch_size, seq_len, _, _ = self._reshape_input(expert_output) # x: [B, e, S, D]

        # 生成原始权重 [B, e, S]
        if attention_mask is None:
            # 均匀权重
            pooling_weights_raw = torch.ones(batch_size, self.num_experts, seq_len,
                                             device=expert_output.device)
        else:
            # 基于掩码的权重 (有效位置为1, 无效为0)
            # mask: [B, S] -> [B, 1, S] -> [B, e, S]
            mask = attention_mask.unsqueeze(1).expand(-1, self.num_experts, -1)
            pooling_weights_raw = mask.float()
            # 将无效位置设为负无穷，以便 entmax 正确处理
            pooling_weights_raw = pooling_weights_raw.masked_fill(~mask, float('-inf'))


        # 使用 entmax 归一化权重 [B, e, S]
        # Note: If using simple mean, divide by seq_len (or masked length) instead of entmax
        pooling_weights = entmax_bisect_lib(pooling_weights_raw, alpha=self.alpha, dim=-1)

        # 应用权重: [B, e, 1, S] @ [B, e, S, D] -> [B, e, 1, D]
        pooled_output = torch.matmul(pooling_weights.unsqueeze(2), x)

        pooled_output = pooled_output.squeeze(2) # [B, e, D]

        return pooled_output, pooling_weights

class MaxPooling(BasePooling):
    """最大池化实现"""
    def __init__(self, num_experts: int, expert_dim: int):
        super().__init__(num_experts, expert_dim)

    def forward(self,
                expert_output: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        x, batch_size, seq_len, _, expert_dim = self._reshape_input(expert_output) # x: [B, e, S, D]

        # 应用掩码到值 [B, e, S, D]
        if attention_mask is not None:
            # mask: [B, S] -> [B, 1, S, 1]
             mask = attention_mask.unsqueeze(1).unsqueeze(-1).bool()
             x_masked = x.masked_fill(~mask, float('-inf'))
        else:
             x_masked = x

        # 沿序列维度 S (dim=2) 找到最大值
        pooled_output, max_indices = torch.max(x_masked, dim=2) # pooled_output: [B, e, D], max_indices: [B, e, D]

        # --- 生成 One-Hot 权重 ---
        # 我们需要一个 [B, e, S] 的权重张量
        # 由于 max_indices 是 [B, e, D]，每个特征维度可能在 S 中有不同的最大值索引
        # 为了简化并得到一个单一的池化权重表示，我们基于第一个特征维度的最大值索引创建 one-hot 编码
        # 或者，可以基于每个 token 的最大特征值来选择 token 索引
        # 方案2: 基于 token 的最大特征值选择索引
        max_val_per_token, _ = torch.max(x_masked, dim=-1) # [B, e, S] (找到每个 token 最显著的特征值)
        max_token_indices = torch.argmax(max_val_per_token, dim=-1) # [B, e] (找到最显著 token 的索引)

        # 使用选择的 token 索引重新计算池化输出 (确保与权重一致)
        # Gather [B, e, S, D] using indices [B, e] -> [B, e, D]
        pooled_output = torch.gather(x, dim=2, index=max_token_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, expert_dim)).squeeze(2)

        # 生成 one-hot 权重 [B, e, S]
        pooling_weights = F.one_hot(max_token_indices, num_classes=seq_len).float()

        # 如果原始序列全被掩码，则 pooled_output 会是 -inf，权重会是全0 (argmax 返回 0)
        # 可以添加处理，例如返回 0 向量和均匀权重
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) # [B]
            all_masked = (seq_lengths == 0) # [B]
            if torch.any(all_masked):
                pooled_output[all_masked.unsqueeze(1).expand(-1, self.num_experts, -1)] = 0.0
                # For weights, maybe set to uniform? Or keep as 0? Let's keep 0 for consistency.
                # pooling_weights[all_masked.unsqueeze(1).expand(-1, self.num_experts, -1)] = 1.0 / seq_len # Uniform?

        return pooled_output, pooling_weights


class CLSPooling(BasePooling):
    """CLS token池化实现 (选择第一个有效 token)"""
    def __init__(self, num_experts: int, expert_dim: int):
        super().__init__(num_experts, expert_dim)

    def forward(self,
                expert_output: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        x, batch_size, seq_len, num_experts, expert_dim = self._reshape_input(expert_output) # x: [B, e, S, D]

        # 确定要选择的 token 索引 (默认为 0)
        cls_indices = torch.zeros(batch_size, dtype=torch.long, device=expert_output.device) # [B]

        if attention_mask is not None:
            # 如果 CLS token (索引0) 被掩码，则找到第一个未被掩码的 token
            is_cls_masked = ~attention_mask[:, 0] # [B]
            if torch.any(is_cls_masked):
                # 找到第一个 True (有效) 的位置
                first_valid_indices = torch.argmax(attention_mask.float(), dim=1) # [B]
                # 更新需要更新的 cls_indices
                cls_indices = torch.where(is_cls_masked, first_valid_indices, cls_indices)

        # 使用 cls_indices 从 x 中 gather 数据
        # indices shape needs to be [B, e, 1, D] for gather along dim 2
        indices_for_gather = cls_indices.view(batch_size, 1, 1, 1).expand(-1, num_experts, -1, expert_dim)
        pooled_output = torch.gather(x, dim=2, index=indices_for_gather).squeeze(2) # [B, e, D]

        # 生成 one-hot 权重 [B, e, S]
        pooling_weights = F.one_hot(cls_indices, num_classes=seq_len).float() # [B, S]
        pooling_weights = pooling_weights.unsqueeze(1).expand(-1, num_experts, -1) # [B, e, S]

        # 处理完全被掩码的序列
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) # [B]
            all_masked = (seq_lengths == 0) # [B]
            if torch.any(all_masked):
                 pooled_output[all_masked.unsqueeze(1).expand(-1, self.num_experts, -1)] = 0.0
                 pooling_weights[all_masked.unsqueeze(1).expand(-1, self.num_experts, -1)] = 0.0


        return pooled_output, pooling_weights


def create_pooling_from_config(config: Dict[str, Any]) -> BasePooling:
    """
    根据配置创建对应的池化实例

    参数:
        config: 池化配置字典 (通常来自模型配置的 "pooling" 部分)
                需要包含 "type", "num_heads" (作为 num_experts), "head_dims" (作为 expert_dim)
                以及可选的 dropout_rate, alpha, ffn_expansion_factor, temperature (可能未使用)

    返回:
        池化实例
    """
    pooling_type = config.get("type", "mean").lower()
    # 使用 num_heads 和 head_dims 作为 num_experts 和 expert_dim
    num_experts = config.get("num_heads")
    expert_dim = config.get("head_dims")

    if num_experts is None or expert_dim is None:
        raise ValueError("Pooling config 必须包含 'num_heads' (作为 num_experts) 和 'head_dims' (作为 expert_dim)")

    dropout_rate = config.get("dropout_rate", 0.1)
    alpha = config.get("alpha", 1.5) # Default alpha for entmax
    ffn_expansion_factor = config.get("ffn_expansion_factor", 1) # Default 1 for linear gate in FFNGated

    if pooling_type == "self_attention":
        return SelfAttentionPooling(
            num_experts=num_experts,
            expert_dim=expert_dim,
            dropout_rate=dropout_rate,
            alpha=alpha
        )
    elif pooling_type == "ffn_gated":
        return FFNGatedPooling(
            num_experts=num_experts,
            expert_dim=expert_dim,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_rate=dropout_rate,
            alpha=alpha
        )
    elif pooling_type == "mean":
        # MeanPooling alpha default might be 1.0 for softmax-like behavior
        mean_alpha = config.get("alpha", 1.0)
        return MeanPooling(
            num_experts=num_experts,
            expert_dim=expert_dim,
            alpha=mean_alpha
        )
    elif pooling_type == "max":
        return MaxPooling(
            num_experts=num_experts,
            expert_dim=expert_dim,
        )
    elif pooling_type == "cls":
        return CLSPooling(
            num_experts=num_experts,
            expert_dim=expert_dim,
        )
    else:
        warnings.warn(f"未知的池化类型: {pooling_type}，将使用默认的平均池化 (alpha=1.0)")
        return MeanPooling(
            num_experts=num_experts,
            expert_dim=expert_dim,
            alpha=1.0 # Default alpha for mean fallback
        )

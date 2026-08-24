from __future__ import annotations
"""
模型其他层的实现
"""
import torch # <--- 添加 torch 导入
import torch.nn as nn # <--- 添加 nn 导入
import torch.nn.functional as F # <--- 添加 F 导入
import warnings # <--- 添加 warnings 导入
from functools import partial
from typing import Dict, Any, Optional, Tuple, List # <--- 添加 List
from torch import Tensor
from entmax import entmax_bisect as entmax_bisect_lib  # Import entmax
import math # <--- 添加 math 导入
from einops import rearrange, repeat # <--- 添加 einops

# --- 从 pooling.py 导入 ---
# 假设 pooling.py 在同一目录下或已正确安装
try:
    # --- 修改：导入 create_pooling_from_config 和 BasePooling ---
    from .pooling import create_pooling_from_config, BasePooling
    # --- 结束修改 ---
except ImportError:
    # Fallback if running script directly or structure issues
    warnings.warn("Could not import pooling functions from .pooling. Ensure pooling.py is accessible.")
    # Define dummy classes/functions if import fails
    class BasePooling(nn.Module):
        def __init__(self, num_experts: int, expert_dim: int):
            super().__init__()
            self.num_experts = num_experts
            self.expert_dim = expert_dim
        def forward(self, expert_output: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
            raise NotImplementedError

    class SelfAttentionPooling(BasePooling): # Keep dummy SelfAttentionPooling for type hints if needed
        def __init__(self, num_experts: int, expert_dim: int, dropout_rate: float = 0.1, alpha: float = 1.5):
            super().__init__(num_experts, expert_dim)
            warnings.warn("Using dummy SelfAttentionPooling due to import error.")
            self.dummy_layer = nn.Linear(expert_dim, expert_dim)

        def forward(self, expert_output: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
            batch_size_times_seq, _, dim = expert_output.shape
            # Simplified dummy forward
            if self.num_experts == 0: # Avoid division by zero
                 seq_len = 0
                 pooled = torch.zeros(0, 0, dim, device=expert_output.device)
                 dummy_weights = torch.zeros(0, 0, 0, device=expert_output.device)
            else:
                 # --- 修正：Dummy forward 的 reshape 逻辑 ---
                 batch_size, seq_len_times_experts, expert_dim = expert_output.shape
                 if self.num_experts > 0:
                     seq_len = seq_len_times_experts // self.num_experts
                     reshaped = expert_output.view(batch_size, seq_len, self.num_experts, expert_dim) # [B, S, E, D]
                     pooled = reshaped.mean(dim=1) # [B, E, D]
                     dummy_weights = torch.ones(batch_size, self.num_experts, seq_len, device=expert_output.device) / seq_len if seq_len > 0 else torch.zeros(batch_size, self.num_experts, 0, device=expert_output.device)
                 else:
                     pooled = torch.zeros(batch_size, 0, expert_dim, device=expert_output.device)
                     dummy_weights = torch.zeros(batch_size, 0, 0, device=expert_output.device)
                 # --- 结束修正 ---
            return pooled, dummy_weights

    def create_pooling_from_config(config: Dict[str, Any]) -> BasePooling:
        warnings.warn("Using dummy create_pooling_from_config due to import error. Returning dummy SelfAttentionPooling.")
        # Extract necessary args for dummy SelfAttentionPooling
        # --- 修正：使用 num_experts 和 expert_dim ---
        num_experts = config.get("num_experts") # Changed from num_heads
        expert_dim = config.get("expert_dim")   # Changed from head_dims
        # --- 结束修正 ---
        dropout_rate = config.get("dropout_rate", 0.1)
        alpha = config.get("alpha", 1.5)
        if num_experts is None or expert_dim is None:
             # --- 修正：错误消息 ---
             raise ValueError("Dummy create_pooling_from_config requires 'num_experts' and 'expert_dim' in config.")
             # --- 结束修正 ---
        return SelfAttentionPooling(num_experts, expert_dim, dropout_rate, alpha)


class RotaryPositionalEncoding(nn.Module):
    """旋转位置编码 (RoPE) 实现   暂时弃用"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 计算旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _generate_cos_sin(self, positions: Tensor):
        """生成指定位置的余弦和正弦值"""
        # 处理多维positions张量 - 只保留最后一个维度
        if positions.dim() > 1:
            pos_flat = positions.view(-1)
        else:
            pos_flat = positions

        # 扩展维度
        if positions.dim() <= 1:
            t = pos_flat.unsqueeze(1) # [seq_len, 1]
        else:
            t = pos_flat.unsqueeze(-1) # [..., seq_len, 1]

        freqs = torch.outer(t.squeeze(-1).float(), self.inv_freq) # [..., seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [..., seq_len, dim]
        cos = emb.cos() # [..., seq_len, dim]
        sin = emb.sin() # [..., seq_len, dim]
        # Reshape back to original positions shape + dim/2
        cos = cos.view(*positions.shape, self.dim // 2)
        sin = sin.view(*positions.shape, self.dim // 2)
        return cos, sin

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """
        应用旋转位置编码

        Args:
            x: 输入张量，形状为 [..., seq_len, dim]
            positions: 可选的位置索引，形状为 [..., seq_len]

        Returns:
            应用位置编码后的张量
        """
        seq_len = x.size(-2)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        cos, sin = self._generate_cos_sin(positions) # cos/sin shape: [..., seq_len, dim/2]

        # 扩展维度以匹配输入形状
        # Ensure cos/sin have the same number of leading dimensions as x (excluding seq_len and dim)
        num_leading_dims_x = x.ndim - 2
        num_leading_dims_cs = cos.ndim - 2 # Assuming positions led to [..., seq_len, dim/2]

        if num_leading_dims_cs < num_leading_dims_x:
             # Add leading dimensions of size 1
             cos = cos.view(*([1] * (num_leading_dims_x - num_leading_dims_cs)), *cos.shape)
             sin = sin.view(*([1] * (num_leading_dims_x - num_leading_dims_cs)), *sin.shape)
        elif num_leading_dims_cs > num_leading_dims_x:
             # This case might indicate an issue, but let's try broadcasting
             warnings.warn("RoPE: cos/sin have more leading dimensions than input x. Broadcasting might be incorrect.")


        # 提取偶数和奇数位置的特征
        x_even = x[..., 0::2]  # 偶数位置 [..., seq_len, dim/2]
        x_odd = x[..., 1::2]   # 奇数位置 [..., seq_len, dim/2]

        # Ensure cos/sin are broadcastable to x_even/x_odd shape
        # Shape of x_even/x_odd: [..., seq_len, dim/2]
        # Shape of cos/sin:      [..., seq_len, dim/2] (after generation)
        # Need to align the [...] parts - handled by broadcasting if leading dims match or are 1

        # Apply rotation using complex number multiplication analogy: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # x' = x * exp(i * m * theta)
        # x_even' = x_even * cos - x_odd * sin
        # x_odd'  = x_even * sin + x_odd * cos

        # Check shapes before operation
        if x_even.shape[-1] != cos.shape[-1]:
             raise ValueError(f"RoPE dimension mismatch: x_even dim {x_even.shape[-1]} vs cos dim {cos.shape[-1]}")

        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave back: Create output tensor and fill
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, dim: int, max_seq_len: int = 2048): # max_seq_len 现在代表最大块数/序列长度
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # 初始化可学习的位置编码
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)  # 初始化为小值

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """
        应用可学习的位置编码

        Args:
            x: 输入张量，形状为 [batch_size, num_blocks_or_seq, dim]
            positions: 可选的位置索引，形状为 [batch_size, num_blocks_or_seq]，默认为 None (未使用)

        Returns:
            加入位置编码后的张量
        """
        num_blocks_or_seq = x.size(1) # 这里的 seq_len 是块的数量或原始序列长度(当L=1时)
        if num_blocks_or_seq > self.max_seq_len:
             warnings.warn(f"Input sequence length ({num_blocks_or_seq}) exceeds LearnablePositionalEncoding max_seq_len ({self.max_seq_len}). Truncating position embeddings.")


        # 添加位置编码
        # 使用广播机制将 [1, num_blocks_or_seq, dim] 加到 [batch_size, num_blocks_or_seq, dim]
        x = x + self.position_embeddings[:, :num_blocks_or_seq, :]
        return x


class MOEffn(nn.Module):
    """池化后的MOE层实现"""
    def __init__(self,
                 head_dims: int,
                 output_dim: int,
                 num_heads: int,
                 gate_expansion_factor: int = 8,
                 gate_sharing: bool = True,
                 exports_expansion_factor: int = 4,
                 exports_sharing: bool = False,
                 dropout_rate: float = 0.1,
                 alpha: float = 1.5):  # Add alpha parameter
        super().__init__()

        self.head_dims = head_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.gate_sharing = gate_sharing
        self.exports_sharing = exports_sharing
        self.alpha = alpha  # Store alpha

        # 添加头信息交互层 - 用于头间信息交流
        # 将所有头的信息展平后进行线性变换，再重塑回原始形状
        self.head_interaction_proj = nn.Linear(num_heads * head_dims, num_heads * head_dims)

        # 创建门控网络
        gate_hidden_dim = head_dims * gate_expansion_factor

        if gate_sharing:
            self.gate_network = nn.Sequential(
                nn.Linear(head_dims, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(gate_hidden_dim, 1)
            )
        else:
            self.gate_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_dims, gate_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(gate_hidden_dim, 1)
                ) for _ in range(num_heads)
            ])

        # 创建专家网络
        expert_hidden_dim = head_dims * exports_expansion_factor

        if exports_sharing:
            self.expert_network = nn.Sequential(
                nn.Linear(head_dims, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_hidden_dim, output_dim)
            )
        else:
            self.expert_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_dims, expert_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(expert_hidden_dim, output_dim)
                ) for _ in range(num_heads)
            ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播

        参数:
            x: 输入张量，维度为 [batch_size, num_heads, head_dims]

        返回:
            weighted_output: 加权后的输出，维度为 [batch_size, output_dim]
            gate_weights: 门控权重，维度为 [batch_size, num_heads, 1]
            expert_outputs: 各专家的原始输出，维度为 [batch_size, num_heads, output_dim]
        """
        batch_size, num_heads, head_dims = x.shape

        # 信息交互层实现
        # 步骤1: 展平为 [batch_size, num_heads*head_dims]
        x_flat = x.reshape(batch_size, -1)

        # 步骤2: 通过线性变换进行信息交流
        x_interacted = self.head_interaction_proj(x_flat)

        # 步骤3: 重塑回 [batch_size, num_heads, head_dims]
        x = x_interacted.reshape(batch_size, num_heads, head_dims)

        # 计算门控权重
        if self.gate_sharing:
            gate_logits = self.gate_network(x) # [B, H, 1]
        else:
            gate_logits_list = [self.gate_networks[i](x[:, i, :].unsqueeze(1)) for i in range(num_heads)]
            gate_logits = torch.cat(gate_logits_list, dim=1) # [B, H, 1]

        # 将门控logits转换为权重，使用entmax确保所有权重和为1
        gate_weights = entmax_bisect_lib(gate_logits, alpha=self.alpha, dim=1)  # Use entmax

        # 计算专家输出
        if self.exports_sharing:
            expert_outputs = self.expert_network(x) # [B, H, output_dim]
        else:
            expert_outputs_list = [self.expert_networks[i](x[:, i, :].unsqueeze(1)) for i in range(num_heads)]
            expert_outputs = torch.cat(expert_outputs_list, dim=1) # [B, H, output_dim]

        # 计算加权输出，[B, H, output_dim] * [B, H, 1] -> [B, H, output_dim]
        weighted_outputs = expert_outputs * gate_weights
        # 沿着专家维度求和，得到最终输出 [B, output_dim]
        final_output = weighted_outputs.sum(dim=1)

        # Return None for weights/expert outputs during training for efficiency
        if self.training:
            return final_output, None, None
        else:
            return final_output, gate_weights, expert_outputs


class ExpertChoiceMoE(nn.Module):
    """
    Expert Choice Mixture of Experts Layer (Batch-Aware).

    按照 "Heterogeneous MoE via Expert Choice" 描述实现，但修改为在批次内独立路由，
    避免混合不同样本的 token。每个专家独立选择每个样本中的 top-k 个 token 进行处理。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 ExpertChoiceMoE 层。

        Args:
            config (Dict[str, Any]): 配置字典，应包含以下键:
                - num_experts (int): 专家数量 (e)。
                - experts_dims (int): 输入/输出 token 的维度 (d_model)。
                                      专家处理的 token 维度。
                - exports_expansion_factor (int): 专家 FFN 隐藏层的扩展因子。
                                                  隐藏维度 = experts_dims * exports_expansion_factor。
                - exports_sharing (bool): 是否所有专家共享相同的 FFN 参数。
                - capacity_factor (float): 容量因子 (c)，用于计算每个专家的容量
                                           k = ceil(seq_len * c / e)。
                - dropout_rate (float): FFN 中的 dropout 比率。
        """
        super().__init__()

        self.num_experts = config["num_experts"]
        self.d_model = config["experts_dims"]
        self.experts_sharing = config["exports_sharing"]
        self.capacity_factor = config["capacity_factor"]
        dropout_rate = config.get("dropout_rate", 0.1)
        ffn_hidden_dim = self.d_model * config["exports_expansion_factor"]
        activation = config.get("activation", "gelu") # Get activation from config
        self.activation_fn = self._get_activation(activation) # Store activation

        # Gating network (Wg): Linear layer mapping token dimension to number of experts
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)

        # Expert networks (FFNs)
        def create_expert_ffn():
            return nn.Sequential(
                nn.Linear(self.d_model, ffn_hidden_dim),
                self.activation_fn, # Use stored activation
                nn.Dropout(dropout_rate),
                nn.Linear(ffn_hidden_dim, self.d_model),
                nn.Dropout(dropout_rate)
            )

        if self.experts_sharing:
            self.expert_network = create_expert_ffn()
        else:
            self.expert_networks = nn.ModuleList([create_expert_ffn() for _ in range(self.num_experts)])

    # _get_activation remains for potential future use
    def _get_activation(self, activation_name: str):
        """Helper function to get activation layer."""
        if activation_name.lower() == "relu":
            return nn.ReLU()
        elif activation_name.lower() == "gelu":
            return nn.GELU()
        elif activation_name.lower() == "silu" or activation_name.lower() == "swish":
            return nn.SiLU()
        else:
            warnings.warn(f"Unsupported activation: {activation_name}. Using GELU.")
            return nn.GELU()

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Batch-Aware Expert Choice MoE 前向传播，支持掩码。

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]。
            attention_mask (Optional[Tensor]): 注意力掩码 [batch_size, seq_len]，
                                                True表示有效位置，False表示需要掩盖的位置。

        Returns:
            Tensor: 输出张量，形状与输入相同 [batch_size, seq_len, d_model]。
        """
        batch_size, seq_len, d_model = x.shape
        num_experts = self.num_experts

        # Handle sequence length of 0
        if seq_len == 0:
            return torch.zeros_like(x)

        # --- Batch-Aware Gating ---
        # Apply gate directly to [B, S, D] input
        router_logits = self.gate(x) # Shape: [B, S, E]

        if attention_mask is not None:
            # Ensure mask is boolean and expanded for broadcasting
            mask_bool = attention_mask.bool().unsqueeze(-1) # [B, S, 1]
            # Mask out logits for padding tokens
            router_logits = router_logits.masked_fill(~mask_bool, float('-inf'))

        # Use float() for softmax stability if using mixed precision
        router_probs = F.softmax(router_logits.float(), dim=-1).type_as(x) # Shape: [B, S, E]

        # --- Batch-Aware Expert Choice Routing ---
        # 计算专家容量 k
        # Calculate effective sequence length if mask is provided
        if attention_mask is not None:
             # Calculate k based on the maximum sequence length in the batch if lengths vary
             # Or use a fixed seq_len if appropriate for the model design
             # For simplicity, let's use the max seq_len (S) here.
             # A more robust approach might consider average length or max valid length.
             effective_seq_len = seq_len # Use max seq len for capacity calculation
        else:
             effective_seq_len = seq_len

        # Calculate capacity k, ensuring it's at least 1
        k_calculated = max(1, int(math.ceil(effective_seq_len * self.capacity_factor / num_experts)))
        k = min(seq_len, k_calculated)

        # Permute to [B, E, S] for topk along sequence dimension
        expert_probs_transposed = router_probs.permute(0, 2, 1) # Shape: [B, E, S]

        if attention_mask is not None:
            # Mask probabilities corresponding to padding tokens before topk
            mask_bool_expanded = attention_mask.bool().unsqueeze(1) # [B, 1, S]
            expert_probs_transposed = expert_probs_transposed.masked_fill(~mask_bool_expanded, float('-inf'))

        # Select top-k tokens *per expert, per sequence*
        # Use float() for topk stability if using mixed precision
        if k <= 0:
            # This case should ideally not happen due to max(1, ...) and seq_len check,
            # but as a safeguard:
            # Create empty tensors with the expected shape if k is 0 or less
            top_k_scores = torch.zeros((batch_size, num_experts, 0), dtype=x.dtype, device=x.device)
            top_k_indices = torch.zeros((batch_size, num_experts, 0), dtype=torch.long, device=x.device)
        else:
            # Ensure the dimension size is valid for topk
            dim_size = expert_probs_transposed.size(2)
            actual_k = min(k, dim_size) # Ensure k is not larger than the dimension size
            if actual_k <= 0: # Handle case where dim_size might be 0
                 top_k_scores = torch.zeros((batch_size, num_experts, 0), dtype=x.dtype, device=x.device)
                 top_k_indices = torch.zeros((batch_size, num_experts, 0), dtype=torch.long, device=x.device)
            else:
                 top_k_scores, top_k_indices = torch.topk(expert_probs_transposed.float(), k=actual_k, dim=2)
                 top_k_scores = top_k_scores.type_as(x)

        # Replace -inf scores (from padding tokens selected if k > num_valid) with 0.0
        valid_scores_mask = torch.isfinite(top_k_scores)
        top_k_scores = top_k_scores.masked_fill(~valid_scores_mask, 0.0)

        # --- Batch-Aware Gather Tokens for Experts ---
        if top_k_indices.numel() == 0: # If k was 0 or dim_size was 0
            tokens_for_experts = torch.zeros((batch_size, num_experts, 0, d_model), dtype=x.dtype, device=x.device)
        else:
            # Expand x for gathering: [B, S, D] -> [B, 1, S, D] -> [B, E, S, D]
            x_expanded = x.unsqueeze(1).expand(-1, num_experts, -1, -1)

            # Expand indices for gathering: [B, E, k] -> [B, E, k, D]
            indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, d_model)

            # Gather tokens along dim=2 (sequence dimension)
            tokens_for_experts = torch.gather(x_expanded, dim=2, index=indices_expanded) # Shape: [B, E, k, D]

        # --- Batch-Aware Process Tokens through Experts ---
        expert_outputs = torch.zeros_like(tokens_for_experts) # Initialize output tensor [B, E, k, D]

        if tokens_for_experts.numel() > 0:
            if self.experts_sharing:
                # Reshape for shared expert: [B, E, k, D] -> [B*E*k, D]
                flat_tokens = tokens_for_experts.reshape(-1, d_model)
                flat_outputs = self.expert_network(flat_tokens)
                expert_outputs = flat_outputs.reshape(batch_size, num_experts, actual_k, d_model) # Use actual_k
            else:
                for i in range(num_experts):
                    # Process each expert's tokens separately
                    expert_tokens = tokens_for_experts[:, i, :, :] # Shape: [B, k, D]
                    # Reshape for expert FFN: [B, k, D] -> [B*k, D]
                    flat_expert_tokens = expert_tokens.reshape(-1, d_model)
                    flat_expert_outputs = self.expert_networks[i](flat_expert_tokens)
                    # Reshape back: [B*k, D] -> [B, k, D]
                    expert_outputs[:, i, :, :] = flat_expert_outputs.reshape(batch_size, actual_k, d_model) # Use actual_k

        # --- Batch-Aware Combine Expert Outputs ---
        # Weight expert outputs by gating scores (G = top_k_scores)
        # G shape: [B, E, k] -> [B, E, k, 1]
        # Xe shape: [B, E, k, D]
        # Scores for any selected padding tokens are 0.0, so they contribute nothing.
        weighted_expert_outputs = expert_outputs * top_k_scores.unsqueeze(-1) # Shape: [B, E, k, D]

        # --- Batch-Aware Scatter-Add ---
        # Scatter-add weighted outputs back to original token positions within each sequence
        # Initialize output tensor
        output = torch.zeros_like(x) # Shape: [B, S, D]

        if weighted_expert_outputs.numel() > 0:
            # Prepare for scatter_add_ along dim=1 (sequence dimension)
            # Source shape: [B, E, k, D] -> [B, E*k, D]
            flat_source = weighted_expert_outputs.reshape(batch_size, num_experts * actual_k, d_model) # Use actual_k
            # Index shape: [B, E, k] -> [B, E*k]
            flat_indices = top_k_indices.reshape(batch_size, num_experts * actual_k) # Use actual_k
            # Expand index for scatter_add_: [B, E*k] -> [B, E*k, D]
            expanded_flat_indices = flat_indices.unsqueeze(-1).expand(-1, -1, d_model)

            # Perform scatter_add_ along sequence dimension (dim=1)
            # Indices corresponding to padding tokens might exist but their source value is zero.
            output.scatter_add_(dim=1, index=expanded_flat_indices, src=flat_source)

        # Final masking step: Ensure padding positions are strictly zero in the output
        if attention_mask is not None:
             output = output.masked_fill(~mask_bool, 0.0) # Use the expanded mask [B, S, 1]

        return output


# --- 新增 Asinh 激活函数模块 (if not already present) ---
class Asinh(nn.Module):
    """反双曲正弦激活函数"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.asinh(x)

class OutputLayer(nn.Module):
    """
    输出层实现 V3 (BN -> FFN(Expand->Activate->Dropout->ContractToPhenotypeDim))
    - FFN 结构: Linear(D_in -> D_exp) -> Activation -> Dropout -> Linear(D_exp -> PhenotypeDim)
    - 输入前的规范化层改为 BatchNorm1d
    """
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()

        self.input_dim = input_dim

        phenotype_dim = config.get("phenotype_dim", 1)
        dropout_rate = config.get("dropout_rate", 0.1)
        activation_name = config.get("activation", "tanh")
        ffn_expansion_factor = config.get("hidden_expansion_dims", 4)

        self.norm_input = nn.BatchNorm1d(input_dim)

        ffn_hidden_dim = input_dim * ffn_expansion_factor
        self.ffn_linear1 = nn.Linear(input_dim, ffn_hidden_dim)

        act_lower = activation_name.lower()
        if act_lower == "relu":
            self.activation_fn = nn.ReLU()
        elif act_lower == "gelu":
            self.activation_fn = nn.GELU()
        elif act_lower == "silu" or act_lower == "swish":
            self.activation_fn = nn.SiLU()
        elif act_lower == "tanh":
            self.activation_fn = nn.Tanh()
        elif act_lower == "asinh":
            self.activation_fn = Asinh()
        elif act_lower == "linear":
            self.activation_fn = nn.Identity()
        else:
            warnings.warn(f"Unsupported activation: {activation_name}. Using Tanh as default for OutputLayer.")
            self.activation_fn = nn.Tanh()

        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_linear2 = nn.Linear(ffn_hidden_dim, phenotype_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            # warnings.warn("OutputLayer received 3D input. Averaging along dim=1.")
            x = x.mean(dim=1)
        elif x.ndim == 1:
            # warnings.warn("OutputLayer received 1D input. Unsqueezing to [1, D].")
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(f"OutputLayer expects 2D [B, D] input, but got {x.ndim}D shape {x.shape}.")

        if x.shape[0] == 0: # Handle empty batch
            return torch.zeros(0, self.ffn_linear2.out_features, device=x.device, dtype=x.dtype)


        if x.shape[1] != self.input_dim:
             raise ValueError(f"OutputLayer input dimension mismatch. Expected {self.input_dim}, got {x.shape[1]}. Input shape: {x.shape}")

        x_norm = self.norm_input(x)
        hidden = self.ffn_linear1(x_norm)
        activated = self.activation_fn(hidden)
        dropped = self.dropout(activated)
        output = self.ffn_linear2(dropped)

        return output


class CNNFeatureExtractor(nn.Module):
    """
    CNN特征提取器组件 - 改进版
    处理单个表型的CNN特征提取，支持多层级和多分支
    明确处理每层的输入/输出通道
    """
    def __init__(self, input_channels: int, cnn_kernels: List[int], cnn_dilations: List[int], 
                 activation_fn: nn.Module, cnn_layers: int = 1, feature_channels: int = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = input_channels
        
        # 每层特征数量 = 内核数量 × 膨胀率数量 × 特征通道数
        features_per_layer = len(cnn_kernels) * len(cnn_dilations) * feature_channels
        
        for layer_idx in range(cnn_layers):
            layer_cnns = nn.ModuleList()
            
            for kernel_size in cnn_kernels:
                for dilation in cnn_dilations:
                    padding = (kernel_size - 1) * dilation // 2
                    
                    # 最后一层输出特征通道，中间层输出更多通道以提高表达能力
                    out_channels = feature_channels
                    
                    conv = nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation
                    )
                    
                    # 对每个卷积分支使用BatchNorm
                    layer_cnns.append(nn.Sequential(
                        conv,
                        nn.BatchNorm1d(out_channels),
                        activation_fn
                    ))
            
            self.layers.append(layer_cnns)
            # 更新下一层的输入通道
            current_channels = features_per_layer
            
        # 输出特征数量 = 最后一层生成的总通道数
        self.output_channels = features_per_layer
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        Args:
            x: [B, C_in, Seq] 输入特征
        Returns:
            [B, C_out, Seq] CNN特征
        """
        for layer_idx, layer_cnns in enumerate(self.layers):
            layer_outputs = []
            
            for cnn_module in layer_cnns:
                # 应用当前层的每个CNN分支
                branch_output = cnn_module(x)
                layer_outputs.append(branch_output)
            
            # 在通道维度上拼接所有分支输出
            x = torch.cat(layer_outputs, dim=1)
            
        return x


class EmbeddingLayer_onlySNP(nn.Module):
    """
    嵌入层 V4 - 优化版 - 仅处理编码后的SNP序列输入。
    CNN提取特征 -> (可选)分块+[LN->Pooling] 或 (L=1时) [LN] -> FFN辅助损失投影 + 输出。
    
    优化点:
    1. 模块化设计，将CNN特征提取分离为独立组件
    2. 提高内存效率，减少中间变量
    3. 代码逻辑分段，提高可读性
    4. 明确的形状注释，便于追踪张量维度
    """
    def __init__(self, input_dim: int, config: Dict[str, Any], phenotype_dim: int,
                 gradient_checkpointing_config: Optional[Dict[str, Any]] = None): # 新增 gradient_checkpointing_config
        super().__init__()
        # === 基本参数初始化 ===
        self.phenotype_dim = phenotype_dim  # E
        self.Block_length = config.get("Block_length")
        
        # 验证配置
        self._validate_config()
        
        # === CNN相关参数 ===
        cnn_kernels = config.get("CNN_Kernel", [3, 5, 7])
        cnn_dilations = config.get("CNN_Dilation", [1, 2])
        cnn_layers = config.get("CNN_layers", 1)
        self.num_cnn_features_per_phenotype = len(cnn_kernels) * len(cnn_dilations)  # C
        self.num_cnn_features_total = self.phenotype_dim * self.num_cnn_features_per_phenotype  # E*C
        
        # === 其他配置参数 ===
        activation_name = config.get("activation", "gelu")
        self.activation_fn = self._get_activation(activation_name)
        dropout_rate = config.get("dropout_rate", 0.1)
        self.position_encoding = config.get("position_encoding", False)
        pooling_alpha = config.get("pooling_alpha", 1.5)
        aux_ffn_expansion_factor = config.get("aux_ffn_expansion_factor", 2)
        
        # === 特征提取器初始化 ===
        # 使用模块化的CNN特征提取器替代之前的嵌套循环结构
        self.feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(
                input_dim, cnn_kernels, cnn_dilations, 
                self.activation_fn, cnn_layers
            ) for _ in range(phenotype_dim)
        ])
        
        # === 池化和归一化层初始化 ===
        self.block_pooling = None
        self.ln_before_pooling = None
        self.ln_before_aux_ffn_no_pool = None
        
        if self.Block_length > 1:
            self._setup_block_pooling(config, dropout_rate, pooling_alpha)
        else:  # Block_length == 1
            self.ln_before_aux_ffn_no_pool = nn.LayerNorm(self.num_cnn_features_per_phenotype)
        
        # === 辅助损失网络初始化 ===
        self._setup_aux_networks(aux_ffn_expansion_factor)
        
        # === Dropout和位置编码初始化 ===
        self.dropout = nn.Dropout(dropout_rate)
        self._setup_positional_encoding(config)

    def _validate_config(self):
        """验证配置参数合法性"""
        if self.Block_length is None:
            raise ValueError("Embedding_onlySNP config requires 'Block_length'.")
        if not isinstance(self.Block_length, int) or self.Block_length <= 0:
            raise ValueError("'Block_length' must be a positive integer for Embedding_onlySNP.")
    
    def _setup_block_pooling(self, config: Dict[str, Any], dropout_rate: float, pooling_alpha: float):
        """设置块池化相关层"""
        pooling_config = {
            "type": "self_attention",
            "num_heads": self.phenotype_dim,  # 表型数作为头数
            "head_dims": self.num_cnn_features_per_phenotype,  # 每个表型的特征数
            "dropout_rate": dropout_rate,
            "alpha": pooling_alpha
        }
        self.block_pooling = create_pooling_from_config(pooling_config)
        self.ln_before_pooling = nn.LayerNorm(self.num_cnn_features_per_phenotype)
    
    def _setup_aux_networks(self, aux_ffn_expansion_factor: float):
        """设置辅助损失网络"""
        aux_hidden_dim = self.num_cnn_features_per_phenotype * aux_ffn_expansion_factor
        
        if self.phenotype_dim == 1:
            self.aux_loss_ffn = nn.Sequential(
                nn.Linear(self.num_cnn_features_per_phenotype, aux_hidden_dim),
                nn.Tanh(),
                nn.Linear(aux_hidden_dim, 1)
            )
        else:
            self.aux_loss_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.num_cnn_features_per_phenotype, aux_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(aux_hidden_dim, 1)
                ) for _ in range(self.phenotype_dim)
            ])
    
    def _setup_positional_encoding(self, config: Dict[str, Any]):
        """设置位置编码"""
        self.pos_encoder = None
        if self.position_encoding:
            max_blocks_for_pos_enc = config.get("max_pos_encoding_blocks", 2048)
            self.pos_encoder = LearnablePositionalEncoding(
                self.num_cnn_features_total, max_blocks_for_pos_enc
            )
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """获取激活函数"""
        act_lower = activation_name.lower()
        if act_lower == "relu": return nn.ReLU()
        elif act_lower == "gelu": return nn.GELU()
        elif act_lower == "silu" or act_lower == "swish": return nn.SiLU()
        elif act_lower == "tanh": return nn.Tanh()
        else:
            warnings.warn(f"Unsupported activation: {activation_name} in EmbeddingLayer_onlySNP. Using GELU.")
            return nn.GELU()
    
    def _extract_cnn_features(self, x: Tensor) -> Tensor:
        """提取CNN特征
        
        Args:
            x: 输入张量 [B, Seq, D]
            
        Returns:
            CNN特征 [B, E*C, Seq]
        """
        # 准备CNN输入 [B, D, Seq]
        cnn_input = x.permute(0, 2, 1)
        
        # 对每个表型提取特征
        phenotype_outputs = []
        for p_idx in range(self.phenotype_dim):
            # 使用模块化的特征提取器
            # 每个特征提取器输出 [B, C, Seq] 形状的特征图
            extractor = self.feature_extractors[p_idx]
            phenotype_feature = extractor(cnn_input)
            
            # 验证输出形状
            expected_channels = self.num_cnn_features_per_phenotype
            if phenotype_feature.size(1) != expected_channels:
                raise ValueError(
                    f"表型 {p_idx} 的CNN特征维度错误。"
                    f"预期 {expected_channels} 通道，但得到 {phenotype_feature.size(1)}。"
                    f"请检查CNNFeatureExtractor配置。"
                )
            
            phenotype_outputs.append(phenotype_feature)
        
        # 拼接所有表型特征 [B, E*C, Seq]
        # 其中每个表型贡献 C 个通道，总共 E 个表型
        return torch.cat(phenotype_outputs, dim=1)    
    
    def _process_no_pooling(self, cnn_output: Tensor, E: int, C: int) -> Tuple[Tensor, int]:
        """处理无需池化的情况 (L=1)
        
        Args:
            cnn_output: CNN输出 [B, E*C, Seq]
            E: 表型数
            C: 每个表型的特征数
            
        Returns:
            - 归一化特征 [B*Seq*E, C]
            - 序列长度/块数量
        """
        # 获取序列长度
        seq_len = cnn_output.size(2)
        
        # 重排为 [B, Seq, E, C]
        cnn_output_permuted = rearrange(cnn_output, 'b (e c) s -> b s e c', e=E, c=C)
        
        # 展平为 [B*Seq*E, C] 用于层归一化
        features_for_ln = rearrange(cnn_output_permuted, 'b s e c -> (b s e) c')
        
        # 应用层归一化
        normed_features = self.ln_before_aux_ffn_no_pool(features_for_ln)
        
        return normed_features, seq_len
    
    def _process_with_pooling(self, cnn_output: Tensor, batch_size: int, E: int, C: int, L: int) -> Tuple[Tensor, int]:
        """处理需要池化的情况 (L>1)
        
        Args:
            cnn_output: CNN输出 [B, E*C, Seq]
            batch_size: 批次大小
            E: 表型数
            C: 每个表型的特征数
            L: 块长度
            
        Returns:
            - 池化后的归一化特征 [B*N*E, C]
            - 块数量
        """
        # 获取序列长度并计算块数
        seq_len = cnn_output.size(2)
        if seq_len % L != 0:
            raise ValueError(f"Input sequence length {seq_len} cannot be divided by Block_length {L}.")
        num_blocks = seq_len // L
        
        # 将CNN输出重排为 [B*N, E*L, C] 用于块池化
        # 每个块内，将L个序列位置的E种表型特征整合
        cnn_output_blocked = rearrange(
            cnn_output, 'b (e c) (n l) -> (b n) (e l) c', 
            e=E, c=C, l=L, n=num_blocks
        )
        
        # 应用层归一化
        normed_pooling_input = self.ln_before_pooling(cnn_output_blocked)
        
        # 应用块池化 - 输出形状 [B*N, E, C]
        pooled_output, _ = self.block_pooling(normed_pooling_input)
        
        # 重排为 [B*N*E, C] 用于后续处理
        normed_features = rearrange(
            pooled_output, '(b n) e c -> (b n e) c', 
            b=batch_size, n=num_blocks
        )
        
        return normed_features, num_blocks
    
    def _compute_aux_loss(self, normed_features: Tensor, batch_size: int, num_blocks: int) -> Tensor:
        """计算辅助损失投影
        
        Args:
            normed_features: 归一化特征 [B*N*E, C]
            batch_size: 批次大小
            num_blocks: 块数或序列长度
            
        Returns:
            辅助损失投影 [B, E]
        """
        E = self.phenotype_dim
        
        if E == 1:
            # 单表型情况
            aux_proj_flat = self.aux_loss_ffn(normed_features)  # [B*N, 1]
            aux_proj_shaped = rearrange(
                aux_proj_flat, '(b k) c_out -> b k 1 c_out', 
                b=batch_size, k=num_blocks
            )
        else:
            # 多表型情况 - 重排为 [B*N, E, C]
            features_grouped = rearrange(
                normed_features, '(bk e) c -> bk e c', 
                e=E
            )
            
            # 对每个表型应用相应的FFN
            aux_projections = []
            for p_idx in range(E):
                features_p = features_grouped[:, p_idx]  # [B*N, C]
                proj_p = self.aux_loss_ffns[p_idx](features_p)  # [B*N, 1]
                aux_projections.append(proj_p)
            
            # 堆叠所有表型的投影 [B*N, E, 1]
            stacked = torch.stack(aux_projections, dim=1)
            
            # 重排为 [B, N, E, 1]
            aux_proj_shaped = rearrange(
                stacked, '(b k) e c_out -> b k e c_out', 
                b=batch_size, k=num_blocks
            )
        
        # 沿块维度平均，并移除尾部的单位维度 [B, E]
        return aux_proj_shaped.mean(dim=1).squeeze(-1)
    
    def _generate_embedding(self, normed_features: Tensor, batch_size: int, num_blocks: int) -> Tensor:
        """生成最终嵌入
        
        Args:
            normed_features: 归一化特征 [B*N*E, C]
            batch_size: 批次大小
            num_blocks: 块数或序列长度
            
        Returns:
            最终嵌入 [B, N, E*C]
        """
        E = self.phenotype_dim
        
        # 重排为 [B, N, E, C]
        embedding_reshaped = rearrange(
            normed_features, '(b k e) c -> b k e c', 
            b=batch_size, k=num_blocks, e=E
        )
        
        # 合并表型和特征维度 [B, N, E*C]
        embedding = rearrange(embedding_reshaped, 'b k e c -> b k (e c)')
        
        # 应用dropout
        embedding = self.dropout(embedding)
        
        # 应用位置编码（如果启用）
        if self.position_encoding and self.pos_encoder is not None:
            embedding = self.pos_encoder(embedding)
        
        return embedding

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """前向传播
        
        Args:
            x: 输入特征 [B, Seq, D]
            positions: 可选的位置信息（未使用）
            
        Returns:
            - final_embedding: 最终嵌入 [B, N, E*C]
            - aux_loss_projection: 辅助损失投影 [B, E]
        """
        batch_size, seq_len, input_feature_dim = x.shape
        E = self.phenotype_dim
        C = self.num_cnn_features_per_phenotype
        L = self.Block_length
        
        # === 1. 处理空序列输入 ===
        if seq_len == 0:
            # 返回空嵌入和零投影
            final_embedding = torch.zeros(batch_size, 0, E*C, device=x.device, dtype=x.dtype)
            aux_loss_projection = torch.zeros(batch_size, E, device=x.device, dtype=x.dtype)
            return final_embedding, aux_loss_projection
        
        # === 2. CNN特征提取 ===
        # [B, E*C, Seq]
        cnn_output = self._extract_cnn_features(x)
        
        # === 3. 特征处理（有/无池化） ===
        if L == 1:
            # 无池化处理 - [B*Seq*E, C]
            normed_features, num_blocks_or_seq = self._process_no_pooling(cnn_output, E, C)
        else:
            # 有池化处理 - [B*N*E, C]
            normed_features, num_blocks_or_seq = self._process_with_pooling(cnn_output, batch_size, E, C, L)
        
        # === 4. 计算辅助损失投影 ===
        # [B, E]
        aux_loss_projection = self._compute_aux_loss(normed_features, batch_size, num_blocks_or_seq)
        
        # === 5. 生成最终嵌入 ===
        # [B, N, E*C]
        final_embedding = self._generate_embedding(normed_features, batch_size, num_blocks_or_seq)
        
        return final_embedding, aux_loss_projection
    

class EmbeddingLayer(nn.Module):
    """
    嵌入层 V4: 处理SNP+POS+Chr。
    CNN提取特征 -> (可选)分块+[LN->Pooling] 或 (L=1时) [LN] -> FFN辅助损失投影 + 输出。
    
    优化点:
    1. 模块化设计，将CNN特征提取分离为独立组件
    2. 提高内存效率，减少中间变量
    3. 代码逻辑分段，提高可读性
    4. 明确的形状注释，便于追踪张量维度
    """
    def __init__(self, input_dim: int, config: Dict[str, Any], phenotype_dim: int):
        super().__init__()
        # === 基本参数初始化 ===
        self.phenotype_dim = phenotype_dim  # E
        self.Block_length = config.get("Block_length")
        
        # 验证基础配置
        self._validate_base_config()
        
        # === 染色体嵌入配置 ===
        num_chromosomes = config.get("num_chromosomes")
        chromosome_embedding_dim = config.get("chromosome_embedding_dim")
        self._validate_chromosome_config(num_chromosomes, chromosome_embedding_dim)
        
        self.chromosome_embed = nn.Embedding(num_embeddings=num_chromosomes, embedding_dim=chromosome_embedding_dim)
        snp_dim = 10  # 固定值
        pos_dim = 4   # 固定值
        initial_cnn_input_channels = snp_dim + chromosome_embedding_dim + pos_dim
        
        # === CNN相关参数 ===
        cnn_kernels = config.get("CNN_Kernel", [3, 5, 7])
        cnn_dilations = config.get("CNN_Dilation", [1, 2])
        cnn_layers = config.get("CNN_layers", 1)
        self.num_cnn_features_per_phenotype = len(cnn_kernels) * len(cnn_dilations)  # C
        self.num_cnn_features_total = self.phenotype_dim * self.num_cnn_features_per_phenotype  # E*C
        
        # === 其他配置参数 ===
        activation_name = config.get("activation", "gelu")
        self.activation_fn = self._get_activation(activation_name)
        dropout_rate = config.get("dropout_rate", 0.1)
        self.position_encoding = config.get("position_encoding", False)
        pooling_alpha = config.get("pooling_alpha", 1.5)
        aux_ffn_expansion_factor = config.get("aux_ffn_expansion_factor", 2)
        
        # === 特征提取器初始化（模块化设计） ===
        self.feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(
                initial_cnn_input_channels, 
                cnn_kernels, 
                cnn_dilations,
                self.activation_fn, 
                cnn_layers
            ) for _ in range(phenotype_dim)
        ])
        
        # === 池化和归一化层初始化 ===
        self.block_pooling = None
        self.ln_before_pooling = None
        self.ln_before_aux_ffn_no_pool = None
        
        if self.Block_length > 1:
            self._setup_block_pooling(config, dropout_rate, pooling_alpha)
        else:  # Block_length == 1
            self.ln_before_aux_ffn_no_pool = nn.LayerNorm(self.num_cnn_features_per_phenotype)
        
        # === 辅助损失网络初始化 ===
        self._setup_aux_networks(aux_ffn_expansion_factor)
        
        # === Dropout和位置编码初始化 ===
        self.dropout = nn.Dropout(dropout_rate)
        self._setup_positional_encoding(config)
        
        # === 验证CNN特征维度 ===
        for i, extractor in enumerate(self.feature_extractors):
            if extractor.output_channels != self.num_cnn_features_per_phenotype:
                raise ValueError(
                    f"表型 {i} 的特征提取器输出通道数 ({extractor.output_channels}) "
                    f"与预期 ({self.num_cnn_features_per_phenotype}) 不匹配"
                )
    
    def _validate_base_config(self):
        """验证基础配置参数合法性"""
        if self.Block_length is None:
            raise ValueError("Embedding config requires 'Block_length'.")
        if not isinstance(self.Block_length, int) or self.Block_length <= 0:
            raise ValueError("'Block_length' must be a positive integer.")
    
    def _validate_chromosome_config(self, num_chromosomes: Optional[int], chromosome_embedding_dim: Optional[int]):
        """验证染色体相关配置"""
        if num_chromosomes is None:
            raise ValueError("Embedding config requires 'num_chromosomes'.")
        if chromosome_embedding_dim is None:
            raise ValueError("Embedding config requires 'chromosome_embedding_dim'.")
    
    def _setup_block_pooling(self, config: Dict[str, Any], dropout_rate: float, pooling_alpha: float):
        """设置块池化相关层"""
        pooling_config = {
            "type": "self_attention",
            "num_heads": self.phenotype_dim,  # 表型数作为头数
            "head_dims": self.num_cnn_features_per_phenotype,  # 每个表型的特征数
            "dropout_rate": dropout_rate,
            "alpha": pooling_alpha
        }
        self.block_pooling = create_pooling_from_config(pooling_config)
        self.ln_before_pooling = nn.LayerNorm(self.num_cnn_features_per_phenotype)
    
    def _setup_aux_networks(self, aux_ffn_expansion_factor: float):
        """设置辅助损失网络"""
        aux_hidden_dim = self.num_cnn_features_per_phenotype * aux_ffn_expansion_factor
        
        if self.phenotype_dim == 1:
            self.aux_loss_ffn = nn.Sequential(
                nn.Linear(self.num_cnn_features_per_phenotype, aux_hidden_dim),
                nn.Tanh(),
                nn.Linear(aux_hidden_dim, 1)
            )
        else:
            self.aux_loss_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.num_cnn_features_per_phenotype, aux_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(aux_hidden_dim, 1)
                ) for _ in range(self.phenotype_dim)
            ])
    
    def _setup_positional_encoding(self, config: Dict[str, Any]):
        """设置位置编码"""
        self.pos_encoder = None
        if self.position_encoding:
            max_blocks_for_pos_enc = config.get("max_pos_encoding_blocks", 2048)
            self.pos_encoder = LearnablePositionalEncoding(
                self.num_cnn_features_total, max_blocks_for_pos_enc
            )
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """获取激活函数"""
        act_lower = activation_name.lower()
        if act_lower == "relu": return nn.ReLU()
        elif act_lower == "gelu": return nn.GELU()
        elif act_lower == "silu" or act_lower == "swish": return nn.SiLU()
        elif act_lower == "tanh": return nn.Tanh()
        else:
            warnings.warn(f"Unsupported activation: {activation_name} in EmbeddingLayer. Using GELU.")
            return nn.GELU()
    
    def _prepare_input_features(self, x: Tensor) -> Tensor:
        """准备输入特征
        
        Args:
            x: 输入张量 [B, Seq, D]
            
        Returns:
            预处理后的特征 [B, D_prep, Seq]
        """
        # 提取 SNP、染色体和位置特征
        snp_features = x[..., :10]
        chromosome_ids = x[..., 10].long()
        other_pos_features = x[..., 11:]
        
        # 染色体ID校正（从1开始变为从0开始）并嵌入
        chromosome_ids_zero_based = torch.clamp(
            chromosome_ids - 1, min=0, max=self.chromosome_embed.num_embeddings - 1
        )
        chrom_embedded = self.chromosome_embed(chromosome_ids_zero_based)
        
        # 合并所有特征
        combined_features = torch.cat([snp_features, chrom_embedded, other_pos_features], dim=-1)
        
        # 转置为卷积层输入格式 [B, D, Seq]
        return combined_features.permute(0, 2, 1)
    
    def _extract_cnn_features(self, cnn_input: Tensor) -> Tensor:
        """提取CNN特征
        
        Args:
            cnn_input: 预处理后的输入 [B, D_prep, Seq]
            
        Returns:
            CNN特征 [B, E*C, Seq]
        """
        # 对每个表型提取特征
        phenotype_outputs = []
        for p_idx in range(self.phenotype_dim):
            # 使用模块化的特征提取器
            # 每个特征提取器输出 [B, C, Seq] 形状的特征图
            extractor = self.feature_extractors[p_idx]
            phenotype_feature = extractor(cnn_input)
            
            # 验证输出形状
            expected_channels = self.num_cnn_features_per_phenotype
            if phenotype_feature.size(1) != expected_channels:
                raise ValueError(
                    f"表型 {p_idx} 的CNN特征维度错误。"
                    f"预期 {expected_channels} 通道，但得到 {phenotype_feature.size(1)}。"
                    f"请检查CNNFeatureExtractor配置。"
                )
            
            phenotype_outputs.append(phenotype_feature)
        
        # 拼接所有表型特征 [B, E*C, Seq]
        # 其中每个表型贡献 C 个通道，总共 E 个表型
        return torch.cat(phenotype_outputs, dim=1)
    
    def _process_no_pooling(self, cnn_output: Tensor, E: int, C: int) -> Tuple[Tensor, int]:
        """处理无需池化的情况 (L=1)
        
        Args:
            cnn_output: CNN输出 [B, E*C, Seq]
            E: 表型数
            C: 每个表型的特征数
            
        Returns:
            - 归一化特征 [B*Seq*E, C]
            - 序列长度/块数量
        """
        # 获取序列长度
        seq_len = cnn_output.size(2)
        
        # 重排为 [B, Seq, E, C]
        cnn_output_permuted = rearrange(cnn_output, 'b (e c) s -> b s e c', e=E, c=C)
        
        # 展平为 [B*Seq*E, C] 用于层归一化
        features_for_ln = rearrange(cnn_output_permuted, 'b s e c -> (b s e) c')
        
        # 应用层归一化
        normed_features = self.ln_before_aux_ffn_no_pool(features_for_ln)
        
        return normed_features, seq_len
    
    def _process_with_pooling(self, cnn_output: Tensor, batch_size: int, E: int, C: int, L: int) -> Tuple[Tensor, int]:
        """处理需要池化的情况 (L>1)
        
        Args:
            cnn_output: CNN输出 [B, E*C, Seq]
            batch_size: 批次大小
            E: 表型数
            C: 每个表型的特征数
            L: 块长度
            
        Returns:
            - 池化后的归一化特征 [B*N*E, C]
            - 块数量
        """
        # 获取序列长度并计算块数
        seq_len = cnn_output.size(2)
        if seq_len % L != 0:
            raise ValueError(f"Input sequence length {seq_len} cannot be divided by Block_length {L}.")
        num_blocks = seq_len // L
        
        # 将CNN输出重排为 [B*N, E*L, C] 用于块池化
        # 每个块内，将L个序列位置的E种表型特征整合
        cnn_output_blocked = rearrange(
            cnn_output, 'b (e c) (n l) -> (b n) (e l) c', 
            e=E, c=C, l=L, n=num_blocks
        )
        
        # 应用层归一化
        normed_pooling_input = self.ln_before_pooling(cnn_output_blocked)
        
        # 应用块池化 - 输出形状 [B*N, E, C]
        pooled_output, _ = self.block_pooling(normed_pooling_input)
        
        # 重排为 [B*N*E, C] 用于后续处理
        normed_features = rearrange(
            pooled_output, '(b n) e c -> (b n e) c', 
            b=batch_size, n=num_blocks
        )
        
        return normed_features, num_blocks
    
    def _compute_aux_loss(self, normed_features: Tensor, batch_size: int, num_blocks: int) -> Tensor:
        """计算辅助损失投影
        
        Args:
            normed_features: 归一化特征 [B*N*E, C]
            batch_size: 批次大小
            num_blocks: 块数或序列长度
            
        Returns:
            辅助损失投影 [B, E]
        """
        E = self.phenotype_dim
        
        if E == 1:
            # 单表型情况
            aux_proj_flat = self.aux_loss_ffn(normed_features)  # [B*N, 1]
            aux_proj_shaped = rearrange(
                aux_proj_flat, '(b k) c_out -> b k 1 c_out', 
                b=batch_size, k=num_blocks
            )
        else:
            # 多表型情况 - 重排为 [B*N, E, C]
            features_grouped = rearrange(
                normed_features, '(bk e) c -> bk e c', 
                e=E
            )
            
            # 对每个表型应用相应的FFN
            aux_projections = []
            for p_idx in range(E):
                features_p = features_grouped[:, p_idx]  # [B*N, C]
                proj_p = self.aux_loss_ffns[p_idx](features_p)  # [B*N, 1]
                aux_projections.append(proj_p)
            
            # 堆叠所有表型的投影 [B*N, E, 1]
            stacked = torch.stack(aux_projections, dim=1)
            
            # 重排为 [B, N, E, 1]
            aux_proj_shaped = rearrange(
                stacked, '(b k) e c_out -> b k e c_out', 
                b=batch_size, k=num_blocks
            )
        
        # 沿块维度平均，并移除尾部的单位维度 [B, E]
        return aux_proj_shaped.mean(dim=1).squeeze(-1)
    
    def _generate_embedding(self, normed_features: Tensor, batch_size: int, num_blocks: int) -> Tensor:
        """生成最终嵌入
        
        Args:
            normed_features: 归一化特征 [B*N*E, C]
            batch_size: 批次大小
            num_blocks: 块数或序列长度
            
        Returns:
            最终嵌入 [B, N, E*C]
        """
        E = self.phenotype_dim
        
        # 重排为 [B, N, E, C]
        embedding_reshaped = rearrange(
            normed_features, '(b k e) c -> b k e c', 
            b=batch_size, k=num_blocks, e=E
        )
        
        # 合并表型和特征维度 [B, N, E*C]
        embedding = rearrange(embedding_reshaped, 'b k e c -> b k (e c)')
        
        # 应用dropout
        embedding = self.dropout(embedding)
        
        # 应用位置编码（如果启用）
        if self.position_encoding and self.pos_encoder is not None:
            embedding = self.pos_encoder(embedding)
        
        return embedding

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """前向传播
        
        Args:
            x: 输入特征 [B, Seq, D]
            positions: 可选的位置信息（未使用）
            
        Returns:
            - final_embedding: 最终嵌入 [B, N, E*C]
            - aux_loss_projection: 辅助损失投影 [B, E]
        """
        batch_size, seq_len, _ = x.shape
        E = self.phenotype_dim
        C = self.num_cnn_features_per_phenotype
        L = self.Block_length
        
        # === 1. 处理空序列输入 ===
        if seq_len == 0:
            # 返回空嵌入和零投影
            final_embedding = torch.zeros(batch_size, 0, E*C, device=x.device, dtype=x.dtype)
            aux_loss_projection = torch.zeros(batch_size, E, device=x.device, dtype=x.dtype)
            return final_embedding, aux_loss_projection
        
        # === 2. 特征预处理 ===
        cnn_input = self._prepare_input_features(x)
        
        # === 3. CNN特征提取 ===
        cnn_output = self._extract_cnn_features(cnn_input)
        
        # === 4. 特征处理（有/无池化） ===
        if L == 1:
            # 无池化处理 - [B*Seq*E, C]
            normed_features, num_blocks_or_seq = self._process_no_pooling(cnn_output, E, C)
        else:
            # 有池化处理 - [B*N*E, C]
            normed_features, num_blocks_or_seq = self._process_with_pooling(cnn_output, batch_size, E, C, L)
        
        # === 5. 计算辅助损失投影 ===
        # [B, E]
        aux_loss_projection = self._compute_aux_loss(normed_features, batch_size, num_blocks_or_seq)
        
        # === 6. 生成最终嵌入 ===
        # [B, N, E*C]
        final_embedding = self._generate_embedding(normed_features, batch_size, num_blocks_or_seq)
        
        return final_embedding, aux_loss_projection



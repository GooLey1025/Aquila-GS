# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
"""
V2 architectural blocks for model-explore experiments.

Tensor contract matches existing trunk blocks: (B, L, C) in / out.
Optional mask: (B, L). Factories accept **kwargs so YAML extras do not crash.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers


def _as_dilation_tuple(dilations: Union[Sequence[int], int]) -> Tuple[int, ...]:
    if isinstance(dilations, int):
        return (dilations,)
    return tuple(int(d) for d in dilations)


############################################################
# Dilated Conv Stack
############################################################


class DilatedConvStack(nn.Module):
    """Parallel residual dilated 1D convolutions for multi-scale LD patterns.

    Each branch applies a dilated depthwise-style conv (implemented as grouped
    Conv1d with groups=d_model) then a pointwise projection. Branch outputs are
    concatenated and projected back to d_model, then added as a residual.
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 5,
        dilations: Union[Sequence[int], int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.dilations = _as_dilation_tuple(dilations)
        self.activation = activation

        self.branches = nn.ModuleList()
        for rate in self.dilations:
            padding = (kernel_size - 1) // 2 * rate
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=rate,
                        groups=d_model,
                        bias=False,
                    ),
                    nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                )
            )

        n_branches = len(self.dilations)
        self.proj = nn.Linear(d_model * n_branches, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, L, C)
        h = self.norm(x)
        h_t = h.transpose(1, 2)  # (B, C, L)
        outs = []
        for branch in self.branches:
            y = branch(h_t).transpose(1, 2)  # (B, L, C)
            outs.append(layers.activate(y, self.activation))
        fused = torch.cat(outs, dim=-1)
        out = self.dropout(self.proj(fused))
        return x + out


def dilated_conv_stack(
    d_model: int = 256,
    kernel_size: int = 5,
    dilations=(1, 2, 4, 8),
    dropout: float = 0.1,
    activation: str = "gelu",
    **kwargs,
):
    return DilatedConvStack(
        d_model=d_model,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        activation=activation,
    )


############################################################
# Squeeze-Excitation Feature Gate
############################################################


class SEFeatureGate(nn.Module):
    """Squeeze-Excitation over the feature/channel dimension.

    Pools across sequence length, predicts per-channel gates, and rescales
    features. Cheap recalibration before attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        reduction: int = 16,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        hidden = max(d_model // int(reduction), 8)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, L, C)
        if mask is not None:
            w = mask.unsqueeze(-1).float()
            denom = w.sum(dim=1).clamp_min(1.0)
            pooled = (x * w).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        gate = torch.sigmoid(self.fc2(F.gelu(self.fc1(pooled))))
        gate = self.dropout(gate).unsqueeze(1)
        return x * gate


def se_feature_gate(
    d_model: int = 256,
    reduction: int = 16,
    dropout: float = 0.1,
    **kwargs,
):
    return SEFeatureGate(d_model=d_model, reduction=reduction, dropout=dropout)


############################################################
# ConvNeXt-style Stack
############################################################


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt-style residual block on (B, L, C)."""

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        hidden = d_model * int(expansion)
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, hidden)
        self.pw2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        residual = x
        h = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        h = self.pw2(self.dropout(F.gelu(self.pw1(h))))
        return residual + self.dropout(h)


class ConvNeXtStack(nn.Module):
    """Stack of ConvNeXt-style residual blocks for local refinement."""

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 7,
        expansion: int = 4,
        repeat: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(int(repeat))
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for block in self.blocks:
            x = block(x)
        return x


def convnext_stack(
    d_model: int = 256,
    kernel_size: int = 7,
    expansion: int = 4,
    repeat: int = 2,
    dropout: float = 0.1,
    **kwargs,
):
    return ConvNeXtStack(
        d_model=d_model,
        kernel_size=kernel_size,
        expansion=expansion,
        repeat=repeat,
        dropout=dropout,
    )


############################################################
# Dual-Axis Attention
############################################################


class DualAxisAttention(nn.Module):
    """Sequence MHA → channel-transpose MHA → FFN as one interaction unit."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model

        self.seq_norm = nn.LayerNorm(d_model)
        self.seq_attn = layers.MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.seq_drop = nn.Dropout(dropout)

        self.chan_norm = nn.LayerNorm(d_model)
        self.chan_attn = layers.MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.chan_drop = nn.Dropout(dropout)
        self.seq_len_proj_down: Optional[nn.Linear] = None
        self.seq_len_proj_up: Optional[nn.Linear] = None

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.ffn_drop = nn.Dropout(dropout)

    def _channel_attn(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        h = x.transpose(1, 2)  # (B, C, L)

        if seq_len != d_model:
            if (
                self.seq_len_proj_down is None
                or self.seq_len_proj_down.in_features != seq_len
            ):
                self.seq_len_proj_down = nn.Linear(seq_len, d_model).to(x.device)
                self.seq_len_proj_up = nn.Linear(d_model, seq_len).to(x.device)
            h = self.seq_len_proj_down(h)  # (B, C, D)

        h_norm = self.chan_norm(h)
        h = h + self.chan_drop(self.chan_attn(h_norm, mask=None))

        if seq_len != d_model:
            h = self.seq_len_proj_up(h)

        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.seq_drop(self.seq_attn(self.seq_norm(x), mask=mask))
        x = x + self._channel_attn(x)
        x = x + self.ffn_drop(self.ffn(self.ffn_norm(x)))
        return x


def dual_axis_attention(
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 256,
    dropout: float = 0.1,
    activation: str = "gelu",
    **kwargs,
):
    return DualAxisAttention(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
    )


############################################################
# Local-Global Mixer
############################################################


class LocalGlobalMixer(nn.Module):
    """Parallel local conv branch + global self-attention; gated residual fuse."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 256,
        kernel_size: int = 7,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.local_norm = nn.LayerNorm(d_model)
        self.local_dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
            bias=True,
        )
        self.local_pw = nn.Linear(d_model, d_model)
        self.local_drop = nn.Dropout(dropout)

        self.global_norm = nn.LayerNorm(d_model)
        self.global_attn = layers.MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.global_ffn = layers.FeedForward(d_model, d_ff, dropout, activation)
        self.global_drop = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Local branch
        local = self.local_norm(x)
        local = self.local_dw(local.transpose(1, 2)).transpose(1, 2)
        local = self.local_drop(
            self.local_pw(layers.activate(local, self.activation))
        )

        # Global branch (pre-norm transformer style)
        g = self.global_norm(x)
        g = x + self.global_drop(self.global_attn(g, mask=mask))
        global_out = g + self.global_drop(self.global_ffn(self.global_norm(g)))

        gate = self.gate(torch.cat([local, global_out], dim=-1))
        fused = gate * local + (1.0 - gate) * global_out
        return x + self.out_drop(fused)


def local_global_mixer(
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 256,
    kernel_size: int = 7,
    dropout: float = 0.1,
    activation: str = "gelu",
    **kwargs,
):
    return LocalGlobalMixer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        kernel_size=kernel_size,
        dropout=dropout,
        activation=activation,
    )


############################################################
# Gated Channel Mixer (GEGLU-style)
############################################################


class GatedChannelMixer(nn.Module):
    """GEGLU-style channel MLP with residual connection."""

    def __init__(
        self,
        d_model: int = 256,
        expansion: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        hidden = d_model * int(expansion)
        self.norm = nn.LayerNorm(d_model)
        self.fc_in = nn.Linear(d_model, hidden * 2)
        self.fc_out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.norm(x)
        u, v = self.fc_in(h).chunk(2, dim=-1)
        h = F.gelu(u) * v
        h = self.dropout(self.fc_out(h))
        return x + h


def gated_channel_mixer(
    d_model: int = 256,
    expansion: int = 4,
    dropout: float = 0.1,
    **kwargs,
):
    return GatedChannelMixer(
        d_model=d_model,
        expansion=expansion,
        dropout=dropout,
    )


############################################################
# MeNet-inspired local / global genomic fusion
############################################################


class MeNetGenomicFusion(nn.Module):
    """Bidirectionally fuse local variant effects and global genotype context.

    The local path uses multi-dilation depthwise convolutions as a proxy for
    MeNet's variant-effect encoder. The global path summarizes fixed genomic
    windows, models interactions among those window tokens, and broadcasts the
    resulting context back to loci as a RepGeno-like representation. Two
    residual flow fields cross-condition both paths before gated fusion.

    This block does not claim to reproduce MeNet's separately pretrained,
    trait-specific relatedness input; it adapts that dual-scale idea to the
    single genotype tensor available in Aquila.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 512,
        kernel_size: int = 7,
        dilations: Union[Sequence[int], int] = (1, 2, 4),
        window_size: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.window_size = int(window_size)
        self.dilations = _as_dilation_tuple(dilations)
        self.activation = activation
        padding_base = (int(kernel_size) - 1) // 2

        self.local_norm = nn.LayerNorm(self.d_model)
        self.local_paths = nn.ModuleList(
            [
                nn.Conv1d(
                    self.d_model,
                    self.d_model,
                    kernel_size=int(kernel_size),
                    padding=padding_base * dilation,
                    dilation=dilation,
                    groups=self.d_model,
                    bias=False,
                )
                for dilation in self.dilations
            ]
        )
        self.local_merge = nn.Linear(
            self.d_model * len(self.dilations), self.d_model
        )

        self.window_norm = nn.LayerNorm(self.d_model)
        self.window_attention = nn.MultiheadAttention(
            self.d_model,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.window_ffn_norm = nn.LayerNorm(self.d_model)
        self.window_ffn = layers.FeedForward(
            self.d_model, int(d_ff), float(dropout), activation
        )

        flow_hidden = max(self.d_model * 2, int(d_ff))
        self.local_flow = nn.Sequential(
            nn.Linear(self.d_model * 2, flow_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(flow_hidden, self.d_model),
        )
        self.global_flow = nn.Sequential(
            nn.Linear(self.d_model * 2, flow_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(flow_hidden, self.d_model),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(self.d_model)
        self.out_ffn = layers.FeedForward(
            self.d_model, int(d_ff), float(dropout), activation
        )
        self.dropout = nn.Dropout(float(dropout))

    def _window_context(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        window_size = min(self.window_size, seq_len)
        pad_len = (-seq_len) % window_size
        if pad_len:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        num_windows = x_padded.size(1) // window_size
        windows = x_padded.view(
            batch_size, num_windows, window_size, channels
        )
        tokens = windows.mean(dim=2)
        tokens_norm = self.window_norm(tokens)
        attended, _ = self.window_attention(
            tokens_norm, tokens_norm, tokens_norm, need_weights=False
        )
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(
            self.window_ffn(self.window_ffn_norm(tokens))
        )
        context = tokens.repeat_interleave(window_size, dim=1)
        return context[:, :seq_len]

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        local_input = self.local_norm(x).transpose(1, 2)
        local_parts = [
            layers.activate(
                path(local_input).transpose(1, 2), self.activation
            )
            for path in self.local_paths
        ]
        local = self.local_merge(torch.cat(local_parts, dim=-1))
        global_context = self._window_context(x)

        joint = torch.cat([local, global_context], dim=-1)
        local_fused = local + self.dropout(self.local_flow(joint))
        global_fused = global_context + self.dropout(self.global_flow(joint))
        gate = self.fusion_gate(
            torch.cat([local_fused, global_fused], dim=-1)
        )
        fused = gate * local_fused + (1.0 - gate) * global_fused
        x = x + self.dropout(fused)
        return x + self.dropout(self.out_ffn(self.out_norm(x)))


def menet_genomic_fusion(
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 512,
    kernel_size: int = 7,
    dilations=(1, 2, 4),
    window_size: int = 64,
    dropout: float = 0.1,
    activation: str = "gelu",
    **kwargs,
):
    return MeNetGenomicFusion(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        kernel_size=kernel_size,
        dilations=dilations,
        window_size=window_size,
        dropout=dropout,
        activation=activation,
    )


############################################################
# End-to-end prototype relation fusion
############################################################


class PrototypeRelationFusion(nn.Module):
    """Fuse locus features with a learned fixed-width relation profile.

    MeNet relates each individual to every cohort member. This block uses a
    small learned prototype bank instead, preserving independent inference and
    avoiding an input width tied to cohort size.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_prototypes: int = 16,
        d_ff: int = 512,
        temperature: float = 0.2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if num_prototypes < 2:
            raise ValueError("num_prototypes must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = float(temperature)
        self.summary_norm = nn.LayerNorm(d_model)
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, d_model))
        nn.init.normal_(self.prototypes, mean=0.0, std=d_model ** -0.5)

        self.profile_projection = nn.Sequential(
            nn.Linear(num_prototypes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        flow_hidden = max(2 * d_model, int(d_ff))
        self.local_flow = nn.Sequential(
            nn.Linear(2 * d_model, flow_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flow_hidden, d_model),
        )
        self.relation_flow = nn.Sequential(
            nn.Linear(2 * d_model, flow_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flow_hidden, d_model),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_ffn = layers.FeedForward(
            d_model, int(d_ff), float(dropout), "gelu"
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _masked_mean(
        x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denominator

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        summary = self.summary_norm(self._masked_mean(x, mask))
        normalized_summary = F.normalize(summary, p=2, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        relation_logits = (
            normalized_summary @ normalized_prototypes.transpose(0, 1)
        ) / self.temperature
        relation_weights = relation_logits.softmax(dim=-1)

        prototype_context = relation_weights @ self.prototypes
        relation_summary = (
            prototype_context + self.profile_projection(relation_logits)
        )
        relation = relation_summary.unsqueeze(1).expand(-1, x.size(1), -1)
        local = self.local_projection(x)

        joint = torch.cat([local, relation], dim=-1)
        local_fused = local + self.dropout(self.local_flow(joint))
        relation_fused = relation + self.dropout(self.relation_flow(joint))
        gate = self.fusion_gate(
            torch.cat([local_fused, relation_fused], dim=-1)
        )
        fused = gate * local_fused + (1.0 - gate) * relation_fused
        x = x + self.dropout(fused)
        return x + self.dropout(self.out_ffn(self.out_norm(x)))


def prototype_relation_fusion(
    d_model: int = 256,
    num_prototypes: int = 16,
    d_ff: int = 512,
    temperature: float = 0.2,
    dropout: float = 0.1,
    **kwargs,
):
    return PrototypeRelationFusion(
        d_model=d_model,
        num_prototypes=num_prototypes,
        d_ff=d_ff,
        temperature=temperature,
        dropout=dropout,
    )


############################################################
# Literature ops: RMSBatchNorm + StandardizedConv1D
############################################################


class RMSBatchNorm1d(nn.Module):
    """Root-mean-square batch normalization for (B, C, L) tensors.

    Same affine scale/offset as BatchNorm, but without mean centering.
    During training, normalizes by per-channel batch variance over (B, L) and
    maintains an EMA of that variance (decay=0.9 → momentum=0.1) for inference.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        if affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_var", torch.ones(self.num_features))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        if self.training:
            # Per-channel variance over batch and length (no mean subtraction).
            var = x.pow(2).mean(dim=(0, 2))  # (C,)
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_var.mul_(1.0 - self.momentum).add_(
                    var * self.momentum
                )
            inv_std = torch.rsqrt(var + self.eps).view(1, -1, 1)
            x = x * inv_std
        else:
            inv_std = torch.rsqrt(self.running_var + self.eps).view(1, -1, 1)
            x = x * inv_std

        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class StandardizedConv1d(nn.Conv1d):
    """1D convolution with scaled weight standardization (Brock et al.).

    At each forward, weights are re-parameterized as:
      WS(w) = (w - mean) / sqrt(var + eps)
      scaled = gain / sqrt(fan_in) * WS(w)
    Stabilizes activation magnitudes vs plain Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        gain: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.gain = float(gain)
        self.eps = float(eps)

    def standardized_weight(self) -> torch.Tensor:
        w = self.weight
        # Standardize over (in_channels / groups, kernel)
        mean = w.mean(dim=(1, 2), keepdim=True)
        var = w.var(dim=(1, 2), keepdim=True, unbiased=False)
        fan_in = float(w[0].numel())
        scale = self.gain * (fan_in ** -0.5)
        return (w - mean) * torch.rsqrt(var + self.eps) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.standardized_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _same_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    if stride == 1:
        return (kernel_size - 1) // 2 * dilation
    # Approximate "same" for stride>1 (matches existing conv_block habit).
    return (kernel_size - 1) // 2 * dilation


def _update_mask_after_stride(
    mask: Optional[torch.Tensor],
    new_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.size(1) == new_seq_len:
        return mask
    mask_float = mask.float().unsqueeze(1)
    mask_pooled = F.max_pool1d(
        mask_float,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
    )
    mask = mask_pooled.squeeze(1).bool()
    if mask.size(1) > new_seq_len:
        mask = mask[:, :new_seq_len]
    elif mask.size(1) < new_seq_len:
        mask = F.pad(mask, (0, new_seq_len - mask.size(1)), value=True)
    return mask


class StdConvUnit(nn.Module):
    """Literature unit: RMSBatchNorm → GeLU → StandardizedConv1D.

    Operates on (B, L, C). Returns (x, mask).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
        gain: float = 1.0,
        bn_momentum: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.padding = _same_padding(kernel_size, stride=stride)
        self.residual = bool(residual)

        self.norm = RMSBatchNorm1d(in_channels, momentum=bn_momentum)
        self.conv = StandardizedConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
            gain=gain,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if residual and (in_channels != out_channels or stride > 1):
            self.residual_proj = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, L, C)
        identity = x.transpose(1, 2) if self.residual else None
        h = x.transpose(1, 2)  # (B, C, L)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.conv(h)

        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            if identity.size(2) != h.size(2):
                # Align lengths if "same" padding is imperfect.
                L = h.size(2)
                if identity.size(2) > L:
                    identity = identity[:, :, :L]
                else:
                    identity = F.pad(
                        identity, (0, L - identity.size(2)), mode="replicate"
                    )
            h = h + identity

        h = self.dropout(h)
        out = h.transpose(1, 2)
        mask = _update_mask_after_stride(
            mask, out.size(1), self.kernel_size, self.stride, self.padding
        )
        return out, mask


class StdDownConvBlock(nn.Module):
    """Feature extract (stride=1) + downsample (stride=2) using StdConvUnit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
        residual: bool = True,
        gain: float = 1.0,
        bn_momentum: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feat = StdConvUnit(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            residual=residual,
            gain=gain,
            bn_momentum=bn_momentum,
        )
        self.down = StdConvUnit(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout=dropout,
            residual=residual,
            gain=gain,
            bn_momentum=bn_momentum,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x, mask = self.feat(x, mask)
        x, mask = self.down(x, mask)
        return x, mask


class ConvNeXtDownBlock(nn.Module):
    """ConvNeXt-style refine (stride=1) + stride-2 downsample.

    Optional literature ops: if use_stdconv=True, downsample uses StdConvUnit
    and the depthwise path uses StandardizedConv1d + RMSBatchNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
        use_stdconv: bool = False,
        gain: float = 1.0,
        bn_momentum: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.use_stdconv = bool(use_stdconv)
        pad = (kernel_size - 1) // 2
        hidden = out_channels * int(expansion)

        # Channel align if needed before ConvNeXt body.
        if in_channels != out_channels:
            self.in_proj = nn.Linear(in_channels, out_channels)
        else:
            self.in_proj = None

        if use_stdconv:
            self.dw = StandardizedConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=pad,
                groups=out_channels,
                bias=True,
                gain=gain,
            )
            self.dw_norm = RMSBatchNorm1d(out_channels, momentum=bn_momentum)
        else:
            self.dw = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=pad,
                groups=out_channels,
                bias=True,
            )
            self.dw_norm = nn.GroupNorm(1, out_channels)

        self.pw1 = nn.Linear(out_channels, hidden)
        self.pw2 = nn.Linear(hidden, out_channels)
        self.drop = nn.Dropout(dropout)

        if use_stdconv:
            self.down = StdConvUnit(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                dropout=dropout,
                residual=True,
                gain=gain,
                bn_momentum=bn_momentum,
            )
        else:
            self.down = None
            self.down_conv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True,
            )
            self.down_norm = nn.GroupNorm(1, out_channels)
            self.down_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.in_proj is not None:
            x = self.in_proj(x)

        residual = x
        h = x.transpose(1, 2)
        if self.use_stdconv:
            h = self.dw_norm(h)
            h = F.gelu(h)
            h = self.dw(h)
        else:
            h = self.dw(h)
            h = self.dw_norm(h)
        h = h.transpose(1, 2)
        h = self.pw2(self.drop(F.gelu(self.pw1(h))))
        x = residual + self.drop(h)

        if self.down is not None:
            return self.down(x, mask)

        identity = x.transpose(1, 2)
        h = self.down_conv(identity)
        # Align residual length for stride-2.
        if identity.size(2) != h.size(2):
            # Average-pool residual to match.
            identity = F.avg_pool1d(identity, kernel_size=2, stride=2, ceil_mode=True)
            if identity.size(2) > h.size(2):
                identity = identity[:, :, : h.size(2)]
            elif identity.size(2) < h.size(2):
                identity = F.pad(identity, (0, h.size(2) - identity.size(2)))
        h = self.down_drop(F.gelu(self.down_norm(h + identity)))
        out = h.transpose(1, 2)
        mask = _update_mask_after_stride(mask, out.size(1), 5, 2, 2)
        return out, mask


class DilatedDownBlock(nn.Module):
    """Multi-dilation residual refine + stride-2 downsample (adaptive tower stage)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilations: Union[Sequence[int], int] = (1, 2, 4),
        dropout: float = 0.1,
        use_stdconv: bool = False,
        gain: float = 1.0,
        bn_momentum: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.dilations = _as_dilation_tuple(dilations)
        self.use_stdconv = bool(use_stdconv)

        if in_channels != out_channels:
            self.in_proj = nn.Linear(in_channels, out_channels)
        else:
            self.in_proj = None

        if use_stdconv:
            self.pre_norm = RMSBatchNorm1d(out_channels, momentum=bn_momentum)
        else:
            self.pre_norm = nn.GroupNorm(1, out_channels)

        self.branches = nn.ModuleList()
        for rate in self.dilations:
            pad = (kernel_size - 1) // 2 * rate
            if use_stdconv:
                dw = StandardizedConv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=pad,
                    dilation=rate,
                    groups=out_channels,
                    bias=False,
                    gain=gain,
                )
                pw = StandardizedConv1d(
                    out_channels, out_channels, kernel_size=1, bias=True, gain=gain
                )
            else:
                dw = nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=pad,
                    dilation=rate,
                    groups=out_channels,
                    bias=False,
                )
                pw = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
            self.branches.append(nn.Sequential(dw, pw))

        self.fuse = nn.Linear(out_channels * len(self.dilations), out_channels)
        self.drop = nn.Dropout(dropout)

        if use_stdconv:
            self.down = StdConvUnit(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                dropout=dropout,
                residual=True,
                gain=gain,
                bn_momentum=bn_momentum,
            )
            self.down_conv = None
            self.down_norm = None
            self.down_drop = None
        else:
            self.down = None
            self.down_conv = nn.Conv1d(
                out_channels, out_channels, kernel_size=5, stride=2, padding=2
            )
            self.down_norm = nn.GroupNorm(1, out_channels)
            self.down_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.in_proj is not None:
            x = self.in_proj(x)

        h = x.transpose(1, 2)
        h = self.pre_norm(h)
        h = F.gelu(h)
        outs = [branch(h).transpose(1, 2) for branch in self.branches]
        fused = self.drop(self.fuse(torch.cat(outs, dim=-1)))
        x = x + fused

        if self.down is not None:
            return self.down(x, mask)

        identity = x.transpose(1, 2)
        h = self.down_conv(identity)
        identity = F.avg_pool1d(identity, kernel_size=2, stride=2, ceil_mode=True)
        if identity.size(2) > h.size(2):
            identity = identity[:, :, : h.size(2)]
        elif identity.size(2) < h.size(2):
            identity = F.pad(identity, (0, h.size(2) - identity.size(2)))
        h = self.down_drop(F.gelu(self.down_norm(h + identity)))
        out = h.transpose(1, 2)
        mask = _update_mask_after_stride(mask, out.size(1), 5, 2, 2)
        return out, mask


class AdaptiveDownsampleTower(nn.Module):
    """Generic auto-downsample tower (same policy as DownConvTower).

    If repeat is None, applies stage blocks while seq_len > seq_len_threshold.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        repeat: Optional[int] = None,
        seq_len_threshold: int = 1500,
        name: str = "AdaptiveDownsampleTower",
    ):
        super().__init__()
        self.blocks = blocks
        self.repeat = repeat
        self.seq_len_threshold = int(seq_len_threshold)
        self._tower_name = name

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.repeat is not None:
            for i in range(self.repeat):
                x, mask = self.blocks[i](x, mask)
            return x, mask

        current_seq_len = x.size(1)
        block_idx = 0
        while current_seq_len > self.seq_len_threshold and block_idx < len(self.blocks):
            x, mask = self.blocks[block_idx](x, mask)
            current_seq_len = x.size(1)
            block_idx += 1

        if current_seq_len > self.seq_len_threshold:
            import warnings

            warnings.warn(
                f"{self._tower_name}: Sequence length ({current_seq_len}) still exceeds "
                f"threshold ({self.seq_len_threshold}) after using all {len(self.blocks)} "
                f"blocks. Consider a larger repeat."
            )
        return x, mask


def _make_tower_blocks(
    stage_cls,
    in_channels: Optional[int],
    out_channels: int,
    repeat: Optional[int],
    stage_kwargs: dict,
) -> nn.ModuleList:
    num_blocks = int(repeat) if repeat is not None else 20
    blocks = nn.ModuleList()
    current = in_channels if in_channels is not None else out_channels
    for _ in range(num_blocks):
        blocks.append(
            stage_cls(
                in_channels=current,
                out_channels=out_channels,
                **stage_kwargs,
            )
        )
        current = out_channels
    return blocks


def std_down_conv_tower(
    in_channels=None,
    out_channels=256,
    kernel_size=5,
    dropout=0.1,
    residual=True,
    repeat=None,
    seq_len_threshold=1500,
    gain=1.0,
    bn_momentum=0.1,
    **kwargs,
):
    """Adaptive downsample tower with RMSBatchNorm + StandardizedConv1D units."""
    blocks = _make_tower_blocks(
        StdDownConvBlock,
        in_channels,
        out_channels,
        repeat,
        dict(
            kernel_size=kernel_size,
            dropout=dropout,
            residual=residual,
            gain=gain,
            bn_momentum=bn_momentum,
        ),
    )
    return AdaptiveDownsampleTower(
        blocks,
        repeat=repeat,
        seq_len_threshold=seq_len_threshold,
        name="StdDownConvTower",
    )


def convnext_down_conv_tower(
    in_channels=None,
    out_channels=256,
    kernel_size=7,
    expansion=4,
    dropout=0.1,
    repeat=None,
    seq_len_threshold=1500,
    use_stdconv=False,
    gain=1.0,
    bn_momentum=0.1,
    **kwargs,
):
    """Adaptive downsample tower with ConvNeXt-style stages."""
    blocks = _make_tower_blocks(
        ConvNeXtDownBlock,
        in_channels,
        out_channels,
        repeat,
        dict(
            kernel_size=kernel_size,
            expansion=expansion,
            dropout=dropout,
            use_stdconv=use_stdconv,
            gain=gain,
            bn_momentum=bn_momentum,
        ),
    )
    return AdaptiveDownsampleTower(
        blocks,
        repeat=repeat,
        seq_len_threshold=seq_len_threshold,
        name="ConvNeXtDownConvTower",
    )


def dilated_down_conv_tower(
    in_channels=None,
    out_channels=256,
    kernel_size=5,
    dilations=(1, 2, 4),
    dropout=0.1,
    repeat=None,
    seq_len_threshold=1500,
    use_stdconv=False,
    gain=1.0,
    bn_momentum=0.1,
    **kwargs,
):
    """Adaptive downsample tower with multi-dilation stages."""
    blocks = _make_tower_blocks(
        DilatedDownBlock,
        in_channels,
        out_channels,
        repeat,
        dict(
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            use_stdconv=use_stdconv,
            gain=gain,
            bn_momentum=bn_momentum,
        ),
    )
    return AdaptiveDownsampleTower(
        blocks,
        repeat=repeat,
        seq_len_threshold=seq_len_threshold,
        name="DilatedDownConvTower",
    )


def std_conv_block(
    in_channels=None,
    out_channels=256,
    kernel_size=5,
    stride=1,
    dropout=0.1,
    residual=True,
    gain=1.0,
    bn_momentum=0.1,
    **kwargs,
):
    """Single StdConvUnit for optional embedder / non-tower use."""
    if in_channels is None:
        raise ValueError("std_conv_block requires in_channels (no Lazy path).")
    return StdConvUnit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dropout=dropout,
        residual=residual,
        gain=gain,
        bn_momentum=bn_momentum,
    )


############################################################
# Block Dictionary
############################################################


name_func = {
    "dilated_conv_stack": dilated_conv_stack,
    "se_feature_gate": se_feature_gate,
    "convnext_stack": convnext_stack,
    "dual_axis_attention": dual_axis_attention,
    "local_global_mixer": local_global_mixer,
    "gated_channel_mixer": gated_channel_mixer,
    "menet_genomic_fusion": menet_genomic_fusion,
    "prototype_relation_fusion": prototype_relation_fusion,
    # Literature ops + adaptive towers (replace down_conv_tower internals)
    "std_conv_block": std_conv_block,
    "std_down_conv_tower": std_down_conv_tower,
    "convnext_down_conv_tower": convnext_down_conv_tower,
    "dilated_down_conv_tower": dilated_down_conv_tower,
}

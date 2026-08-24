# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Shape and grouping tests for multi-task regression heads."""

from __future__ import annotations

import torch

from aquila.blocks import (
    expert_choice_moe,
    expert_choice_moe_pool,
    family_grouped_regression_head,
    film_regression_head,
    group_traits_by_family,
    mmoe_regression_head,
    per_trait_regression_head,
    shared_stem_family_head,
    shared_stem_private_head,
    trait_query_regression_head,
    transformer,
)
from aquila.varnn import create_model_from_config


def test_group_traits_by_family_uses_prefix() -> None:
    families = group_traits_by_family(["PPNP_LingS16", "PPNP_BLUP", "HD_WenJ15"])
    assert families == {"PPNP": [0, 1], "HD": [2]}


def test_per_trait_and_shared_stem_shapes() -> None:
    x = torch.randn(4, 256)
    per_trait = per_trait_regression_head(in_features=256, num_targets=5, hidden_features=16)
    shared = shared_stem_private_head(
        in_features=256, num_targets=5, stem_features=32, hidden_features=16
    )
    assert per_trait(x).shape == (4, 5)
    assert shared(x).shape == (4, 5)


def test_family_grouped_writes_family_outputs_to_original_order() -> None:
    names = ["PPNP_A", "HD_A", "PPNP_B"]
    head = family_grouped_regression_head(
        in_features=8,
        num_targets=3,
        hidden_features=4,
        task_names=names,
    )
    assert set(head.families) == {"PPNP", "HD"}
    assert head.families["PPNP"] == [0, 2]
    out = head(torch.randn(2, 8))
    assert out.shape == (2, 3)
    stem_family = shared_stem_family_head(
        in_features=8,
        num_targets=3,
        stem_features=4,
        task_names=names,
    )
    assert stem_family(torch.randn(2, 8)).shape == (2, 3)
    assert stem_family.families["PPNP"] == [0, 2]


def test_mmoe_and_film_shapes() -> None:
    x = torch.randn(3, 16)
    mmoe = mmoe_regression_head(
        in_features=16, num_targets=6, num_experts=3, expert_dim=8, tower_hidden=4
    )
    linear = mmoe_regression_head(
        in_features=16, num_targets=6, num_experts=3, expert_dim=8, tower_hidden=None
    )
    static = mmoe_regression_head(
        in_features=16, num_targets=6, num_experts=4, expert_dim=8,
        tower_hidden=None, gate_type='static',
    )
    film = film_regression_head(in_features=16, num_targets=6, hidden_features=8)
    assert mmoe(x).shape == (3, 6)
    assert linear(x).shape == (3, 6)
    assert static(x).shape == (3, 6)
    assert film(x).shape == (3, 6)
    assert len(mmoe.experts) == 3
    assert static.gate_logits.shape == (6, 4)


def test_trait_query_attends_over_sequence() -> None:
    x = torch.randn(2, 12, 16)
    head = trait_query_regression_head(
        d_model=16, num_targets=5, num_heads=4, hidden_features=8
    )
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -3:] = False
    out = head(x, mask=mask)
    assert out.shape == (2, 5)


def test_expert_choice_moe_keeps_token_shape() -> None:
    x = torch.randn(2, 20, 16)
    block = expert_choice_moe(
        d_model=16, num_experts=3, expansion_factor=2, capacity_factor=1.25
    )
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[:, -4:] = False
    out = block(x, mask=mask)
    assert out.shape == x.shape


def test_expert_choice_moe_scatter_under_cuda_bf16_autocast() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return
    device = torch.device("cuda")
    block = expert_choice_moe(
        d_model=16, num_experts=3, expansion_factor=2, capacity_factor=1.25
    ).to(device)
    x = torch.randn(4, 32, 16, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = block(x)
        loss = out.float().pow(2).mean()
    loss.backward()
    assert out.shape == x.shape


def test_expert_choice_moe_pool_concatenates_experts() -> None:
    x = torch.randn(2, 20, 16)
    block = expert_choice_moe_pool(
        d_model=16, num_experts=3, expansion_factor=2, capacity_factor=1.25
    )
    out = block(x)
    assert out.shape == (2, 3 * 16)


def test_create_model_injects_num_targets_and_task_names() -> None:
    config = {
        "model": {
            "architecture_type": "single",
            "embedder": [],
            "trunk": [],
            "heads": {
                "regression": [
                    {
                        "name": "family_grouped_regression_head",
                        "in_features": 4,
                        "hidden_features": 4,
                    }
                ]
            },
        }
    }
    model = create_model_from_config(
        config,
        seq_length=1,
        regression_tasks=["PPNP_A", "HD_A", "PPNP_B"],
    )
    head = model.head_blocks["regression"][0]
    assert head.num_targets == 3
    assert head.families["PPNP"] == [0, 2]
    assert model(torch.randn(2, 1, 4))["regression"].shape == (2, 3)


def test_transformer_ffn_extra_hidden_layer_keeps_shape() -> None:
    x = torch.randn(2, 8, 32)
    one = transformer(d_model=32, num_heads=4, d_ff=64, ffn_num_hidden_layers=1)
    two = transformer(d_model=32, num_heads=4, d_ff=64, ffn_num_hidden_layers=2)
    assert len(one.ffn.hidden) == 1
    assert len(two.ffn.hidden) == 2
    assert one(x).shape == x.shape
    assert two(x).shape == x.shape


def test_transformer_entmax_attention_keeps_shape() -> None:
    x = torch.randn(2, 8, 32)
    block = transformer(d_model=32, num_heads=4, d_ff=64, attn_normalize="entmax15")
    assert block.attention.attn_normalize == "entmax15"
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_multi_head_pool_entmax_axis2_keeps_shape() -> None:
    from aquila.blocks import multi_head_pool

    x = torch.randn(2, 16, 32)
    soft = multi_head_pool(d_model=32, num_heads=4, pool_axis=2, attn_normalize="softmax")
    sparse = multi_head_pool(d_model=32, num_heads=4, pool_axis=2, attn_normalize="entmax15")
    assert soft.attn_normalize == "softmax"
    assert sparse.attn_normalize == "entmax15"
    assert soft(x).shape == (2, 16)
    assert sparse(x).shape == (2, 16)
    assert torch.isfinite(sparse(x)).all()


def test_regression_head_extra_mlp_layers() -> None:
    from aquila.blocks import regression_head

    one = regression_head(in_features=16, num_targets=3, hidden_features=8, num_hidden_layers=1)
    two = regression_head(in_features=16, num_targets=3, hidden_features=8, num_hidden_layers=2)
    three = regression_head(in_features=16, num_targets=3, hidden_features=8, num_hidden_layers=3)
    x = torch.randn(4, 16)
    assert one.num_hidden_layers == 1
    assert two.num_hidden_layers == 2
    assert three.num_hidden_layers == 3
    assert one(x).shape == (4, 3)
    assert two(x).shape == (4, 3)
    assert three(x).shape == (4, 3)
    assert sum(isinstance(m, torch.nn.Linear) for m in two.network) == 3
    assert sum(isinstance(m, torch.nn.Linear) for m in three.network) == 4


def test_skipfuse_configs_forward() -> None:
    import yaml
    from pathlib import Path
    from aquila.varnn import create_model_from_config

    root = Path("/home/gulei/projects/Aquila-GS/benchmark/aquila-snp/configs")
    x = torch.randn(2, 128, 8)
    for name in (
        "v5-1.skipfuse-preattn-seq.yaml",
        "v5-2.skipfuse-preattn-pool.yaml",
        "v5-3.skipfuse-embed-seq.yaml",
        "v5-4.skipfuse-embed-pool.yaml",
        "v5-5.skipfuse-both-seq.yaml",
        "v5-6.skipfuse-both-pool.yaml",
    ):
        cfg = yaml.safe_load((root / name).read_text())
        model = create_model_from_config(
            cfg, seq_length=128, regression_tasks=["t0", "t1"]
        )
        model.eval()
        with torch.no_grad():
            out = model(x)["regression"]
        assert out.shape == (2, 2), name
        assert torch.isfinite(out).all(), name

    cfg = yaml.safe_load((root / "v5-3.skipfuse-embed-seq.yaml").read_text())
    model = create_model_from_config(
        cfg, seq_length=4096, regression_tasks=["t0", "t1"]
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 4096, 8))["regression"]
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()

# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Shape and grouping tests for multi-task regression heads."""

from __future__ import annotations

import torch

from aquila.blocks import (
    blur_pool_down_conv_tower,
    expert_choice_moe,
    expert_choice_moe_pool,
    family_grouped_regression_head,
    fixed_depth_adaptive_stride_down_conv_tower,
    film_regression_head,
    group_traits_by_family,
    growing_channel_down_conv_tower,
    mean_max_residual_down_conv_tower,
    mean_residual_down_conv_tower,
    mmoe_regression_head,
    patch_merge_down_conv_tower,
    per_trait_regression_head,
    raw_attention_residual_down_conv_tower,
    raw_gated_mean_residual_down_conv_tower,
    raw_max_attention_residual_down_conv_tower,
    raw_mean_attention_residual_down_conv_tower,
    raw_mean_residual_down_conv_tower,
    raw_multi_pool_residual_down_conv_tower,
    raw_std_attention_residual_down_conv_tower,
    shared_stem_family_head,
    shared_stem_private_head,
    trait_query_regression_head,
    transformer,
    select_adaptive_downsample_strides,
)
from aquila.varnn import create_model_from_config


def test_adaptive_stride_schedule_matches_requested_boundaries() -> None:
    expected = {
        2_000: ((1, 2), 1_000),
        4_000: ((2, 2), 1_000),
        8_000: ((2, 4), 1_000),
        12_000: ((2, 4), 1_500),
        20_000: ((4, 4), 1_250),
        40_000: ((4, 8), 1_250),
        80_000: ((8, 8), 1_250),
    }
    for seq_length, result in expected.items():
        assert select_adaptive_downsample_strides(seq_length) == result


def test_fixed_depth_adaptive_stride_tower_shape_and_capacity() -> None:
    tower = fixed_depth_adaptive_stride_down_conv_tower(
        seq_length=8_000,
        in_channels=8,
        out_channels=16,
        kernel_size=5,
        num_stages=2,
        max_seq_len=1_500,
        stride_choices=[1, 2, 4, 8],
    )
    x = torch.randn(2, 8_000, 8)
    mask = torch.ones(2, 8_000, dtype=torch.bool)
    output, output_mask = tower(x, mask)
    assert tower.strides == (2, 4)
    assert len(tower.blocks) == 2
    assert output.shape == (2, 1_000, 16)
    assert output_mask.shape == (2, 1_000)


def test_information_preserving_tower_variants_keep_expected_shape() -> None:
    factories = (
        patch_merge_down_conv_tower,
        mean_residual_down_conv_tower,
        mean_max_residual_down_conv_tower,
        blur_pool_down_conv_tower,
    )
    x = torch.randn(2, 65, 8)
    mask = torch.ones(2, 65, dtype=torch.bool)
    mask[0, -3:] = False
    for factory in factories:
        tower = factory(
            seq_length=65,
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            num_stages=2,
            max_seq_len=10,
            stride_choices=[1, 2, 4, 8],
        )
        output, output_mask = tower(x, mask)
        assert tower.strides == (2, 4)
        assert output.shape == (2, 9, 16)
        assert output_mask.shape == (2, 9)
        assert torch.isfinite(output).all()


def test_raw_mean_residual_tower_matches_original_dynamic_depth() -> None:
    tower = raw_mean_residual_down_conv_tower(
        in_channels=8,
        out_channels=16,
        kernel_size=5,
        seq_len_threshold=1_500,
    )
    output, output_mask = tower(
        torch.randn(1, 8_592, 8),
        torch.ones(1, 8_592, dtype=torch.bool),
    )
    assert output.shape == (1, 1_074, 16)
    assert output_mask.shape == (1, 1_074)
    assert all(block.stride == 2 for block in tower.blocks)
    assert torch.isfinite(output).all()


def test_raw_attention_residual_variants_match_dynamic_depth() -> None:
    factories = (
        raw_attention_residual_down_conv_tower,
        raw_gated_mean_residual_down_conv_tower,
        raw_max_attention_residual_down_conv_tower,
        raw_mean_attention_residual_down_conv_tower,
        raw_multi_pool_residual_down_conv_tower,
        raw_std_attention_residual_down_conv_tower,
    )
    x = torch.randn(1, 65, 8)
    mask = torch.ones(1, 65, dtype=torch.bool)
    mask[:, -1] = False
    for factory in factories:
        tower = factory(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            seq_len_threshold=10,
        )
        output, output_mask = tower(x, mask)
        assert output.shape == (1, 9, 16)
        assert output_mask.shape == (1, 9)
        assert torch.isfinite(output).all()


def test_growing_channel_tower_schedule_and_output_projection() -> None:
    for growth, expected in ((64, (320, 384, 448)), (128, (384, 512, 640))):
        tower = growing_channel_down_conv_tower(
            seq_length=8_592,
            in_channels=256,
            out_channels=256,
            channel_growth=growth,
            kernel_size=5,
            seq_len_threshold=1_500,
        )
        assert tower.num_stages == 3
        assert tower.stage_channels == expected
        output, output_mask = tower(
            torch.randn(1, 8_592, 256),
            torch.ones(1, 8_592, dtype=torch.bool),
        )
        assert output.shape == (1, 1_074, 256)
        assert output_mask.shape == (1, 1_074)


def test_patch_merge_uses_all_values_in_stride_window() -> None:
    tower = patch_merge_down_conv_tower(
        seq_length=8,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        dropout=0.0,
        residual=False,
        num_stages=1,
        max_seq_len=4,
        stride_choices=[2],
        layer_scale_init=0.0,
    )
    stage = tower.blocks[0]
    with torch.no_grad():
        stage.merge_norm = torch.nn.Identity()
        stage.merge_projection.weight.fill_(1.0)
        stage.merge_projection.bias.zero_()
    x = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1)
    # Bypass feature extraction to isolate the window merge contract.
    stage.conv_feat = torch.nn.Identity()
    stage.conv_feat.forward = lambda value, mask=None: (value, mask)
    output, _ = tower(x)
    assert torch.equal(output.squeeze(), torch.tensor([3.0, 7.0, 11.0, 15.0]))


def test_model_builder_injects_sequence_length_into_adaptive_tower() -> None:
    config = {
        "model": {
            "architecture_type": "single",
            "d_model": 16,
            "embedder": {
                "name": "conv_block",
                "in_channels": 8,
                "out_channels": "d_model",
                "kernel_size": 3,
            },
            "trunk": [{
                "name": "fixed_depth_adaptive_stride_down_conv_tower",
                "in_channels": "d_model",
                "out_channels": "d_model",
                "kernel_size": 3,
                "num_stages": 2,
                "max_seq_len": 1_500,
                "stride_choices": [1, 2, 4, 8],
            }],
            "heads": {
                "regression": {
                    "name": "regression_head",
                    "in_features": None,
                    "hidden_features": 8,
                }
            },
        }
    }
    model = create_model_from_config(
        config,
        seq_length=4_000,
        regression_tasks=["trait"],
    )
    tower = model.trunk_blocks[0]
    assert tower.seq_length == 4_000
    assert tower.strides == (2, 2)


def test_d_model_rewrites_trait_query_head_width() -> None:
    config = {
        "model": {
            "architecture_type": "single",
            "d_model": 128,
            "embedder": {
                "name": "conv_block",
                "in_channels": 8,
                "out_channels": "d_model",
                "kernel_size": 3,
            },
            "trunk": [{
                "name": "fixed_depth_adaptive_stride_down_conv_tower",
                "in_channels": "d_model",
                "out_channels": "d_model",
                "kernel_size": 3,
                "num_stages": 2,
                "max_seq_len": 1_500,
            }],
            "heads": {
                "regression": {
                    "name": "trait_query_regression_head",
                    "d_model": "d_model",
                    "num_heads": 8,
                    "hidden_features": 16,
                }
            },
        }
    }
    model = create_model_from_config(
        config,
        seq_length=2_000,
        regression_tasks=["trait_a", "trait_b"],
    )
    head = model.head_blocks["regression"][0]
    assert head.queries.shape == (2, 128)
    assert model(torch.randn(2, 2_000, 8))["regression"].shape == (2, 2)


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


def test_multi_head_pool_selectable_axis2_heads_keep_shape() -> None:
    from aquila.blocks import multi_head_pool

    x = torch.randn(2, 16, 32)
    mean_max_attention = multi_head_pool(
        d_model=32,
        num_heads=3,
        pool_axis=2,
        pool_types=["mean", "max", "attention"],
    )
    mean_attention = multi_head_pool(
        d_model=32,
        num_heads=2,
        pool_axis=2,
        pool_types=["mean", "attention"],
    )
    assert mean_max_attention.pool_types == ("mean", "max", "attention")
    assert mean_attention.pool_types == ("mean", "attention")
    assert mean_max_attention(x).shape == (2, 16)
    assert mean_attention(x).shape == (2, 16)
    assert torch.isfinite(mean_max_attention(x)).all()
    assert torch.isfinite(mean_attention(x)).all()


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


def test_mqa_d_ff_is_configurable() -> None:
    from aquila.blocks import transformer_mqa

    wide = transformer_mqa(d_model=32, num_query_heads=4, qk_head_dim=8, v_head_dim=8)
    narrow = transformer_mqa(
        d_model=32, num_query_heads=4, qk_head_dim=8, v_head_dim=8, d_ff=32
    )
    assert wide.ffn.hidden[0].out_features == 64
    assert narrow.ffn.hidden[0].out_features == 32
    x = torch.randn(2, 16, 32)
    assert wide(x).shape == x.shape
    assert narrow(x).shape == x.shape


def test_multi_branch_d_model_and_dropout_are_global() -> None:
    from aquila.blocks import GatedFusionBlock, MLPBlock, RegressionHead
    from aquila.blocks_v2 import AdaptiveDownsampleTower
    from aquila.varnn import create_model_from_config

    branch = {
        "embedder": [
            {
                "name": "conv_block",
                "in_channels": 8,
                "out_channels": 256,
                "kernel_size": 15,
            }
        ],
        "trunk": [
            {
                "name": "std_down_conv_tower",
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": 9,
                "seq_len_threshold": 16,
            },
            {
                "name": "transformer_mqa",
                "d_model": 256,
                "num_query_heads": 4,
                "qk_head_dim": 8,
                "v_head_dim": 8,
                "d_ff": 256,
            },
            {
                "name": "multi_head_pool",
                "d_model": 256,
                "num_heads": 4,
                "pool_axis": 2,
            },
        ],
    }
    config = {
        "model": {
            "architecture_type": "multi_branch",
            "d_model": 32,
            "dropout": 0.5,
        },
        "train": {
            "branches": {
                "snp": branch,
                "indel": {
                    **branch,
                    "embedder": [{**branch["embedder"][0], "in_channels": 4}],
                },
                "sv": {
                    **branch,
                    "embedder": [{**branch["embedder"][0], "in_channels": 4}],
                },
            },
            "fusion": [
                {
                    "name": "gated_fusion",
                    "fusion_dim": 256,
                    "num_branches": 3,
                }
            ],
            "shared_trunk": [
                {
                    "name": "mlp_block",
                    "in_features": 256,
                    "hidden_features": 64,
                    "out_features": 256,
                    "num_layers": 2,
                }
            ],
            "heads": {
                "regression": [
                    {
                        "name": "regression_head",
                        "in_features": None,
                        "hidden_features": 16,
                    }
                ]
            },
        },
    }

    model = create_model_from_config(
        config,
        seq_length={"snp": 32, "indel": 32, "sv": 32},
        regression_tasks=["trait"],
    )

    tower = model.branch_trunks["snp"][0]
    fusion = model.fusion_blocks[0]
    shared = model.shared_trunk_blocks[0]
    head = model.head_blocks["regression"][0]
    assert isinstance(tower, AdaptiveDownsampleTower)
    assert isinstance(fusion, GatedFusionBlock)
    assert isinstance(shared, MLPBlock)
    assert isinstance(head, RegressionHead)
    assert tower.blocks[0].feat.conv.out_channels == 32
    assert fusion.fusion_dim == 32
    assert fusion.dropout.p == 0.5
    assert shared.network[0].in_features == 32
    assert shared.network[3].p == 0.5
    assert head.network[3].p == 0.5


def test_v5_transformer_ablation_configs_forward() -> None:
    import yaml
    from pathlib import Path
    from aquila.varnn import create_model_from_config

    root = Path("/home/gulei/projects/Aquila-GS/benchmark/aquila-snp/configs")
    x = torch.randn(2, 128, 8)
    for name in (
        "v5.baseline.yaml",
        "v5-1.transformer-mqa-norope.yaml",
        "v5-2.transformer-mha-rope.yaml",
        "v5-0.transformer-mqa.yaml",
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

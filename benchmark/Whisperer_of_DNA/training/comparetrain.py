import sys
import os

# 将项目根目录添加到 sys.path
# __file__ 是 train.py 的路径
# os.path.dirname(__file__) 是 training/ 目录
# os.path.dirname(os.path.dirname(__file__)) 是项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import time
import logging
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import traceback
import h5py 

from data.datamodule import WhisperDNADataModule
from data.datamodule_onlySNP import WhisperDNADataModule_onlySNP
from models.DNAWhisper import DNAWhisper
from systems.dynamic_optimizer import DynamicTrainingCallback
from training.utils.config_utils import load_config
from typing import List, Optional, Dict, Any


# 移除旧的 save_predictions_to_csv 和 save_pooling_weights 函数
def save_predictions_and_weights(
    model: DNAWhisper,
    datamodule: Any,
    stage: str,
    output_dir: Path,
    phenotype_names: List[str],
    data_loader_params: Dict[str, Any],
    collate_fn: Optional[callable],
    trainer_config: Dict[str, Any]
):
    """
    在单个设备上使用给定模型和数据集运行预测。
    将预测结果保存到 CSV 文件，并将池化权重保存到 HDF5 文件。
    推理只执行一次。权重分批保存以减少内存占用。
    假设 model.predict_step 返回必要的 'batch_size', 
    'num_embedding_blocks', 和 'num_gfi_blocks_list' 用于重塑。
    """
    dataset_to_process = None
    if stage == 'train':
        dataset_to_process = getattr(datamodule, 'train_dataset', None)
    elif stage == 'val':
        dataset_to_process = getattr(datamodule, 'val_dataset', None)
    elif stage == 'test':
        dataset_to_process = getattr(datamodule, 'test_dataset', None)

    if dataset_to_process is None:
        print(f"警告: 阶段 '{stage}' 的数据集未提供，跳过预测和权重保存。")
        return
    if len(dataset_to_process) == 0:
        print(f"警告: 阶段 '{stage}' 的数据集为空，跳过预测和权重保存。")
        return

    print(f"开始为阶段 '{stage}' 进行预测并提取池化权重...")
    model.eval()

    # DataLoader 配置
    loader_batch_size = data_loader_params.get("val_batch_size", data_loader_params.get("batch_size", 64))
    loader_num_workers = data_loader_params.get("num_workers", 0)
    loader_pin_memory = data_loader_params.get("pin_memory", True)

    predict_dataloader = DataLoader(
        dataset=dataset_to_process,
        batch_size=loader_batch_size, # DataLoader 的 batch_size 是真实样本批次大小 B
        num_workers=loader_num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_pin_memory,
        sampler=SequentialSampler(dataset_to_process),
        shuffle=False
    )
    print(f"为阶段 '{stage}' 创建了预测专用的 DataLoader。样本数: {len(dataset_to_process)}")

    predict_trainer = pl.Trainer(
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=1,
        precision=str(trainer_config.get("precision", "32-true")),
        logger=False,
        callbacks=[],
        strategy='auto'
    )
    print(f"预测 Trainer 实例化完成。")

    # 执行一次预测，获取所有批次的输出
    # DNAWhisper.predict_step 应该返回一个包含 'preds', 'labels', 
    # 'embedding_pooling_weights', 'gfi_pooling_weights_list',
    # 'batch_size', 'num_embedding_blocks', 'num_gfi_blocks_list' 的字典
    all_batch_outputs = predict_trainer.predict(model=model, dataloaders=predict_dataloader, return_predictions=True)

    if not all_batch_outputs:
        print(f"警告: 阶段 '{stage}' 没有从 predict_trainer 返回任何输出。")
        return

    # --- 初始化 CSV 保存所需列表 ---
    all_preds_list_for_csv = []
    all_labels_list_for_csv = []

    # --- HDF5 文件和权重保存设置 ---
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_file_path = output_dir / f"{stage}_pooling_weights.h5"
    
    num_gfi_model_blocks = 0
    if hasattr(model, 'model') and hasattr(model.model, 'gfi_former') and hasattr(model.model.gfi_former, 'blocks'):
        num_gfi_model_blocks = len(model.model.gfi_former.blocks)
    elif all_batch_outputs and isinstance(all_batch_outputs[0], dict) and 'gfi_pooling_weights_list' in all_batch_outputs[0] and \
         all_batch_outputs[0]['gfi_pooling_weights_list'] is not None:
        num_gfi_model_blocks = len(all_batch_outputs[0]['gfi_pooling_weights_list'])
        print(f"从第一个批次推断 GFI Block 数量: {num_gfi_model_blocks}")
    else:
        print(f"警告: 无法从模型或第一个批次输出确定 GFI Block 数量。GFI 权重可能无法正确保存。")


    # HDF5 数据集创建标志和形状信息 (从第一个有效批次确定)
    emb_ds_created = False
    gfi_ds_created = [False] * num_gfi_model_blocks
    
    # 用于存储每个数据集的 (N, E, S) 形状，其中 N 是块数，E 是表型/专家数，S 是序列长度
    # 这些将从第一个包含有效权重的批次中确定
    emb_shape_N_E_S = None 
    gfi_shapes_N_E_S_list = [None] * num_gfi_model_blocks


    try:
        with h5py.File(h5_file_path, 'w') as hf:
            print(f"正在处理批次数据以保存预测 (CSV) 和池化权重 (HDF5) 至: {output_dir} (文件: {h5_file_path})")

            for batch_idx, batch_output in enumerate(all_batch_outputs):
                if not isinstance(batch_output, dict):
                    print(f"警告: 阶段 '{stage}' 的批次 {batch_idx} 输出不是字典。跳过此批次。内容: {batch_output}")
                    continue

                # --- 收集 CSV 预测数据 ---
                if 'preds' in batch_output and batch_output['preds'] is not None and \
                   'labels' in batch_output and batch_output['labels'] is not None:
                    all_preds_list_for_csv.append(batch_output['preds'].detach().cpu())
                    all_labels_list_for_csv.append(batch_output['labels'].detach().cpu())
                else:
                    print(f"警告: 阶段 '{stage}' 的批次 {batch_idx} 缺少 'preds' 或 'labels'。")

                # --- 分批保存池化权重到 HDF5 ---
                # 获取真实批次大小 B (由 predict_step 返回)
                # 注意: DataLoader 的 batch_size 是 B，但模型内部处理的可能是 B*N
                # 我们需要 predict_step 明确返回真实的 B，以及每个模块的 N
                
                true_batch_size_from_model = batch_output.get('batch_size') # 期望 predict_step 返回这个
                if true_batch_size_from_model is None:
                    print(f"警告: 阶段 '{stage}' 的批次 {batch_idx} 输出中未找到 'batch_size'。无法正确重塑权重。跳过此批次的权重保存。")
                    continue
                
                # Embedding pooling weights
                # 期望形状: (B_true * N_emb, E_emb, S_emb)
                emb_weights_flat = batch_output.get('embedding_pooling_weights')
                # 期望 predict_step 返回每个真实样本的 Embedding 块数
                N_emb_per_sample = batch_output.get('num_embedding_blocks') 

                if emb_weights_flat is not None and N_emb_per_sample is not None and N_emb_per_sample > 0:
                    if emb_weights_flat.shape[0] == true_batch_size_from_model * N_emb_per_sample:
                        E_emb, S_emb = emb_weights_flat.shape[1], emb_weights_flat.shape[2]
                        
                        if not emb_ds_created:
                            emb_shape_N_E_S = (N_emb_per_sample, E_emb, S_emb)
                            # HDF5 数据集形状: (累计真实样本数, N_emb, E_emb, S_emb)
                            hf.create_dataset('embedding/pooling_weights', 
                                              shape=(0, N_emb_per_sample, E_emb, S_emb), 
                                              maxshape=(None, N_emb_per_sample, E_emb, S_emb),
                                              dtype=emb_weights_flat.dtype.name.lower(),
                                              compression="gzip", chunks=True)
                            emb_ds_created = True
                        
                        # 检查形状是否与初始创建时一致
                        if emb_shape_N_E_S == (N_emb_per_sample, E_emb, S_emb):
                            # 重塑: (B_true * N_emb, E_emb, S_emb) -> (B_true, N_emb, E_emb, S_emb)
                            emb_weights_reshaped = emb_weights_flat.reshape(
                                true_batch_size_from_model, N_emb_per_sample, E_emb, S_emb
                            )
                            dset = hf['embedding/pooling_weights']
                            dset.resize(dset.shape[0] + true_batch_size_from_model, axis=0)
                            dset[-true_batch_size_from_model:] = emb_weights_reshaped.cpu().numpy()
                        else:
                            print(f"警告: 阶段 '{stage}' 批次 {batch_idx} Embedding 权重块/特征维度与首次创建不一致。预期 N,E,S: {emb_shape_N_E_S}, 得到: {(N_emb_per_sample, E_emb, S_emb)}。跳过。")
                    else:
                        print(f"警告: 阶段 '{stage}' 批次 {batch_idx} Embedding 权重维度0 ({emb_weights_flat.shape[0]}) "
                              f"与 B_true*N_emb ({true_batch_size_from_model}*{N_emb_per_sample}) 不匹配。跳过。")
                elif emb_weights_flat is not None and (N_emb_per_sample is None or N_emb_per_sample == 0):
                     print(f"信息: 阶段 '{stage}' 批次 {batch_idx} Embedding 权重存在，但 num_embedding_blocks 为 None 或 0。跳过。")

                
                # GFIFormer block pooling weights
                # gfi_weights_flat_list 中每个元素的期望形状: (B_true * N_gfi_i, E_gfi_i, S_gfi_i)
                gfi_weights_flat_list = batch_output.get('gfi_pooling_weights_list', [])
                # num_gfi_blocks_list 中每个元素是对应 GFI Block 的 N_gfi_i
                num_gfi_blocks_per_sample_list = batch_output.get('num_gfi_blocks_list', [])

                if len(gfi_weights_flat_list) == num_gfi_model_blocks and \
                   len(num_gfi_blocks_per_sample_list) == num_gfi_model_blocks:
                    
                    gfi_group_path = 'gfi_blocks'
                    if num_gfi_model_blocks > 0 and gfi_group_path not in hf:
                        hf.create_group(gfi_group_path)

                    for i in range(num_gfi_model_blocks):
                        gfi_weights_flat_i = gfi_weights_flat_list[i]
                        N_gfi_i_per_sample = num_gfi_blocks_per_sample_list[i]

                        if gfi_weights_flat_i is not None and N_gfi_i_per_sample is not None and N_gfi_i_per_sample > 0:
                            if gfi_weights_flat_i.shape[0] == true_batch_size_from_model * N_gfi_i_per_sample:
                                E_gfi_i, S_gfi_i = gfi_weights_flat_i.shape[1], gfi_weights_flat_i.shape[2]
                                dataset_path = f'{gfi_group_path}/block_{i}/pooling_weights'
                                
                                if not gfi_ds_created[i]:
                                    gfi_shapes_N_E_S_list[i] = (N_gfi_i_per_sample, E_gfi_i, S_gfi_i)
                                    # HDF5 数据集形状: (累计真实样本数, N_gfi_i, E_gfi_i, S_gfi_i)
                                    hf.create_dataset(dataset_path, 
                                                      shape=(0, N_gfi_i_per_sample, E_gfi_i, S_gfi_i), 
                                                      maxshape=(None, N_gfi_i_per_sample, E_gfi_i, S_gfi_i),
                                                      dtype=gfi_weights_flat_i.dtype.name.lower(),
                                                      compression="gzip", chunks=True)
                                    gfi_ds_created[i] = True

                                if gfi_shapes_N_E_S_list[i] == (N_gfi_i_per_sample, E_gfi_i, S_gfi_i):
                                    # 重塑: (B_true * N_gfi_i, E_gfi_i, S_gfi_i) -> (B_true, N_gfi_i, E_gfi_i, S_gfi_i)
                                    gfi_weights_reshaped_i = gfi_weights_flat_i.reshape(
                                        true_batch_size_from_model, N_gfi_i_per_sample, E_gfi_i, S_gfi_i
                                    )
                                    dset_gfi = hf[dataset_path]
                                    dset_gfi.resize(dset_gfi.shape[0] + true_batch_size_from_model, axis=0)
                                    dset_gfi[-true_batch_size_from_model:] = gfi_weights_reshaped_i.cpu().numpy()
                                else:
                                    print(f"警告: 阶段 '{stage}' 批次 {batch_idx} GFI Block {i} 权重块/特征维度与首次创建不一致。预期 N,E,S: {gfi_shapes_N_E_S_list[i]}, 得到: {(N_gfi_i_per_sample, E_gfi_i, S_gfi_i)}。跳过。")
                            else:
                                print(f"警告: 阶段 '{stage}' 批次 {batch_idx} GFI Block {i} 权重维度0 ({gfi_weights_flat_i.shape[0]}) "
                                      f"与 B_true*N_gfi ({true_batch_size_from_model}*{N_gfi_i_per_sample}) 不匹配。跳过。")
                        elif gfi_weights_flat_i is not None and (N_gfi_i_per_sample is None or N_gfi_i_per_sample == 0):
                            print(f"信息: 阶段 '{stage}' 批次 {batch_idx} GFI Block {i} 权重存在，但 num_gfi_blocks_list[{i}] 为 None 或 0。跳过。")
                elif len(gfi_weights_flat_list) != num_gfi_model_blocks or len(num_gfi_blocks_per_sample_list) != num_gfi_model_blocks :
                    if batch_idx == 0: # 仅在第一个批次警告结构不匹配
                        print(f"警告: 阶段 '{stage}' 批次 {batch_idx} GFI Block 数量或 N_gfi 列表长度与模型不匹配。模型期望 {num_gfi_model_blocks} 个块。 "
                              f"权重列表长度: {len(gfi_weights_flat_list)}, N_gfi 列表长度: {len(num_gfi_blocks_per_sample_list)}. GFI 权重可能无法正确保存。")
            
            if emb_ds_created:
                 print(f"  阶段 '{stage}': Embedding 池化权重已保存/追加。最终HDF5形状 (总真实样本数, N_emb, E_emb, S_emb): {hf['embedding/pooling_weights'].shape}")
            for i in range(num_gfi_model_blocks):
                if gfi_ds_created[i]:
                    dataset_path = f'gfi_blocks/block_{i}/pooling_weights'
                    print(f"  阶段 '{stage}': GFI Block {i} 池化权重已保存/追加。最终HDF5形状 (总真实样本数, N_gfi_{i}, E_gfi_{i}, S_gfi_{i}): {hf[dataset_path].shape}")
            print(f"阶段 '{stage}' 的池化权重已成功保存至 HDF5 文件: {h5_file_path}")

    except Exception as e:
        print(f"错误: 在为阶段 '{stage}' 保存池化权重到 HDF5 时发生错误: {e}")
        traceback.print_exc()

    # --- 完成 CSV 预测结果的保存 ---
    if all_preds_list_for_csv and all_labels_list_for_csv:
        all_preds_np = torch.cat(all_preds_list_for_csv, dim=0).float().numpy()
        all_labels_np = torch.cat(all_labels_list_for_csv, dim=0).float().numpy()

        print(f"阶段 '{stage}' CSV: 收集到的预测数量: {all_preds_np.shape[0]}, 标签数量: {all_labels_np.shape[0]}")

        if all_preds_np.shape[0] != len(dataset_to_process):
             print(f"警告: 阶段 '{stage}' CSV 收集到的预测数量 ({all_preds_np.shape[0]}) 与数据集大小 ({len(dataset_to_process)}) 不匹配。")

        # phenotype_names 来自函数参数，是配置中的表型名称
        num_phenotypes_config = len(phenotype_names)
        
        # 检查预测和标签的维度是否与表型数量匹配
        preds_cols_ok = (all_preds_np.ndim == 2 and all_preds_np.shape[1] == num_phenotypes_config)
        
        labels_cols_ok = False
        if all_labels_np.ndim == 2 and all_labels_np.shape[1] == num_phenotypes_config:
            labels_cols_ok = True
        elif all_labels_np.ndim == 1 and num_phenotypes_config == 1: # 单表型，标签是一维的
            all_labels_np = all_labels_np.reshape(-1, 1) # 转换为 (N, 1)
            labels_cols_ok = True

        data_for_df = {}
        if preds_cols_ok and labels_cols_ok:
            for i, name in enumerate(phenotype_names):
                data_for_df[f'{name}_pred'] = all_preds_np[:, i]
                data_for_df[f'{name}_label'] = all_labels_np[:, i]
        else:
            print(f"警告: 阶段 '{stage}' 的预测/标签列数与配置的表型数量 ({num_phenotypes_config}) 不匹配。")
            print(f"  预测形状: {all_preds_np.shape}, 标签形状: {all_labels_np.shape}")
            print(f"  将尝试使用预测的列数和通用表型名称保存 CSV。")
            
            num_cols_to_save = all_preds_np.shape[1]
            for i in range(num_cols_to_save):
                data_for_df[f'phenotype_{i+1}_pred'] = all_preds_np[:, i]
                if all_labels_np.ndim == 2 and i < all_labels_np.shape[1]:
                    data_for_df[f'phenotype_{i+1}_label'] = all_labels_np[:, i]
                elif all_labels_np.ndim == 1 and i == 0 : # 如果标签是一维的，只用于第一个通用表型
                     data_for_df[f'phenotype_{i+1}_label'] = all_labels_np
        
        if not data_for_df:
            print(f"警告: 阶段 '{stage}' 未能为 CSV 文件准备任何数据列。")
        else:
            try:
                df = pd.DataFrame(data_for_df)
                csv_output_path = output_dir / f"{stage}_predictions.csv"
                df.to_csv(csv_output_path, index=False)
                print(f"已将阶段 '{stage}' 的预测结果保存至 CSV: {csv_output_path}")
            except ValueError as ve:
                print(f"错误: 创建或保存阶段 '{stage}' 的 CSV 文件时发生错误 (可能是列长度不匹配): {ve}")
                traceback.print_exc()
    else:
        print(f"警告: 阶段 '{stage}' 未收集到预测或标签数据，跳过 CSV 保存。")

    print(f"阶段 '{stage}' 的预测和权重保存处理完成。")


def train(args):
    """使用给定的配置进行训练"""
    print(f"加载训练配置: {args.training_config}")
    training_config = load_config(args.training_config)

    print(f"加载模型配置: {args.model_config}")
    model_config = load_config(args.model_config)

    # 合并配置 (如果需要)
    if "gradient_checkpointing" in training_config.get("training", {}):
        model_config["gradient_checkpointing"] = training_config["training"]["gradient_checkpointing"]
        print("已将 training_config 中的 gradient_checkpointing 配置传递给模型配置")

    if "logging" in training_config:
        if "logging" not in model_config: model_config["logging"] = {}
        model_config["logging"].update(training_config.get("logging", {}))
        print("已将 training_config 中的 logging 配置合并到模型配置")
    
    # 从 training_config 获取 optimizer 和 scheduler 配置
    optimizer_config = training_config.get("optimizer", {})
    scheduler_config = training_config.get("scheduler", {})
    print(f"优化器配置: {optimizer_config}")
    print(f"调度器配置: {scheduler_config}")

    # 从 model_config 获取表型名称
    phenotype_names = model_config.get("output_layer", {}).get("phenotype_name", None)
    if phenotype_names is None:
        raise ValueError("模型配置文件 model_config.json 中 output_layer.phenotype_name 未定义")
    print(f"使用的表型: {phenotype_names}")

    # 设置随机种子
    random_seed = training_config.get("training", {}).get("random_seed", 42)
    pl.seed_everything(random_seed, workers=True)
    print(f"设置随机种子: {random_seed}")

    # 数据配置
    data_config = training_config.get("data", {})
    h5_file_path = data_config.get("data_path", None)
    if not h5_file_path or not Path(h5_file_path).exists():
        raise FileNotFoundError(f"HDF5 数据文件未找到或未在 training_config.yml 的 data.data_path 中配置: {h5_file_path}")
    print(f"使用 HDF5 数据文件: {h5_file_path}")

    # 训练参数
    train_params = training_config.get("training", {})
    log_params = training_config.get("logging", {})
    checkpoint_params = training_config.get("checkpoint", {})
    early_stopping_params = training_config.get("early_stopping", {})
    dynamic_optimizer_params = training_config.get("dynamic_optimizer", {})
    
    # 交叉验证参数
    cv_params = {
        'use_cv': train_params.get('use_cv_folds', False),
        'n_splits': train_params.get('cv_n_splits', 5),
        'start_fold': train_params.get('cv_fold_idx', 0) # 用于从特定折开始
    }

    # DataLoader 参数 (用于预测和权重保存)
    data_loader_params = {
        "batch_size": data_config.get("batch_size", 64), # 用于训练集预测
        "val_batch_size": data_config.get("val_batch_size", data_config.get("batch_size", 64)), # 用于验证/测试集预测
        "num_workers": data_config.get("num_workers", 8),
        "pin_memory": data_config.get("pin_memory", True),
    }
    
    # 预测 Trainer 配置 (用于 save_predictions_and_weights)
    # 从 training_config 的 training 部分获取 accelerator 和 precision
    # 注意: model_config 中的 precision 用于模型内部，trainer 的 precision 用于 PL Trainer
    trainer_precision_config = str(train_params.get("precision", "32-true"))
    if trainer_precision_config == "16": trainer_precision_config = "16-mixed"
    if trainer_precision_config == "bf16": trainer_precision_config = "bf16-mixed"

    predict_trainer_config = {
        "accelerator": train_params.get("accelerator", "gpu"),
        "precision": trainer_precision_config,
    }
    print(f"预测 Trainer 配置: {predict_trainer_config}")


    # --- 主训练逻辑 ---
    base_log_dir = Path(log_params.get('save_dir', 'logs'))
    project_name = log_params.get('project_name', 'DNAWhisper')
    experiment_base_name = log_params.get('experiment_name', 'run')


    if cv_params['use_cv']:
        print(f"启用 K={cv_params['n_splits']} 折交叉验证，从折 {cv_params['start_fold'] + 1} 开始。")
        all_folds_test_results = []

        for fold_idx in range(cv_params['start_fold'], cv_params['n_splits']):
            current_fold_experiment_name = f"{experiment_base_name}_fold_{fold_idx}"
            fold_log_dir = base_log_dir / project_name / current_fold_experiment_name
            print(f"\n--- 开始训练第 {fold_idx + 1}/{cv_params['n_splits']} 折 (实验名: {current_fold_experiment_name}) ---")

            # 为当前折设置 DataModule
            # 传递 fold_idx 以便 DataModule 内部处理数据分割
            current_training_config_for_dm = training_config.copy() # 确保 training_config 包含 cv_fold_idx
            current_training_config_for_dm['training']['cv_fold_idx'] = fold_idx 
            
            datamodule = WhisperDNADataModule_onlySNP(
                h5_file_path=h5_file_path,
                config=current_training_config_for_dm, # 包含当前折信息的训练配置
                model_config=model_config,      # 原始模型配置
                phenotype_names=phenotype_names,
                seed=random_seed + fold_idx,    # 每折使用不同种子确保数据分割不同
                logger=logging.getLogger(f"DataModule_Fold{fold_idx}")
            )
            datamodule.setup(stage='fit') # 为训练和验证准备数据
            # datamodule.setup(stage='test') # 如果测试集也按折分割，否则在外部加载一次

            collate_fn_for_predict = getattr(datamodule, 'collate_fn_predict', getattr(datamodule, 'collate_fn', None))
            if collate_fn_for_predict is None:
                print(f"警告 (折 {fold_idx + 1}): 未从 DataModule 获取 collate_fn_predict 或 collate_fn。")


            # 模型实例化
            model_fold = DNAWhisper(
                config=model_config, # 传递完整的模型配置字典
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config
            )

            # 日志记录器
            tb_logger_fold = TensorBoardLogger(save_dir=str(base_log_dir / project_name), name=current_fold_experiment_name, version="tb_logs")
            csv_logger_fold = CSVLogger(save_dir=str(base_log_dir / project_name), name=current_fold_experiment_name, version="csv_logs")
            fold_loggers = [tb_logger_fold, csv_logger_fold]

            # 回调函数
            fold_callbacks = []
            checkpoint_dir_fold = fold_log_dir / "checkpoints"
            checkpoint_callback_fold = ModelCheckpoint(
                dirpath=str(checkpoint_dir_fold),
                filename=checkpoint_params.get('filename', '{epoch}-{val_loss:.4f}'),
                monitor=checkpoint_params.get('monitor', 'val_loss'),
                mode=checkpoint_params.get('mode', 'min'),
                save_top_k=checkpoint_params.get('save_top_k', 1), # 通常每折只保存最好的
                save_last=checkpoint_params.get('save_last', True)
            )
            fold_callbacks.append(checkpoint_callback_fold)
            fold_callbacks.append(LearningRateMonitor(logging_interval='step'))
            if early_stopping_params.get("enabled", True):
                fold_callbacks.append(EarlyStopping(
                    monitor=early_stopping_params.get("monitor", "val_loss"),
                    patience=early_stopping_params.get("patience", 15),
                    mode=early_stopping_params.get("mode", "min"),
                    min_delta=early_stopping_params.get("min_delta", 0.0001),
                    verbose=True
                ))
            if dynamic_optimizer_params.get("enabled", False):
                fold_callbacks.append(DynamicTrainingCallback(dynamic_optimizer_params))

            # Trainer
            trainer_fold = pl.Trainer(
                max_epochs=train_params.get("max_epochs", 100),
                accelerator=train_params.get("accelerator", "gpu"),
                devices=train_params.get("devices", 1),
                precision=trainer_precision_config,
                accumulate_grad_batches=train_params.get("accumulate_grad_batches", 1),
                gradient_clip_val=train_params.get("gradient_clip_val", 1.0),
                val_check_interval=train_params.get("val_check_interval", 1.0),
                log_every_n_steps=log_params.get("log_every_n_steps", 50),
                callbacks=fold_callbacks,
                logger=fold_loggers,
                strategy=DDPStrategy(find_unused_parameters=True) if isinstance(train_params.get("devices", 1), (list, int)) and ((isinstance(train_params.get("devices",1), list) and len(train_params.get("devices",1)) > 1) or (isinstance(train_params.get("devices",1), int) and train_params.get("devices",1) > 1)) else 'auto'
            )

            print(f"开始训练模型 (折 {fold_idx + 1})...")
            trainer_fold.fit(model_fold, datamodule=datamodule, ckpt_path=args.checkpoint if fold_idx == cv_params['start_fold'] and args.checkpoint else None)
            
            best_ckpt_path_fold = checkpoint_callback_fold.best_model_path
            print(f"折 {fold_idx + 1} 训练完成。最佳检查点: {best_ckpt_path_fold}")

            # 测试当前折的最佳模型
            if best_ckpt_path_fold:
                print(f"使用检查点进行测试 (折 {fold_idx + 1}): {best_ckpt_path_fold}")
                # datamodule.setup(stage='test') # 确保测试数据已加载 (如果测试集也按折分割)
                test_results_fold = trainer_fold.test(model=model_fold, datamodule=datamodule, ckpt_path=best_ckpt_path_fold)
                if test_results_fold:
                    print(f"模型测试完成 (折 {fold_idx + 1}). 测试结果: {test_results_fold[0]}")
                    all_folds_test_results.append(test_results_fold[0])
            else:
                print(f"警告 (折 {fold_idx + 1}): 未找到最佳检查点，跳过测试。")


            if trainer_fold.is_global_zero:
                if best_ckpt_path_fold:
                    print(f"\n开始加载最佳模型并保存预测结果与池化权重 (折 {fold_idx + 1}, Rank {trainer_fold.global_rank})...")
                    try:
                        model_to_predict = DNAWhisper.load_from_checkpoint(best_ckpt_path_fold)
                        phenotype_names_from_model = model_to_predict.phenotype_names
                        
                        # 确保 datamodule 具有 train_dataset, val_dataset, test_dataset 属性
                        # 如果 datamodule.setup() 没有保留它们，需要在这里重新加载或调整 datamodule
                        datamodule.setup(stage='predict') # 确保所有数据集都已加载并可访问

                        for pred_stage in ['train', 'val', 'test']:
                            print(f"  正在处理阶段: {pred_stage} (折 {fold_idx + 1})")
                            save_predictions_and_weights(
                                model=model_to_predict,
                                datamodule=datamodule,
                                stage=pred_stage,
                                output_dir=fold_log_dir, # 保存到当前折的日志目录
                                phenotype_names=phenotype_names_from_model,
                                data_loader_params=data_loader_params,
                                collate_fn=collate_fn_for_predict,
                                trainer_config=predict_trainer_config
                            )
                        print(f"所有阶段的预测和池化权重保存完成 (折 {fold_idx + 1})。")
                    except Exception as e:
                        print(f"错误 (折 {fold_idx + 1}, Rank {trainer_fold.global_rank}): 在加载模型、保存预测或权重时发生错误: {e}")
                        traceback.print_exc()
                else:
                    print(f"警告 (折 {fold_idx + 1}, Rank {trainer_fold.global_rank}): 由于未找到最佳检查点，跳过保存预测和权重。")
            
            del model_fold, trainer_fold, checkpoint_callback_fold, tb_logger_fold, csv_logger_fold, datamodule
            torch.cuda.empty_cache()
        
        # K 折交叉验证结束后的处理
        if all_folds_test_results:
            avg_results = {}
            all_keys = set().union(*(d.keys() for d in all_folds_test_results))
            for key in all_keys:
                valid_values = [res[key] for res in all_folds_test_results if key in res and isinstance(res[key], (int, float))]
                if valid_values: avg_results[key] = sum(valid_values) / len(valid_values)
            print("\n--- K 折交叉验证平均测试结果 ---")
            for key, value in avg_results.items(): print(f"  {key}: {value:.4f}")

    else: # 单次训练 (非交叉验证)
        exp_log_dir = base_log_dir / project_name / experiment_base_name
        print(f"执行单次训练。日志目录: {exp_log_dir}")

        datamodule = WhisperDNADataModule_onlySNP(
            h5_file_path=h5_file_path,
            config=training_config, # 原始训练配置
            model_config=model_config,
            phenotype_names=phenotype_names,
            seed=random_seed,
            logger=logging.getLogger("DataModule_SingleRun")
        )
        datamodule.setup(stage='fit')
        # datamodule.setup(stage='test') # 如果测试集在 fit 阶段未加载

        collate_fn_for_predict = getattr(datamodule, 'collate_fn_predict', getattr(datamodule, 'collate_fn', None))
        if collate_fn_for_predict is None:
            print(f"警告: 未从 DataModule 获取 collate_fn_predict 或 collate_fn。")

        model = DNAWhisper(
            config=model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )

        # 日志和回调 (与交叉验证中的单折类似，但路径不同)
        tb_logger = TensorBoardLogger(save_dir=str(base_log_dir / project_name), name=experiment_base_name, version="tb_logs")
        csv_logger = CSVLogger(save_dir=str(base_log_dir / project_name), name=experiment_base_name, version="csv_logs")
        loggers = [tb_logger, csv_logger]

        callbacks_list = []
        checkpoint_dir_exp = exp_log_dir / "checkpoints"
        checkpoint_callback_exp = ModelCheckpoint(
            dirpath=str(checkpoint_dir_exp),
            filename=checkpoint_params.get('filename', '{epoch}-{val_loss:.4f}'),
            monitor=checkpoint_params.get('monitor', 'val_loss'),
            mode=checkpoint_params.get('mode', 'min'),
            save_top_k=checkpoint_params.get('save_top_k', 3),
            save_last=checkpoint_params.get('save_last', True)
        )
        callbacks_list.append(checkpoint_callback_exp)
        callbacks_list.append(LearningRateMonitor(logging_interval='step'))
        if early_stopping_params.get("enabled", True):
            callbacks_list.append(EarlyStopping(
                monitor=early_stopping_params.get("monitor", "val_loss"),
                patience=early_stopping_params.get("patience", 15),
                mode=early_stopping_params.get("mode", "min"),
                min_delta=early_stopping_params.get("min_delta", 0.0001),
                verbose=True
            ))
        if dynamic_optimizer_params.get("enabled", False):
            callbacks_list.append(DynamicTrainingCallback(dynamic_optimizer_params))
        
        trainer = pl.Trainer(
            max_epochs=train_params.get("max_epochs", 100),
            accelerator=train_params.get("accelerator", "gpu"),
            devices=train_params.get("devices", 1),
            precision=trainer_precision_config,
            accumulate_grad_batches=train_params.get("accumulate_grad_batches", 1),
            gradient_clip_val=train_params.get("gradient_clip_val", 1.0),
            val_check_interval=train_params.get("val_check_interval", 1.0),
            log_every_n_steps=log_params.get("log_every_n_steps", 50),
            callbacks=callbacks_list,
            logger=loggers,
            strategy=DDPStrategy(find_unused_parameters=True) if isinstance(train_params.get("devices", 1), (list, int)) and ((isinstance(train_params.get("devices",1), list) and len(train_params.get("devices",1)) > 1) or (isinstance(train_params.get("devices",1), int) and train_params.get("devices",1) > 1)) else 'auto'
        )

        print("开始训练模型 (单次运行)...")
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)
        
        best_ckpt_path = checkpoint_callback_exp.best_model_path
        print(f"训练完成。最佳检查点: {best_ckpt_path}")

        if best_ckpt_path:
            print(f"使用检查点进行测试: {best_ckpt_path}")
            test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path)
            if test_results: print(f"模型测试完成. 测试结果: {test_results[0]}")
        else:
            print("警告: 未找到最佳检查点，跳过测试。")

        if trainer.is_global_zero:
            if best_ckpt_path:
                print(f"\n开始加载最佳模型并保存预测结果与池化权重 (Rank {trainer.global_rank})...")
                try:
                    model_to_predict = DNAWhisper.load_from_checkpoint(best_ckpt_path)
                    phenotype_names_from_model = model_to_predict.phenotype_names
                    
                    datamodule.setup(stage='predict') # 确保所有数据集都已加载并可访问

                    for pred_stage in ['train', 'val', 'test']:
                        print(f"  正在处理阶段: {pred_stage}")
                        save_predictions_and_weights(
                            model=model_to_predict,
                            datamodule=datamodule,
                            stage=pred_stage,
                            output_dir=exp_log_dir, # 保存到实验的日志目录
                            phenotype_names=phenotype_names_from_model,
                            data_loader_params=data_loader_params,
                            collate_fn=collate_fn_for_predict,
                            trainer_config=predict_trainer_config
                        )
                    print(f"所有阶段的预测和池化权重保存完成。")
                except Exception as e:
                    print(f"错误 (Rank {trainer.global_rank}): 在加载模型、保存预测或权重时发生错误: {e}")
                    traceback.print_exc()
            else:
                print(f"警告 (Rank {trainer.global_rank}): 由于未找到最佳检查点，跳过保存预测和权重。")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" # Or configure as needed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    parser = argparse.ArgumentParser(description="DNAWhisper模型训练程序")

    parser.add_argument("--model-config", type=str, default="training/config/model_config.json",
                        help="模型配置文件路径")
    parser.add_argument("--training-config", type=str, default="training/config/training_config.yml",
                        help="训练配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径（可选，用于恢复训练）")
    
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
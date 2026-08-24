import sys
import os

# 将项目根目录添加到 sys.path
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
from pytorch_lightning.loggers import TensorBoardLogger # CSVLogger can be added if needed
from pytorch_lightning.strategies import DDPStrategy
import torch
import time
import logging
from torch.utils.data import DataLoader, SequentialSampler # Dataset removed as it's not directly used here
import traceback
import h5py

# from data.datamodule import WhisperDNADataModule # Keep if other general types might be used
# from data.datamodule_onlySNP import WhisperDNADataModule_onlySNP # Replaced by CSV version
from data.datamodule_bycsv import WhisperDNADataModule_byCSV # Use this DataModule
from models.DNAWhisper import DNAWhisper
from systems.dynamic_optimizer import DynamicTrainingCallback
from training.utils.config_utils import load_config
from typing import List, Optional, Dict, Any


def save_predictions_and_weights(
    model: DNAWhisper,
    datamodule: Any, # Should be WhisperDNADataModule_byCSV or similar
    stage: str,
    output_dir: Path,
    phenotype_names: List[str], # These are the names expected in the output CSV
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

    loader_batch_size = data_loader_params.get("val_batch_size", data_loader_params.get("batch_size", 64))
    loader_num_workers = data_loader_params.get("num_workers", 0)
    loader_pin_memory = data_loader_params.get("pin_memory", True)

    predict_dataloader = DataLoader(
        dataset=dataset_to_process,
        batch_size=loader_batch_size,
        num_workers=loader_num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_pin_memory,
        sampler=SequentialSampler(dataset_to_process),
        shuffle=False
    )
    print(f"为阶段 '{stage}' 创建了预测专用的 DataLoader。样本数: {len(dataset_to_process)}")

    predict_trainer = pl.Trainer(
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=1, # Prediction on a single device
        precision=str(trainer_config.get("precision", "32-true")),
        logger=False,
        callbacks=[],
        strategy='auto'
    )
    print(f"预测 Trainer 实例化完成。")

    all_batch_outputs = predict_trainer.predict(model=model, dataloaders=predict_dataloader, return_predictions=True)

    if not all_batch_outputs:
        print(f"警告: 阶段 '{stage}' 没有从 predict_trainer 返回任何输出。")
        return

    all_preds_list_for_csv = []
    all_labels_list_for_csv = []

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

    emb_ds_created = False
    gfi_ds_created = [False] * num_gfi_model_blocks
    emb_shape_N_E_S = None 
    gfi_shapes_N_E_S_list = [None] * num_gfi_model_blocks

    try:
        with h5py.File(h5_file_path, 'w') as hf:
            print(f"正在处理批次数据以保存预测 (CSV) 和池化权重 (HDF5) 至: {output_dir} (文件: {h5_file_path})")

            for batch_idx, batch_output in enumerate(all_batch_outputs):
                if not isinstance(batch_output, dict):
                    print(f"警告: 阶段 '{stage}' 的批次 {batch_idx} 输出不是字典。跳过此批次。内容: {batch_output}")
                    continue

                if 'preds' in batch_output and batch_output['preds'] is not None and \
                   'labels' in batch_output and batch_output['labels'] is not None:
                    all_preds_list_for_csv.append(batch_output['preds'].detach().cpu())
                    all_labels_list_for_csv.append(batch_output['labels'].detach().cpu())
                else:
                    print(f"警告: 阶段 '{stage}' 的批次 {batch_idx} 缺少 'preds' 或 'labels'。")
                
                true_batch_size_from_model = batch_output.get('batch_size')
                if true_batch_size_from_model is None:
                    print(f"警告: 阶段 '{stage}' 的批次 {batch_idx} 输出中未找到 'batch_size'。无法正确重塑权重。跳过此批次的权重保存。")
                    continue
                
                emb_weights_flat = batch_output.get('embedding_pooling_weights')
                N_emb_per_sample = batch_output.get('num_embedding_blocks') 

                if emb_weights_flat is not None and N_emb_per_sample is not None and N_emb_per_sample > 0:
                    if emb_weights_flat.shape[0] == true_batch_size_from_model * N_emb_per_sample:
                        E_emb, S_emb = emb_weights_flat.shape[1], emb_weights_flat.shape[2]
                        
                        if not emb_ds_created:
                            emb_shape_N_E_S = (N_emb_per_sample, E_emb, S_emb)
                            hf.create_dataset('embedding/pooling_weights', 
                                              shape=(0, N_emb_per_sample, E_emb, S_emb), 
                                              maxshape=(None, N_emb_per_sample, E_emb, S_emb),
                                              dtype=emb_weights_flat.cpu().numpy().dtype,
                                              compression="gzip", chunks=True)
                            emb_ds_created = True
                        
                        if emb_shape_N_E_S == (N_emb_per_sample, E_emb, S_emb):
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
                
                gfi_weights_flat_list = batch_output.get('gfi_pooling_weights_list', [])
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
                                    hf.create_dataset(dataset_path, 
                                                      shape=(0, N_gfi_i_per_sample, E_gfi_i, S_gfi_i), 
                                                      maxshape=(None, N_gfi_i_per_sample, E_gfi_i, S_gfi_i),
                                                      dtype=gfi_weights_flat_i.cpu().numpy().dtype,
                                                      compression="gzip", chunks=True)
                                    gfi_ds_created[i] = True

                                if gfi_shapes_N_E_S_list[i] == (N_gfi_i_per_sample, E_gfi_i, S_gfi_i):
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
                    if batch_idx == 0:
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

    if all_preds_list_for_csv and all_labels_list_for_csv:
        all_preds_np = torch.cat(all_preds_list_for_csv, dim=0).float().numpy()
        all_labels_np = torch.cat(all_labels_list_for_csv, dim=0).float().numpy()

        print(f"阶段 '{stage}' CSV: 收集到的预测数量: {all_preds_np.shape[0]}, 标签数量: {all_labels_np.shape[0]}")

        if all_preds_np.shape[0] != len(dataset_to_process):
             print(f"警告: 阶段 '{stage}' CSV 收集到的预测数量 ({all_preds_np.shape[0]}) 与数据集大小 ({len(dataset_to_process)}) 不匹配。")

        num_phenotypes_config = len(phenotype_names)
        
        preds_cols_ok = (all_preds_np.ndim == 2 and all_preds_np.shape[1] == num_phenotypes_config)
        
        labels_cols_ok = False
        if all_labels_np.ndim == 2 and all_labels_np.shape[1] == num_phenotypes_config:
            labels_cols_ok = True
        elif all_labels_np.ndim == 1 and num_phenotypes_config == 1:
            all_labels_np = all_labels_np.reshape(-1, 1)
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
                elif all_labels_np.ndim == 1 and i == 0 :
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
    training_config = load_config(args.training_config) # Loads training_config_csv.yml

    print(f"加载模型配置: {args.model_config}")
    model_config = load_config(args.model_config) # Loads model_config_csv.json

    if "gradient_checkpointing" in training_config:
        model_config["gradient_checkpointing"] = training_config["gradient_checkpointing"]
        print("已将 gradient_checkpointing 配置传递给模型配置")

    if "logging" in training_config:
        if "logging" not in model_config:
            model_config["logging"] = {}
        model_config["logging"].update(training_config.get("logging", {}))
        print("已将 training_config 中的 logging 配置合并到模型配置")

    optimizer_config = training_config.get("optimizer", {})
    scheduler_config = training_config.get("scheduler", {})
    print(f"优化器配置: {optimizer_config}")
    print(f"调度器配置: {scheduler_config}")

    phenotype_names = model_config.get("output_layer", {}).get("phenotype_name", None)
    if not phenotype_names: # Check if it's None or empty list
        raise ValueError(f"模型配置文件 {args.model_config} 中 output_layer.phenotype_name 未定义或为空")
    if isinstance(phenotype_names, str): # Ensure it's a list
        phenotype_names = [phenotype_names]
    print(f"使用的表型: {phenotype_names}")

    random_seed = training_config.get("training", {}).get("random_seed", 42)
    pl.seed_everything(random_seed, workers=True)
    print(f"设置随机种子: {random_seed}")

    data_config = training_config.get("data", {})
    # CSV paths will be read by WhisperDNADataModule_byCSV from its 'config' argument
    genotype_csv_path = data_config.get("genotype_csv_path")
    phenotype_csv_path = data_config.get("phenotype_csv_path")
    if not genotype_csv_path:
        raise FileNotFoundError(f"基因型 CSV 文件路径未在 {args.training_config} 的 data.genotype_csv_path 中配置")
    if not Path(genotype_csv_path).exists():
        raise FileNotFoundError(f"基因型 CSV 文件未找到: {genotype_csv_path}")
    if not phenotype_csv_path:
        raise FileNotFoundError(f"表型 CSV 文件路径未在 {args.training_config} 的 data.phenotype_csv_path 中配置")
    if not Path(phenotype_csv_path).exists():
        raise FileNotFoundError(f"表型 CSV 文件未找到: {phenotype_csv_path}")
    print(f"将使用基因型 CSV 文件: {genotype_csv_path}")
    print(f"将使用表型 CSV 文件: {phenotype_csv_path}")


    train_params = training_config.get("training", {})
    log_params = training_config.get("logging", {})
    checkpoint_params = training_config.get("checkpoint", {})
    early_stopping_params = training_config.get("early_stopping", {})
    dynamic_optimizer_params = training_config.get("dynamic_optimizer", {})
    cv_params = {
        'use_cv': train_params.get('use_cv_folds', False),
        'n_splits': train_params.get('cv_n_splits', 5),
        'start_fold': train_params.get('cv_fold_idx', 0) # cv_fold_idx is 0-based
    }

    data_loader_params = {
        "batch_size": data_config.get("batch_size", 64),
        "val_batch_size": data_config.get("val_batch_size", data_config.get("batch_size", 64)),
        "num_workers": data_config.get("num_workers", 8),
        "pin_memory": data_config.get("pin_memory", True),
    }
    
    # Ensure precision from training_config is a string for pl.Trainer
    # Common values: "32-true", "16-mixed", "bf16-mixed"
    trainer_precision = str(train_params.get("precision", "32-true")) 
    # Handle older numeric config values if necessary, but prefer string in yml
    if trainer_precision == "16": trainer_precision = "16-mixed"
    if trainer_precision == "32": trainer_precision = "32-true"


    predict_trainer_config = {
        "accelerator": train_params.get("accelerator", "gpu"),
        "precision": trainer_precision,
    }

    if cv_params['use_cv']:
        print(f"启用 K={cv_params['n_splits']} 折交叉验证")
        results = []
        # Ensure start_fold is within valid range [0, n_splits-1]
        start_fold_idx = max(0, min(cv_params['start_fold'], cv_params['n_splits'] - 1))

        for fold_idx in range(start_fold_idx, cv_params['n_splits']):
            print(f"\n--- 开始训练第 {fold_idx + 1}/{cv_params['n_splits']} 折 ---")

            current_training_config = training_config.copy() # Deepcopy might be safer if nested dicts are modified
            current_training_config['training']['cv_fold_idx'] = fold_idx

            print(f"设置数据模块 (折 {fold_idx})...")
            datamodule = WhisperDNADataModule_byCSV(
                config=current_training_config, # Contains data paths and split ratios
                model_config=model_config,      # Contains embedding config like Block_length
                phenotype_names=phenotype_names,# Explicitly pass phenotype names from model_config_csv.json
                seed=random_seed,
                logger=logging.getLogger(f"WhisperDNADataModule_Fold{fold_idx}")
            )
            datamodule.setup(stage='fit') # This will trigger data loading and splitting
            print(f"DataModule (折 {fold_idx}) setup 完成.")

            collate_fn = getattr(datamodule, 'collate_fn', None)
            if collate_fn:
                print(f"从 DataModule (折 {fold_idx}) 获取 collate_fn。")
            else:
                print(f"警告: DataModule (折 {fold_idx}) 未提供 collate_fn，将使用 DataLoader 默认值。")

            print(f"实例化模型 (折 {fold_idx})...")
            model = DNAWhisper(
                config=model_config, # Model structure, output layer, loss config
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config
            )
            print(f"模型 (折 {fold_idx}) 实例化完成.")

            if args.checkpoint:
                print(f"正在为第 {fold_idx + 1} 折从检查点加载预训练权重: {args.checkpoint}")
                try:
                    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
                    if 'state_dict' in checkpoint_data:
                        model_state_dict = checkpoint_data['state_dict']
                        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                        if missing_keys: print(f"警告 (折 {fold_idx + 1}): 加载权重时，模型中缺少以下键: {missing_keys}")
                        if unexpected_keys: print(f"警告 (折 {fold_idx + 1}): 加载权重时，检查点中存在模型不需要的额外键: {unexpected_keys}")
                        print(f"预训练权重加载成功 (折 {fold_idx + 1})。")
                    else:
                        print(f"警告 (折 {fold_idx + 1}): 检查点 {args.checkpoint} 中未找到 'state_dict'。将从头开始训练。")
                except FileNotFoundError:
                    print(f"错误 (折 {fold_idx + 1}): 检查点文件未找到: {args.checkpoint}。将从头开始训练。")
                except Exception as e:
                    print(f"错误 (折 {fold_idx + 1}): 加载检查点 {args.checkpoint} 时发生错误: {e}。将从头开始训练。")
                    traceback.print_exc()
            
            log_save_dir = Path(log_params.get('save_dir', 'logs'))
            project_name = log_params.get('project_name', 'DNAWhisper')
            fold_version = f"{log_params.get('experiment_name', 'run')}_fold_{fold_idx}"
            fold_log_dir = log_save_dir / project_name / fold_version

            loggers = [TensorBoardLogger(save_dir=str(log_save_dir), name=project_name, version=fold_version)]
            print(f"日志记录器 (折 {fold_idx}) 配置完成, 保存至: {fold_log_dir}")

            callbacks = []
            checkpoint_dir = Path(checkpoint_params.get('dirpath', 'checkpoints')) / fold_version
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=checkpoint_params.get('filename', '{epoch}-{val_loss:.4f}'),
                monitor=checkpoint_params.get('monitor', 'val_loss'),
                mode=checkpoint_params.get('mode', 'min'),
                save_top_k=checkpoint_params.get('save_top_k', 3),
                save_last=checkpoint_params.get('save_last', True),
                every_n_epochs=checkpoint_params.get('every_n_epochs', 1)
            )
            callbacks.append(checkpoint_callback)
            print(f"模型检查点 (折 {fold_idx}) 配置完成, 保存至: {checkpoint_dir}")

            callbacks.append(LearningRateMonitor(logging_interval='step'))
            print("设置学习率监控回调")

            if early_stopping_params.get("enabled", True):
                callbacks.append(EarlyStopping(
                    monitor=early_stopping_params.get("monitor", "val_loss"),
                    patience=early_stopping_params.get("patience", 15),
                    mode=early_stopping_params.get("mode", "min"),
                    min_delta=early_stopping_params.get("min_delta", 0.0001),
                    verbose=early_stopping_params.get("verbose", True)
                ))
                print(f"设置早停回调: monitor={early_stopping_params.get('monitor', 'val_loss')}, patience={early_stopping_params.get('patience', 15)}")

            if dynamic_optimizer_params.get("enabled", False):
                callbacks.append(DynamicTrainingCallback(dynamic_optimizer_params))
                print("设置动态训练优化器回调")

            if train_params.get('tensor_cores'):
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision(train_params['tensor_cores'])
                    print(f"设置 TensorFloat32 精度: {train_params['tensor_cores']}")
                else:
                    print("警告: 当前 PyTorch 版本不支持 torch.set_float32_matmul_precision")

            trainer = pl.Trainer(
                max_epochs=train_params.get("max_epochs", 100),
                accelerator=train_params.get("accelerator", "gpu"),
                devices=train_params.get("devices", [0]) if isinstance(train_params.get("devices"), list) else train_params.get("devices", 1),
                precision=trainer_precision,
                accumulate_grad_batches=train_params.get("accumulate_grad_batches", 1),
                gradient_clip_val=train_params.get("gradient_clip_val", 1.0),
                val_check_interval=train_params.get("val_check_interval", 1.0),
                log_every_n_steps=log_params.get("log_every_n_steps", 50),
                callbacks=callbacks,
                logger=loggers,
                # strategy=DDPStrategy(find_unused_parameters=True) if isinstance(train_params.get("devices", 1), list) and len(train_params.get("devices", 1)) > 1 else 'auto'
                # 修改为
                strategy=DDPStrategy(
                    find_unused_parameters=True,  # 修改为False，因为您的模型没有未使用的参数
                    gradient_as_bucket_view=True  # 添加此参数，避免梯度步幅不匹配问题
                ) if isinstance(train_params.get("devices", 1), list) and len(train_params.get("devices", [])) > 1 else 'auto'
            )
            print(f"Trainer (折 {fold_idx}) 实例化完成.")

            print(f"开始训练模型 (折 {fold_idx})...")
            start_time = time.time()
            trainer.fit(model, datamodule=datamodule)
            train_time = time.time() - start_time
            print(f"模型训练完成 (折 {fold_idx}), 耗时 {train_time/3600:.2f} 小时 ({train_time/60:.2f} 分钟)")

            print(f"开始测试模型 (折 {fold_idx})...")
            best_ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
            if best_ckpt_path:
                print(f"使用检查点进行测试: {best_ckpt_path}")
                test_results_list = trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path)
                if test_results_list:
                    print(f"模型测试完成 (折 {fold_idx}). 测试结果: {test_results_list[0]}")
                    results.append(test_results_list[0])
            else:
                 print(f"错误 (折 {fold_idx}): 无法找到任何检查点进行测试。跳过测试。")

            if trainer.is_global_zero and best_ckpt_path:
                print(f"\n开始加载最佳模型并保存预测结果 (折 {fold_idx}, Rank {trainer.global_rank})...")
                try:
                    model_to_predict = DNAWhisper.load_from_checkpoint(best_ckpt_path)
                    # phenotype_names_from_model = model_to_predict.phenotype_names # Already have phenotype_names from config
                    print(f"模型加载成功，将使用配置中的表型名称: {phenotype_names}")
                    
                    output_dir_preds = fold_log_dir # Save predictions in the fold's log directory
                    for pred_stage in ['train', 'val', 'test']:
                        print(f"  正在为阶段 '{pred_stage}' 保存预测和池化权重...")
                        save_predictions_and_weights(
                            model=model_to_predict, datamodule=datamodule, stage=pred_stage,
                            output_dir=output_dir_preds, phenotype_names=phenotype_names, # Use original config phenotype_names
                            data_loader_params=data_loader_params, collate_fn=collate_fn,
                            trainer_config=predict_trainer_config
                        )
                    print(f"所有预测和池化权重保存完成 (折 {fold_idx + 1})。")
                except Exception as e:
                    print(f"错误 (折 {fold_idx}, Rank {trainer.global_rank}): 在加载模型或保存预测结果时发生错误: {e}")
                    traceback.print_exc()
        
        print("\n--- K 折交叉验证完成 ---")
        if results:
            avg_results = {}
            all_keys = set().union(*(d.keys() for d in results if isinstance(d, dict)))
            for key in all_keys:
                valid_values = [res[key] for res in results if isinstance(res, dict) and key in res and isinstance(res[key], (int, float))]
                if valid_values: avg_results[key] = sum(valid_values) / len(valid_values)
            print("平均测试结果:")
            for key, value in avg_results.items(): print(f"  {key}: {value:.4f}")
    else: # Single run (no CV)
        print("未启用 K 折交叉验证，执行单次训练。")
        print("设置数据模块...")
        datamodule = WhisperDNADataModule_byCSV(
            config=training_config,
            model_config=model_config,
            phenotype_names=phenotype_names,
            seed=random_seed,
            logger=logging.getLogger("WhisperDNADataModule_byCSV")
        )
        datamodule.setup(stage='fit')
        print("DataModule setup 完成.")

        collate_fn = getattr(datamodule, 'collate_fn', None)
        if collate_fn: print("从 DataModule 获取 collate_fn。")
        else: print("警告: DataModule 未提供 collate_fn，将使用 DataLoader 默认值。")

        print("实例化模型...")
        model = DNAWhisper(
            config=model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        print("模型实例化完成.")

        if args.checkpoint:
            print(f"正在从检查点加载预训练权重: {args.checkpoint}")
            try:
                checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
                if 'state_dict' in checkpoint_data:
                    model_state_dict = checkpoint_data['state_dict']
                    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                    if missing_keys: print(f"警告: 加载权重时，模型中缺少以下键: {missing_keys}")
                    if unexpected_keys: print(f"警告: 加载权重时，检查点中存在模型不需要的额外键: {unexpected_keys}")
                    print("预训练权重加载成功。")
                else: print(f"警告: 检查点 {args.checkpoint} 中未找到 'state_dict'。将从头开始训练。")
            except FileNotFoundError: print(f"错误: 检查点文件未找到: {args.checkpoint}。将从头开始训练。")
            except Exception as e:
                print(f"错误: 加载检查点 {args.checkpoint} 时发生错误: {e}。将从头开始训练。")
                traceback.print_exc()

        log_save_dir = Path(log_params.get('save_dir', 'logs'))
        project_name = log_params.get('project_name', 'DNAWhisper')
        experiment_version = log_params.get('experiment_name', 'run_csv') # differentiate from h5 runs
        exp_log_dir = log_save_dir / project_name / experiment_version

        loggers = [TensorBoardLogger(save_dir=str(log_save_dir), name=project_name, version=experiment_version)]
        print(f"日志记录器配置完成, 保存至: {exp_log_dir}")

        callbacks = []
        checkpoint_dir = Path(checkpoint_params.get('dirpath', 'checkpoints')) / experiment_version
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=checkpoint_params.get('filename', '{epoch}-{val_loss:.4f}'),
            monitor=checkpoint_params.get('monitor', 'val_loss'),
            mode=checkpoint_params.get('mode', 'min'),
            save_top_k=checkpoint_params.get('save_top_k', 3),
            save_last=checkpoint_params.get('save_last', True),
            every_n_epochs=checkpoint_params.get('every_n_epochs', 1)
        )
        callbacks.append(checkpoint_callback)
        print(f"模型检查点配置完成, 保存至: {checkpoint_dir}")

        callbacks.append(LearningRateMonitor(logging_interval='step'))
        print("设置学习率监控回调")

        if early_stopping_params.get("enabled", True):
            callbacks.append(EarlyStopping(
                monitor=early_stopping_params.get("monitor", "val_loss"),
                patience=early_stopping_params.get("patience", 15),
                mode=early_stopping_params.get("mode", "min"),
                min_delta=early_stopping_params.get("min_delta", 0.0001),
                verbose=early_stopping_params.get("verbose", True)
            ))
            print(f"设置早停回调: monitor={early_stopping_params.get('monitor', 'val_loss')}, patience={early_stopping_params.get('patience', 15)}")

        if dynamic_optimizer_params.get("enabled", False):
            callbacks.append(DynamicTrainingCallback(dynamic_optimizer_params))
            print("设置动态训练优化器回调")
        
        if train_params.get('tensor_cores'):
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision(train_params['tensor_cores'])
                print(f"设置 TensorFloat32 精度: {train_params['tensor_cores']}")
            else:
                print("警告: 当前 PyTorch 版本不支持 torch.set_float32_matmul_precision")

        trainer = pl.Trainer(
            max_epochs=train_params.get("max_epochs", 100),
            accelerator=train_params.get("accelerator", "gpu"),
            devices=train_params.get("devices", [0]) if isinstance(train_params.get("devices"), list) else train_params.get("devices", 1),
            precision=trainer_precision,
            accumulate_grad_batches=train_params.get("accumulate_grad_batches", 1),
            gradient_clip_val=train_params.get("gradient_clip_val", 1.0),
            val_check_interval=train_params.get("val_check_interval", 1.0),
            log_every_n_steps=log_params.get("log_every_n_steps", 50),
            callbacks=callbacks,
            logger=loggers,
            # strategy=DDPStrategy(find_unused_parameters=True) if isinstance(train_params.get("devices", 1), list) and len(train_params.get("devices", 1)) > 1 else 'auto'
            strategy=DDPStrategy(
                find_unused_parameters=True,  # 修改为False，因为您的模型没有未使用的参数
                gradient_as_bucket_view=True  # 添加此参数，避免梯度步幅不匹配问题
            ) if isinstance(train_params.get("devices", 1), list) and len(train_params.get("devices", [])) > 1 else 'auto'
        )
        print(f"Trainer 实例化完成.")

        print("开始训练...")
        start_time = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_time = time.time() - start_time
        print(f"训练完成，耗时 {train_time/3600:.2f} 小时 ({train_time/60:.2f} 分钟)")

        print("使用最佳模型进行测试...")
        best_ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        if best_ckpt_path:
            print(f"使用检查点进行测试: {best_ckpt_path}")
            test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path)
            if test_results: print(f"模型测试完成. 测试结果: {test_results[0]}")
        else:
             print(f"错误: 无法找到任何检查点进行测试。跳过测试。")

        if trainer.is_global_zero and best_ckpt_path:
            print(f"\n开始加载最佳模型并保存预测结果 (Rank {trainer.global_rank})...")
            try:
                model_to_predict = DNAWhisper.load_from_checkpoint(best_ckpt_path)
                # phenotype_names_from_model = model_to_predict.phenotype_names
                print(f"模型加载成功，将使用配置中的表型名称: {phenotype_names}")

                output_dir_preds = exp_log_dir # Save predictions in the experiment's log directory
                for pred_stage in ['train', 'val', 'test']:
                    print(f"  正在为阶段 '{pred_stage}' 保存预测和池化权重...")
                    save_predictions_and_weights(
                        model=model_to_predict, datamodule=datamodule, stage=pred_stage,
                        output_dir=output_dir_preds, phenotype_names=phenotype_names, # Use original config phenotype_names
                        data_loader_params=data_loader_params, collate_fn=collate_fn,
                        trainer_config=predict_trainer_config
                    )
                print(f"所有预测和池化权重保存完成。")
            except Exception as e:
                print(f"错误 (Rank {trainer.global_rank}): 在加载模型或保存预测结果时发生错误: {e}")
                traceback.print_exc()

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" # Or configure as needed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    parser = argparse.ArgumentParser(description="DNAWhisper模型训练程序 (CSV 数据源)")

    parser.add_argument("--model-config", type=str, 
                        default="training/config/model_config_csv.json", # UPDATED
                        help="模型配置文件路径 (e.g., model_config_csv.json)")
    parser.add_argument("--training-config", type=str, 
                        default="training/config/training_config_csv.yml", # UPDATED
                        help="训练配置文件路径 (e.g., training_config_csv.yml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径（可选，用于恢复训练）")

    args = parser.parse_args()
    
    # Basic validation of config file existence
    if not Path(args.model_config).exists():
        print(f"错误: 模型配置文件未找到: {args.model_config}")
        sys.exit(1)
    if not Path(args.training_config).exists():
        print(f"错误: 训练配置文件未找到: {args.training_config}")
        sys.exit(1)

    train(args)

if __name__ == "__main__":
    main()

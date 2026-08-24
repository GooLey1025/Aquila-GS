import os
import numpy as np
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import KFold

class WhisperDNADataset_byCSV(Dataset):
    """
    Whisper of DNA 数据集 (从 CSV 文件加载)，用于处理 CSV 格式的 SNP 数据和表型数据。
    """

    def __init__(
        self,
        genotype_csv_path: Union[str, Path],
        phenotype_csv_path: Union[str, Path],
        indices: Optional[np.ndarray] = None,
        phenotype_names: Optional[List[str]] = None,
        normalize_phenotype: bool = True,
        phenotype_norm_method: str = "minmax",
        logger: Optional[logging.Logger] = None,
        block_length: Optional[int] = None,
        seed: Optional[int] = None,
        sample_id_col_geno: Optional[Union[int, str]] = None, # 基因型文件中的样本ID列名或索引
        sample_id_col_pheno: Optional[Union[int, str]] = None, # 表型文件中的样本ID列名或索引
    ):
        """
        初始化 WhisperDNADataset_byCSV

        Args:
            genotype_csv_path: 基因型 CSV 文件路径
            phenotype_csv_path: 表型 CSV 文件路径
            indices: 要使用的样本索引 (基于加载和对齐后的数据)，None 表示使用所有样本
            phenotype_names: 需要的表型名称列表
            normalize_phenotype: 是否对表型数据进行归一化
            phenotype_norm_method: 表型归一化方法 ("standard", "minmax")
            logger: 日志记录器
            block_length: 模型 embedding 层期望的块长度 (用于对齐 SNP 数量)
            seed: 用于随机操作的种子，确保可复现性
            sample_id_col_geno: 基因型文件中的样本ID列名或索引。如果为None，则假定没有ID列。
            sample_id_col_pheno: 表型文件中的样本ID列名或索引。如果为None，则假定没有ID列。
        """
        self.genotype_csv_path = Path(genotype_csv_path)
        self.phenotype_csv_path = Path(phenotype_csv_path)
        self.indices = indices
        self.phenotype_names = phenotype_names
        self.normalize_phenotype = normalize_phenotype
        self.phenotype_norm_method = phenotype_norm_method
        self.logger = logger or logging.getLogger("WhisperDNADataset_byCSV")
        self.block_length = block_length
        self.seed = seed
        self.sample_id_col_geno = sample_id_col_geno
        self.sample_id_col_pheno = sample_id_col_pheno

        self._phenotype_stats = None
        self.sample_ids = []

        # 添加索引追踪变量
        self.current_snp_indices_original = None
        self.final_kept_snp_indices_original = None

        self._load_data()

    def _load_data(self):
        """从 CSV 文件加载数据"""
        if not self.genotype_csv_path.exists():
            raise FileNotFoundError(f"基因型 CSV 文件不存在: {self.genotype_csv_path}")
        if not self.phenotype_csv_path.exists():
            raise FileNotFoundError(f"表型 CSV 文件不存在: {self.phenotype_csv_path}")

        self.logger.info(f"从 CSV 文件加载基因型数据: {self.genotype_csv_path}")
        df_geno = pd.read_csv(self.genotype_csv_path)
        
        if self.sample_id_col_geno is not None:
            geno_sample_ids = df_geno[self.sample_id_col_geno if isinstance(self.sample_id_col_geno, str) else df_geno.columns[self.sample_id_col_geno]].tolist()
            genotype_values = df_geno.drop(columns=[self.sample_id_col_geno if isinstance(self.sample_id_col_geno, str) else df_geno.columns[self.sample_id_col_geno]]).to_numpy()
        else:
            # 假设没有样本ID列，所有列都是特征
            geno_sample_ids = [f"geno_sample_{i}" for i in range(len(df_geno))]
            genotype_values = df_geno.to_numpy()
        
        self.n_samples_geno, self.n_snps_raw = genotype_values.shape
        # 假设每个 SNP 只有一个特征值，将其 reshape 为 [n_samples, n_snps, 1]
        self.genotype_data = genotype_values.reshape(self.n_samples_geno, self.n_snps_raw, 1)
        self.n_snps = self.n_snps_raw
        # 初始化原始SNP索引跟踪数组
        self.current_snp_indices_original = np.arange(self.n_snps_raw, dtype=int)
        self.logger.info(f"初始化原始SNP索引跟踪数组，大小: {len(self.current_snp_indices_original)}")
        self.logger.info(f"原始基因型数据形状: {genotype_values.shape}, 重塑后: {self.genotype_data.shape}")

        self.logger.info(f"从 CSV 文件加载表型数据: {self.phenotype_csv_path}")
        df_pheno = pd.read_csv(self.phenotype_csv_path)

        if self.sample_id_col_pheno is not None:
            pheno_sample_ids = df_pheno[self.sample_id_col_pheno if isinstance(self.sample_id_col_pheno, str) else df_pheno.columns[self.sample_id_col_pheno]].tolist()
            phenotype_values_df = df_pheno.drop(columns=[self.sample_id_col_pheno if isinstance(self.sample_id_col_pheno, str) else df_pheno.columns[self.sample_id_col_pheno]])
        else:
            pheno_sample_ids = [f"pheno_sample_{i}" for i in range(len(df_pheno))]
            phenotype_values_df = df_pheno

        self.all_phenotype_names = phenotype_values_df.columns.tolist()
        self.all_phenotypes = phenotype_values_df.to_numpy()
        self.n_samples_pheno = self.all_phenotypes.shape[0]
        self.logger.info(f"加载的表型名称: {self.all_phenotype_names}")
        self.logger.info(f"加载的表型数据形状: {self.all_phenotypes.shape}")

        # 检查样本数量和ID是否一致 (如果提供了ID)
        if self.sample_id_col_geno is not None and self.sample_id_col_pheno is not None:
            if set(geno_sample_ids) != set(pheno_sample_ids):
                self.logger.warning("基因型和表型样本ID不完全匹配。将尝试基于ID交集对齐。")
                # 实现基于ID的对齐逻辑 (复杂，此处简化为顺序匹配或报错)
                # For simplicity, if IDs are provided but don't match perfectly, we might error or take intersection.
                # Here, we'll assume if IDs are provided, they are used for alignment.
                # A robust solution would involve merging dataframes on IDs.
                # For now, we rely on the prompt's "按顺序对应" or raise error if IDs mismatch.
                if self.n_samples_geno != self.n_samples_pheno or geno_sample_ids != pheno_sample_ids: # Strict check if IDs are used
                     raise ValueError("基因型和表型样本ID不匹配或顺序不一致。请确保CSV文件中的样本ID对应。")
                self.sample_ids = geno_sample_ids
            else: # IDs match
                self.sample_ids = geno_sample_ids
                # Reorder df_pheno to match df_geno order if necessary (not done here, assumes order or perfect match)
        elif self.n_samples_geno != self.n_samples_pheno:
            raise ValueError(f"基因型样本数 ({self.n_samples_geno}) 与表型样本数 ({self.n_samples_pheno}) 不匹配。")
        else:
            # 无ID列，依赖顺序
            self.sample_ids = [f"sample_{i}" for i in range(self.n_samples_geno)]
        
        self.n_samples = self.n_samples_geno


        # --- 随机丢弃 SNP 以对齐 Block_length ---
        if self.block_length is not None and self.block_length > 0 and self.n_snps > 0:
            num_snps_to_drop = self.n_snps % self.block_length
            if num_snps_to_drop > 0:
                self.logger.info(f"当前 SNP 数量 ({self.n_snps}) 不是 Block_length ({self.block_length}) 的倍数。")
                self.logger.info(f"将随机丢弃 {num_snps_to_drop} 个 SNP。")

                if self.seed is not None:
                    random.seed(self.seed)
                    self.logger.debug(f"为 SNP 随机丢弃设置种子: {self.seed}")
                else:
                    self.logger.warning("未提供种子给 Dataset，SNP 随机丢弃将不可复现。")

                indices_to_drop = set(random.sample(range(self.n_snps), num_snps_to_drop))
                indices_to_keep_final = [i for i in range(self.n_snps) if i not in indices_to_drop]

                if not indices_to_keep_final:
                     self.logger.warning("随机丢弃后没有剩余的 SNP！请检查 Block_length 和 SNP 数量。")
                else:
                    self.genotype_data = self.genotype_data[:, indices_to_keep_final, :]
                    # 更新原始索引跟踪数组
                    self.current_snp_indices_original = self.current_snp_indices_original[indices_to_keep_final]
                    self.logger.info(f"随机丢弃后保留的原始SNP索引数量: {len(self.current_snp_indices_original)}")
                    self.n_snps = self.genotype_data.shape[1]
                    self.logger.info(f"随机丢弃 {num_snps_to_drop} 个 SNP 后，最终 SNP 数量: {self.n_snps}")
                    if self.n_snps % self.block_length != 0:
                         self.logger.error(f"错误：随机丢弃后 SNP 数量 ({self.n_snps}) 仍然不是 Block_length ({self.block_length}) 的倍数！")
            else:
                self.logger.info(f"当前 SNP 数量 ({self.n_snps}) 已是 Block_length ({self.block_length}) 的倍数，无需丢弃。")
        elif self.block_length is None:
             self.logger.info("未提供 Block_length，跳过 SNP 数量对齐步骤。")
        
        self.na_mask = np.isnan(self.all_phenotypes)
        self.logger.info(f"表型数据 NA 值掩码已生成。存在NA的比例: {np.sum(self.na_mask) / self.na_mask.size:.2%}")

        self._process_phenotypes()

        if self.indices is not None:
            self._apply_indices_filter()
       
        self.final_kept_snp_indices_original = self.current_snp_indices_original
        self.logger.info(f"最终保留的SNP原始索引数量: {len(self.final_kept_snp_indices_original)}")
        self._prepare_features()
 

    def _process_phenotypes(self):
        """处理和过滤表型数据"""
        if self.phenotype_names is None: # User did not specify, try to load from config or use all
            try:
                # Attempt to load from a standard model config if available (modify path as needed)
                model_config_path = Path(__file__).parent.parent.parent / "config" / "model_config_csv.json"
                if model_config_path.exists():
                    with open(model_config_path, 'r') as f:
                        model_config = json.load(f)
                    self.phenotype_names = model_config.get("output_layer", {}).get("phenotype_name", self.all_phenotype_names)
                    self.logger.info(f"从模型配置加载表型名称: {self.phenotype_names}")
                else:
                    self.logger.info("模型配置文件未找到，将使用 CSV 中的所有表型。")
                    self.phenotype_names = self.all_phenotype_names
            except Exception as e:
                self.logger.warning(f"读取模型配置失败: {e}, 使用 CSV 中的所有表型")
                self.phenotype_names = self.all_phenotype_names
        
        self.logger.info(f"期望匹配的表型名称: {self.phenotype_names}")
        self.logger.info(f"数据集中可用的表型名称: {self.all_phenotype_names}")

        phenotype_indices = []
        unmatched_phenotypes = []
        
        # Ensure self.phenotype_names is a list
        if isinstance(self.phenotype_names, str):
            self.phenotype_names = [self.phenotype_names]


        for name in self.phenotype_names:
            try:
                idx = self.all_phenotype_names.index(name)
                phenotype_indices.append(idx)
                self.logger.info(f"成功匹配表型: '{name}' 位于索引 {idx}")
            except ValueError:
                # Try case-insensitive match
                found_case_insensitive = False
                for i, dataset_name in enumerate(self.all_phenotype_names):
                    if isinstance(name, str) and isinstance(dataset_name, str) and name.lower() == dataset_name.lower():
                        phenotype_indices.append(i)
                        self.logger.info(f"不区分大小写匹配表型: '{name}' -> '{dataset_name}' 位于索引 {i}")
                        found_case_insensitive = True
                        break
                if not found_case_insensitive:
                    unmatched_phenotypes.append(name)
                    self.logger.warning(f"表型 '{name}' 不在数据集中，跳过")
        
        if unmatched_phenotypes:
            self.logger.warning(f"以下表型未能匹配: {unmatched_phenotypes}")

        self.phenotype_indices = phenotype_indices
        if len(phenotype_indices) > 0:
            self.phenotypes = self.all_phenotypes[:, phenotype_indices]
            self.logger.info(f"选择的表型数据形状: {self.phenotypes.shape}, 包含表型: {[self.all_phenotype_names[idx] for idx in phenotype_indices]}")
        else:
            self.logger.warning("未找到任何匹配的表型或未指定表型，将使用 CSV 中的所有表型。")
            self.phenotypes = self.all_phenotypes
            self.phenotype_indices = list(range(self.all_phenotypes.shape[1]))
            self.logger.info(f"使用全部表型: {len(self.phenotype_indices)} 个, 名称: {self.all_phenotype_names}")


        # Filter samples based on NA mask for selected phenotypes
        if self.na_mask is not None and self.phenotypes.size > 0:
            selected_na_mask = self.na_mask[:, self.phenotype_indices]
            valid_samples_mask = ~np.any(selected_na_mask, axis=1)

            num_valid = np.sum(valid_samples_mask)
            self.logger.info(f"过滤 NA 前的样本数: {self.n_samples}")
            self.logger.info(f"过滤 NA 后的有效样本数: {num_valid}/{self.n_samples}")

            if num_valid < self.n_samples:
                 self.logger.info(f"因NA值移除了 {self.n_samples - num_valid} 个样本。")
            if num_valid == 0 and self.n_samples > 0:
                self.logger.error("过滤 NA 后没有剩余样本！请检查表型数据。")
                # Keep original data if all samples would be removed, but log error
            elif num_valid < self.n_samples : # Apply filter only if some samples are removed but not all
                self.valid_sample_indices = np.where(valid_samples_mask)[0]
                self.genotype_data = self.genotype_data[valid_samples_mask]
                self.phenotypes = self.phenotypes[valid_samples_mask]
                original_sample_ids = self.sample_ids
                self.sample_ids = [original_sample_ids[i] for i in self.valid_sample_indices]
                self.n_samples = self.genotype_data.shape[0]
            else: # num_valid == self.n_samples (no NAs in selected or no NAs at all)
                self.valid_sample_indices = np.arange(self.n_samples)

            self.logger.info(f"过滤 NA 后的最终样本数: {self.n_samples}")
        else:
            self.valid_sample_indices = np.arange(self.n_samples)


    def _apply_indices_filter(self):
        """应用指定的样本索引过滤"""
        if hasattr(self, 'valid_sample_indices'):
            valid_indices_set = set(self.valid_sample_indices)
            filtered_indices = [idx for idx in self.indices if idx in valid_indices_set]
            idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(self.valid_sample_indices)}
            local_indices = [idx_map[idx] for idx in filtered_indices if idx in idx_map]
            self.logger.info(f"应用索引过滤 (基于NA过滤后)，从 {len(self.valid_sample_indices)} 样本中选择 {len(local_indices)} 样本")
        else: # No NA filtering was done or all samples were valid
            local_indices = [idx for idx in self.indices if 0 <= idx < self.n_samples]
            if len(local_indices) < len(self.indices):
                 self.logger.warning(f"提供的样本索引中有 {len(self.indices) - len(local_indices)} 个无效或超出范围。")
            self.logger.info(f"应用索引过滤，从 {self.n_samples} 样本中选择 {len(local_indices)} 样本")

        if not local_indices and len(self.indices) > 0:
            self.logger.error("索引过滤后没有剩余样本。请检查索引和数据。")
            # Avoid changing data if local_indices is empty to prevent errors, or handle as needed
            return

        self.genotype_data = self.genotype_data[local_indices]
        self.phenotypes = self.phenotypes[local_indices]
        original_sample_ids = self.sample_ids
        self.sample_ids = [original_sample_ids[i] for i in local_indices]
        self.n_samples = self.genotype_data.shape[0]
        self.logger.info(f"索引过滤后的最终样本数: {self.n_samples}")


    def _prepare_features(self):
        """准备输入特征"""
        self.normalized_genotype_data = self.genotype_data

        if self.phenotypes.size == 0:
            self.logger.warning("表型数据为空，无法进行归一化。")
            self.normalized_phenotypes = self.phenotypes
            self._phenotype_stats = None
            return

        if self.normalize_phenotype:
            if self.phenotype_norm_method == "standard":
                phenotype_mean = np.nanmean(self.phenotypes, axis=0)
                phenotype_std = np.nanstd(self.phenotypes, axis=0)
                phenotype_std = np.where(phenotype_std > 1e-8, phenotype_std, 1.0)
                self._phenotype_stats = {"method": "standard", "mean": phenotype_mean, "std": phenotype_std}
                self.normalized_phenotypes = (self.phenotypes - phenotype_mean) / phenotype_std
                self.logger.info(f"表型已使用 'standard' (z-score) 方法归一化。 Mean: {phenotype_mean}, Std: {phenotype_std}")
            elif self.phenotype_norm_method == "minmax":
                phenotype_min = np.nanmin(self.phenotypes, axis=0)
                phenotype_max = np.nanmax(self.phenotypes, axis=0)
                phenotype_range = phenotype_max - phenotype_min
                phenotype_range = np.where(phenotype_range > 1e-8, phenotype_range, 1.0)
                self._phenotype_stats = {"method": "minmax", "min": phenotype_min, "max": phenotype_max}
                self.normalized_phenotypes = (self.phenotypes - phenotype_min) / phenotype_range
                self.logger.info(f"表型已使用 'minmax' 方法归一化。 Min: {phenotype_min}, Max: {phenotype_max}")
            else:
                self.logger.warning(f"未知的 phenotype_norm_method: '{self.phenotype_norm_method}'. 表型将不会被归一化。")
                self.normalized_phenotypes = self.phenotypes
                self._phenotype_stats = None
        else:
            self.normalized_phenotypes = self.phenotypes
            self._phenotype_stats = None
            self.logger.info("表型归一化已禁用。")
        
        # Handle any NaNs that might remain after normalization (e.g. if a column was all NaNs)
        if np.isnan(self.normalized_phenotypes).any():
            self.logger.warning("归一化后的表型数据中仍存在NaN值，将替换为0。")
            self.normalized_phenotypes = np.nan_to_num(self.normalized_phenotypes, nan=0.0)


        self.logger.info("输入特征准备完成")

    # 在WhisperDNADataset_byCSV类中添加新方法：
    def save_snp_indices_to_file(self, file_path):
        """
        将最终保留的SNP原始索引保存到文件
        
        Args:
            file_path: 保存索引的文件路径
        """
        if self.final_kept_snp_indices_original is None:
            self.logger.warning("没有可保存的SNP索引信息")
            return False
        
        try:
            # 创建父目录（如果不存在）
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存索引到文件
            np.savetxt(file_path, self.final_kept_snp_indices_original, fmt='%d')
            
            self.logger.info(f"成功将{len(self.final_kept_snp_indices_original)}个保留的SNP索引保存到: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存SNP索引到文件时出错: {e}")
            return False

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.n_samples}")

        snp_data = self.normalized_genotype_data[idx]    # Shape: [n_snps, 1]
        phenotype_data = self.normalized_phenotypes[idx] # Shape: [n_phenotypes]

        features_tensor = torch.FloatTensor(snp_data).contiguous()
        phenotype_tensor = torch.FloatTensor(phenotype_data).contiguous()

        return {
            "features": features_tensor,
            "phenotype": phenotype_tensor,
            "sample_idx": idx # This is local index within this specific dataset split
        }

class WhisperDNADataModule_byCSV(pl.LightningDataModule):
    """Whisper of DNA 数据模块 (从 CSV 加载)，用于PyTorch Lightning训练"""

    def __init__(
        self,
        # genotype_csv_path: Union[str, Path],
        # phenotype_csv_path: Union[str, Path],
        config: Dict[str, Any],
        model_config: Dict[str, Any], # 包含模型配置的字典 (embedding, GFI_FormerBLOCKS etc.)
        phenotype_names: Optional[List[str]], # 从模型配置中提取的表型名称列表
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        # self.genotype_csv_path = Path(genotype_csv_path)
        # self.phenotype_csv_path = Path(phenotype_csv_path)
        self.config = config
        self.model_config = model_config
        self.seed = seed
        self.logger = logger or logging.getLogger("WhisperDNADataModule_byCSV")
        self._phenotype_names_config = phenotype_names # Store names from model_config

        data_config = config.get('data', {}) # Ensure data_config is defined early
        
        # Get CSV paths from config
        geno_csv_path_str = data_config.get('genotype_csv_path')
        pheno_csv_path_str = data_config.get('phenotype_csv_path')

        if not geno_csv_path_str:
            raise ValueError("datamodule_bycsv: training_config.yml 中 data.genotype_csv_path 未配置")
        if not pheno_csv_path_str:
            raise ValueError("datamodule_bycsv: training_config.yml 中 data.phenotype_csv_path 未配置")

        self.genotype_csv_path = Path(geno_csv_path_str)
        self.phenotype_csv_path = Path(pheno_csv_path_str)

        embedding_config = self.model_config.get('embedding', {})
        self.block_length = embedding_config.get('Block_length')
        if self.block_length is None:
            self.logger.warning("模型配置中未找到 embedding.Block_length，无法执行 SNP 数量对齐。")
        elif not isinstance(self.block_length, int) or self.block_length <= 0:
            self.logger.warning(f"模型配置中的 embedding.Block_length ({self.block_length}) 不是正整数。")
            self.block_length = None
        else:
            self.logger.info(f"从模型配置中读取 Block_length: {self.block_length}")

        data_config = config.get('data', {})
        self.train_batch_size = data_config.get('batch_size', 32)
        self.val_batch_size = data_config.get('val_batch_size', self.train_batch_size)
        # Add test_batch_size if it's in your config, otherwise default like val
        self.test_batch_size = data_config.get('test_batch_size', self.val_batch_size) 
        self.num_workers = data_config.get('num_workers', 4)
        self.pin_memory = data_config.get('pin_memory', True)
        self.shuffle_train = data_config.get('shuffle', True)
        
        self.normalize_phenotype = data_config.get('normalize_phenotype', True)
        self.phenotype_norm_method = data_config.get('phenotype_norm_method', "minmax")
        self.sample_id_col_geno = data_config.get('sample_id_col_geno', None)
        self.sample_id_col_pheno = data_config.get('sample_id_col_pheno', None)


        split_config = config.get('training', {})
        self.train_ratio = split_config.get('train_ratio', 0.7)
        self.val_ratio = split_config.get('val_ratio', 0.15)
        self.test_ratio = split_config.get('test_ratio', 0.15)
        if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            self.logger.warning(f"Train ({self.train_ratio}) + Val ({self.val_ratio}) + Test ({self.test_ratio}) ratios do not sum to 1.0. Normalizing.")
            total = self.train_ratio + self.val_ratio + self.test_ratio
            if total > 0:
                self.train_ratio /= total
                self.val_ratio /= total
                self.test_ratio = 1.0 - self.train_ratio - self.val_ratio
            else: # Avoid division by zero if all are zero
                self.logger.error("All split ratios are zero. Setting to default 70/15/15.")
                self.train_ratio, self.val_ratio, self.test_ratio = 0.7, 0.15, 0.15


        self.use_cv_folds = split_config.get('use_cv_folds', False)
        self.cv_n_splits = split_config.get('cv_n_splits', 5)
        self.cv_fold_idx = split_config.get('cv_fold_idx', 0)

        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.test_indices: Optional[np.ndarray] = None
        
        self.dataset: Optional[WhisperDNADataset_byCSV] = None
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None


    def prepare_data(self):
        if not self.genotype_csv_path.exists():
            raise FileNotFoundError(f"基因型 CSV 文件不存在: {self.genotype_csv_path}")
        if not self.phenotype_csv_path.exists():
            raise FileNotFoundError(f"表型 CSV 文件不存在: {self.phenotype_csv_path}")

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None: # Load main dataset only once
            self.logger.info(f"首次设置 (stage: {stage}), 从 CSV 加载数据集")
            self.dataset = WhisperDNADataset_byCSV(
                genotype_csv_path=self.genotype_csv_path, # Uses path from config
                phenotype_csv_path=self.phenotype_csv_path, # Uses path from config
                phenotype_names=self._phenotype_names_config, # Pass names from model_config
                normalize_phenotype=self.normalize_phenotype,
                phenotype_norm_method=self.phenotype_norm_method,
                logger=self.logger,
                block_length=self.block_length,
                seed=self.seed,
                sample_id_col_geno=self.sample_id_col_geno, # Already read from config
                sample_id_col_pheno=self.sample_id_col_pheno  # Already read from config
            )
            self._prepare_splits()

            # 在创建训练/验证/测试集之前，保存SNP索引
            if hasattr(self.dataset, 'final_kept_snp_indices_original') and self.dataset.final_kept_snp_indices_original is not None:
                # 构建日志文件名（使用数据集名称和时间戳）
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_name = self.genotype_csv_path.stem
                
                # 构建保存路径
                log_dir = Path(self.config.get('logging', {}).get('save_dir', ''))
                project_name = self.config.get('logging', {}).get('project_name', '')
                experiment_name = self.config.get('logging', {}).get('experiment_name', '')
                save_path = log_dir / project_name / experiment_name
                save_path.mkdir(parents=True, exist_ok=True)
                snp_indices_file = save_path / f"{csv_name}_kept_snp_indices_{timestamp}.txt"
                
                # 保存SNP索引
                self.dataset.save_snp_indices_to_file(snp_indices_file)
                
                # 记录一些SNP索引的基本信息到日志
                total_snps = self.dataset.genotype_data.shape[1]  # 当前SNP数量
                self.logger.info(f"SNP统计: 已保存{len(self.dataset.final_kept_snp_indices_original)}个SNP索引 (总SNP: {total_snps})")
                if len(self.dataset.final_kept_snp_indices_original) > 0:
                    self.logger.info(f"SNP索引范围: 从{self.dataset.final_kept_snp_indices_original.min()}到{self.dataset.final_kept_snp_indices_original.max()}")

            if self.train_indices is not None and len(self.train_indices) > 0:
                self.train_dataset = Subset(self.dataset, self.train_indices)
            elif self.train_indices is not None: # Empty but not None
                 self.logger.warning("训练索引为空，train_dataset 将为空。")
                 self.train_dataset = Subset(self.dataset, []) # Empty subset
            
            if self.val_indices is not None and len(self.val_indices) > 0:
                self.val_dataset = Subset(self.dataset, self.val_indices)
            elif self.val_indices is not None:
                 self.logger.warning("验证索引为空，val_dataset 将为空。")
                 self.val_dataset = Subset(self.dataset, [])

            if self.test_indices is not None and len(self.test_indices) > 0:
                self.test_dataset = Subset(self.dataset, self.test_indices)
            elif self.test_indices is not None:
                 self.logger.warning("测试索引为空，test_dataset 将为空。")
                 self.test_dataset = Subset(self.dataset, [])
        else:
            self.logger.info(f"数据集已加载，跳过重新加载 (stage: {stage})")


    def _prepare_splits(self):
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized in setup before calling _prepare_splits.")

        n_samples_total = len(self.dataset) # Number of samples after Dataset's internal filtering (NA, etc.)
        self.logger.info(f"数据集总样本数 (Dataset内部过滤后): {n_samples_total}")

        if n_samples_total == 0:
            self.logger.error("数据集中没有样本，无法进行拆分。")
            self.train_indices = np.array([])
            self.val_indices = np.array([])
            self.test_indices = np.array([])
            return

        # These are indices for the self.dataset (which is already filtered)
        local_indices = np.arange(n_samples_total) 
        np.random.seed(self.seed)
        np.random.shuffle(local_indices)

        # Calculate sizes ensuring they sum up correctly and are integers
        test_size = int(np.floor(n_samples_total * self.test_ratio))
        val_size = int(np.floor(n_samples_total * self.val_ratio))
        
        # Ensure train_size is not negative if ratios are small or sum > 1 due to floor
        train_size = n_samples_total - test_size - val_size
        if train_size < 0: train_size = 0

        # Handle cases where splits might be too small or zero
        if train_size == 0 and self.train_ratio > 0: self.logger.warning("训练集大小为0，但比例大于0。")
        if val_size == 0 and self.val_ratio > 0: self.logger.warning("验证集大小为0，但比例大于0。")
        if test_size == 0 and self.test_ratio > 0: self.logger.warning("测试集大小为0，但比例大于0。")


        initial_train_indices = local_indices[:train_size]
        initial_val_indices = local_indices[train_size : train_size + val_size]
        self.test_indices = local_indices[train_size + val_size :]
        
        self.logger.info(f"初始随机拆分: 训练 {len(initial_train_indices)}, 验证 {len(initial_val_indices)}, 测试 {len(self.test_indices)}")

        if self.use_cv_folds:
            self.logger.info(f"启用 K 折交叉验证 (K={self.cv_n_splits}), 选择折索引: {self.cv_fold_idx}")
            train_val_indices = np.concatenate([initial_train_indices, initial_val_indices])

            if len(train_val_indices) < self.cv_n_splits:
                self.logger.warning(f"训练+验证集样本数 ({len(train_val_indices)}) 小于 K ({self.cv_n_splits})。无法执行 K 折。将使用初始训练/验证拆分。")
                self.train_indices = initial_train_indices
                self.val_indices = initial_val_indices
            else:
                kf = KFold(n_splits=self.cv_n_splits, shuffle=True, random_state=self.seed)
                try:
                    folds = list(kf.split(train_val_indices)) # kf.split needs at least 1D array
                    if not (0 <= self.cv_fold_idx < self.cv_n_splits):
                        self.logger.error(f"指定的 CV 折索引 {self.cv_fold_idx} 超出范围 [0, {self.cv_n_splits-1}]。将使用初始拆分。")
                        self.train_indices = initial_train_indices
                        self.val_indices = initial_val_indices
                    else:
                        train_fold_local_indices, val_fold_local_indices = folds[self.cv_fold_idx]
                        self.train_indices = train_val_indices[train_fold_local_indices]
                        self.val_indices = train_val_indices[val_fold_local_indices]
                        self.logger.info(f"CV 折 {self.cv_fold_idx}: 训练 {len(self.train_indices)}, 验证 {len(self.val_indices)}")
                except ValueError as e:
                    self.logger.error(f"KFold拆分出错: {e}. 可能由于样本数不足。将使用初始拆分。")
                    self.train_indices = initial_train_indices
                    self.val_indices = initial_val_indices
        else:
            self.train_indices = initial_train_indices
            self.val_indices = initial_val_indices
            self.logger.info("未使用 K 折交叉验证。")

        # Final checks
        for name, indices in [("训练", self.train_indices), ("验证", self.val_indices), ("测试", self.test_indices)]:
            if indices is None:
                self.logger.error(f"{name}集索引为 None！")
            elif len(indices) == 0:
                 if (name == "训练" and self.train_ratio > 0) or \
                    (name == "验证" and self.val_ratio > 0) or \
                    (name == "测试" and self.test_ratio > 0):
                    self.logger.warning(f"最终{name}集为空！")


    def train_dataloader(self):
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.warning("train_dataset 未设置或为空，返回一个空的 DataLoader。")
            return DataLoader([], batch_size=self.train_batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True 
        )

    def val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            self.logger.warning("val_dataset 未设置或为空，返回一个空的 DataLoader。")
            return DataLoader([], batch_size=self.val_batch_size)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        if self.test_dataset is None or len(self.test_dataset) == 0:
            self.logger.warning("test_dataset 未设置或为空，返回一个空的 DataLoader。")
            return DataLoader([], batch_size=self.test_batch_size)
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    @property
    def num_phenotypes(self) -> Optional[int]:
        if self.dataset is not None and hasattr(self.dataset, 'normalized_phenotypes') and self.dataset.normalized_phenotypes is not None:
            return self.dataset.normalized_phenotypes.shape[1]
        return None

    @property
    def feature_dim(self) -> Optional[int]:
        """返回每个SNP的特征维度 (对于CSV，假设为1)"""
        if self.dataset is not None and hasattr(self.dataset, 'genotype_data') and self.dataset.genotype_data is not None:
            if self.dataset.genotype_data.ndim == 3:
                return self.dataset.genotype_data.shape[2] # Should be 1 for current CSV assumption
            self.logger.error("无法从 genotype_data 获取特征维度，维度不为3。")
        return None
    
    @property
    def num_snps(self) -> Optional[int]:
        """返回SNP的数量"""
        if self.dataset is not None and hasattr(self.dataset, 'n_snps'):
            return self.dataset.n_snps
        return None

    @property
    def phenotype_names_from_dataset(self) -> Optional[List[str]]:
        """返回数据集中实际使用的表型名称"""
        if self.dataset is not None and hasattr(self.dataset, 'phenotype_indices') and \
           self.dataset.phenotype_indices is not None and hasattr(self.dataset, 'all_phenotype_names'):
            try:
                return [self.dataset.all_phenotype_names[i] for i in self.dataset.phenotype_indices]
            except IndexError:
                self.logger.error("获取数据集表型名称时索引错误。")
                return self._phenotype_names_config # Fallback
        return self._phenotype_names_config # Fallback

    def get_normalized_stats(self) -> Optional[Dict[str, Any]]:
        if self.dataset is not None and hasattr(self.dataset, '_phenotype_stats') and self.dataset._phenotype_stats is not None:
            return {"phenotype": self.dataset._phenotype_stats}
        self.logger.warning("无法获取表型归一化统计信息，Dataset 或 _phenotype_stats 未初始化。")
        return None

import os
import h5py
import numpy as np
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import random

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import KFold
try:
    from .mic_filter import MICFilter
except ImportError:
    MICFilter = None


class WhisperDNADataset(Dataset):
    """Whisper of DNA 数据集，用于处理预处理后的 SNP 数据"""

    def __init__(
        self,
        h5_file_path: Union[str, Path],
        indices: Optional[np.ndarray] = None,
        phenotype_names: Optional[List[str]] = None,
        normalize_position: bool = True,
        normalize_phenotype: bool = True,
        position_encoding_method: str = "transformer",
        phenotype_norm_method: str = "minmax", # <--- 新增：表型归一化方法
        logger: Optional[logging.Logger] = None,
        snp_indices_to_keep: Optional[np.ndarray] = None,
        block_length: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化 WhisperDNADataset

        Args:
            h5_file_path: HDF5 文件路径
            indices: 要使用的样本索引，None 表示使用所有样本
            phenotype_names: 需要的表型名称列表
            normalize_position: 是否对位置编码进行归一化
            normalize_phenotype: 是否对表型数据进行归一化
            position_encoding_method: 位置编码归一化方法
            phenotype_norm_method: 表型归一化方法 ("standard", "minmax")
            logger: 日志记录器
            snp_indices_to_keep: 要保留的 SNP 索引 (来自 MIC 筛选等)
            block_length: 模型 embedding 层期望的块长度
            seed: 用于随机操作的种子，确保可复现性
        """
        self.h5_file_path = Path(h5_file_path)
        self.indices = indices
        self.phenotype_names = phenotype_names
        self.normalize_position = normalize_position
        self.normalize_phenotype = normalize_phenotype
        self.position_encoding_method = position_encoding_method
        self.phenotype_norm_method = phenotype_norm_method # <--- 新增
        self.logger = logger or logging.getLogger("WhisperDNADataset")
        self.snp_indices_to_keep = snp_indices_to_keep
        self.block_length = block_length
        self.seed = seed

        self._position_stats = None
        self._phenotype_stats = None

        self._load_data()

    def _load_data(self):
        """从 HDF5 文件加载数据"""
        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {self.h5_file_path}")

        self.logger.info(f"从文件加载数据: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, 'r') as f:
            genotype_data = f['features/genotype_features'][:]
            self.logger.info(f"原始基因型数据形状: {genotype_data.shape} [n_snps, n_samples, 10]")
            if genotype_data.ndim != 3 or genotype_data.shape[2] != 10:
                self.logger.warning(f"基因型数据形状不是预期的 [n_snps, n_samples, 10]，而是 {genotype_data.shape}。请确认数据格式。")

            position_data = f['features/position_features'][:]
            self.logger.info(f"位置特征形状: {position_data.shape}")

            phenotypes = f['phenotypes'][:]
            all_phenotype_names = []
            for name_bytes in f['phenotype_names'][:]:
                if isinstance(name_bytes, bytes):
                    name = name_bytes.decode('utf-8')
                elif isinstance(name_bytes, np.ndarray):
                    name = name_bytes.item().decode('utf-8') if hasattr(name_bytes.item(), 'decode') else str(name_bytes.item())
                else:
                    name = str(name_bytes)
                all_phenotype_names.append(name)
            self.logger.info(f"读取到的表型名称: {all_phenotype_names}")

            sample_ids = list(f['sample_ids'][:])

            has_na_mask = 'phenotypes_na_mask' in f
            if has_na_mask:
                na_mask = f['phenotypes_na_mask'][:]
                self.logger.info(f"表型数据存在 NA 值掩码")
            else:
                na_mask = None
                self.logger.info(f"表型数据不存在 NA 值掩码")

        self.genotype_data = np.transpose(genotype_data, (1, 0, 2))
        self.n_samples, self.n_snps = self.genotype_data.shape[:2]
        self.logger.info(f"基因型数据已转置: {self.genotype_data.shape} [n_samples, n_snps, 10]")

        self.position_data = position_data
        if self.position_data.shape[0] != self.n_snps:
            self.logger.warning(f"转置后的 SNP 数量 ({self.n_snps}) 与位置特征数量 ({self.position_data.shape[0]}) 不匹配！")
        self.position_dim = self.position_data.shape[1]
        self.logger.info(f"位置特征维度: {self.position_dim}")

        # --- 应用 MIC 筛选 (如果启用) ---
        if self.snp_indices_to_keep is not None:
            original_n_snps = self.n_snps
            try:
                valid_snp_indices = [idx for idx in self.snp_indices_to_keep if 0 <= idx < original_n_snps]
                num_invalid = len(self.snp_indices_to_keep) - len(valid_snp_indices)
                if num_invalid > 0:
                    self.logger.warning(f"提供的 SNP 索引中有 {num_invalid} 个无效索引被忽略。")

                if not valid_snp_indices:
                    raise ValueError("没有有效的 SNP 索引可供保留。")

                self.logger.info(f"应用 SNP 索引过滤，从 {original_n_snps} SNPs 中保留 {len(valid_snp_indices)} SNPs")
                self.genotype_data = self.genotype_data[:, valid_snp_indices, :]
                self.position_data = self.position_data[valid_snp_indices, :]
                self.n_snps = self.genotype_data.shape[1] # 更新 n_snps
                self.logger.info(f"过滤后基因型数据形状: {self.genotype_data.shape}")
                self.logger.info(f"过滤后位置数据形状: {self.position_data.shape}")
            except Exception as e:
                self.logger.error(f"应用 SNP 索引过滤时出错: {e}. 未执行 SNP 过滤。")
        # --- MIC 筛选结束 ---

        # --- 新增：随机丢弃 SNP 以对齐 Block_length ---
        if self.block_length is not None and self.block_length > 0 and self.n_snps > 0:
            num_snps_to_drop = self.n_snps % self.block_length
            if num_snps_to_drop > 0:
                self.logger.info(f"当前 SNP 数量 ({self.n_snps}) 不是 Block_length ({self.block_length}) 的倍数。")
                self.logger.info(f"将随机丢弃 {num_snps_to_drop} 个 SNP。")

                # --- 设置随机种子 ---
                if self.seed is not None:
                    random.seed(self.seed)
                    self.logger.debug(f"为 SNP 随机丢弃设置种子: {self.seed}")
                else:
                    self.logger.warning("未提供种子给 Dataset，SNP 随机丢弃将不可复现。")
                # --- 结束设置种子 ---

                # 生成要丢弃的 SNP 的 *局部* 索引 (在当前 n_snps 范围内)
                indices_to_drop = set(random.sample(range(self.n_snps), num_snps_to_drop))

                # 生成要保留的 SNP 的 *局部* 索引
                indices_to_keep_final = [i for i in range(self.n_snps) if i not in indices_to_drop]

                if not indices_to_keep_final:
                     self.logger.warning("随机丢弃后没有剩余的 SNP！请检查 Block_length 和 SNP 数量。")
                else:
                    # 应用最终的 SNP 索引
                    self.genotype_data = self.genotype_data[:, indices_to_keep_final, :]
                    self.position_data = self.position_data[indices_to_keep_final, :]
                    original_n_snps_before_drop = self.n_snps
                    self.n_snps = self.genotype_data.shape[1] # 更新最终的 n_snps
                    self.logger.info(f"随机丢弃 {num_snps_to_drop} 个 SNP 后，最终 SNP 数量: {self.n_snps}")
                    if self.n_snps % self.block_length != 0:
                         self.logger.error(f"错误：随机丢弃后 SNP 数量 ({self.n_snps}) 仍然不是 Block_length ({self.block_length}) 的倍数！")

            else:
                self.logger.info(f"当前 SNP 数量 ({self.n_snps}) 已是 Block_length ({self.block_length}) 的倍数，无需丢弃。")
        elif self.block_length is None:
             self.logger.info("未提供 Block_length，跳过 SNP 数量对齐步骤。")
        # --- 随机丢弃结束 ---

        self.all_phenotype_names = all_phenotype_names
        self.all_phenotypes = phenotypes
        self.na_mask = na_mask
        self.sample_ids = sample_ids

        self._process_phenotypes()

        if self.indices is not None:
            self._apply_indices_filter()

        if self.normalize_position:
            self._normalize_position_data()

        self._prepare_features()

    def _process_phenotypes(self):
        """处理和过滤表型数据"""
        if self.phenotype_names is None:
            try:
                model_config_path = Path(__file__).parent.parent.parent / "config" / "model_config.json"
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                self.phenotype_names = model_config["output_layer"]["phenotype_name"]
                self.logger.info(f"从模型配置加载表型名称: {self.phenotype_names}")
            except Exception as e:
                self.logger.warning(f"读取模型配置失败: {e}, 使用默认表型名称")
                self.phenotype_names = self.all_phenotype_names[:3]

        self.logger.info(f"待匹配的表型名称: {self.phenotype_names}")
        self.logger.info(f"数据集中的表型名称: {self.all_phenotype_names}")

        phenotype_indices = []
        unmatched_phenotypes = []

        for name in self.phenotype_names:
            if name in self.all_phenotype_names:
                idx = self.all_phenotype_names.index(name)
                phenotype_indices.append(idx)
                self.logger.info(f"成功匹配表型: '{name}' 位于索引 {idx}")
            else:
                found = False
                for i, dataset_name in enumerate(self.all_phenotype_names):
                    if isinstance(name, str) and isinstance(dataset_name, str) and name.lower() == dataset_name.lower():
                        phenotype_indices.append(i)
                        self.logger.info(f"不区分大小写匹配表型: '{name}' -> '{dataset_name}' 位于索引 {i}")
                        found = True
                        break

                if not found:
                    unmatched_phenotypes.append(name)
                    self.logger.warning(f"表型 '{name}' 不在数据集中，跳过")

        if unmatched_phenotypes:
            self.logger.warning(f"以下表型未能匹配: {unmatched_phenotypes}")
            self.logger.info(f"数据集中可用的表型: {self.all_phenotype_names}")

        self.phenotype_indices = phenotype_indices
        if len(phenotype_indices) > 0:
            self.phenotypes = self.all_phenotypes[:, phenotype_indices]
            self.logger.info(f"选择的表型数据形状: {self.phenotypes.shape}, 包含表型: {[self.all_phenotype_names[idx] for idx in phenotype_indices]}")
        else:
            self.logger.warning("未找到任何匹配的表型，使用全部表型")
            self.phenotypes = self.all_phenotypes
            self.phenotype_indices = list(range(self.all_phenotypes.shape[1]))
            self.logger.info(f"使用全部表型: {len(self.phenotype_indices)} 个")

        if self.na_mask is not None:
            selected_na_mask = self.na_mask[:, self.phenotype_indices]
            valid_samples = ~np.any(selected_na_mask, axis=1)

            num_valid = np.sum(valid_samples)
            self.logger.info(f"过滤 NA 后的有效样本数: {num_valid}/{self.n_samples}")

            if num_valid < 0.5 * self.n_samples:
                self.logger.warning(f"过滤 NA 后样本数减少了 {self.n_samples - num_valid} ({(self.n_samples - num_valid) / self.n_samples:.1%})")

            self.valid_sample_indices = np.where(valid_samples)[0]
            self.genotype_data = self.genotype_data[valid_samples]
            self.phenotypes = self.phenotypes[valid_samples]
            self.n_samples = self.genotype_data.shape[0]
            self.logger.info(f"过滤 NA 后的样本数: {self.n_samples}")
        else:
            self.valid_sample_indices = np.arange(self.n_samples)

    def _apply_indices_filter(self):
        """应用指定的样本索引过滤"""
        if hasattr(self, 'valid_sample_indices'):
            valid_indices_set = set(self.valid_sample_indices)
            filtered_indices = [idx for idx in self.indices if idx in valid_indices_set]

            idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(self.valid_sample_indices)}
            local_indices = [idx_map[idx] for idx in filtered_indices if idx in idx_map]

            self.logger.info(f"应用索引过滤，从 {self.n_samples} 样本中选择 {len(local_indices)} 样本")

            self.genotype_data = self.genotype_data[local_indices]
            self.phenotypes = self.phenotypes[local_indices]
            self.n_samples = self.genotype_data.shape[0]
        else:
            self.logger.info(f"应用索引过滤，从 {self.n_samples} 样本中选择 {len(self.indices)} 样本")

            valid_indices = [idx for idx in self.indices if 0 <= idx < self.n_samples]
            if len(valid_indices) < len(self.indices):
                self.logger.warning(f"有 {len(self.indices) - len(valid_indices)} 个无效索引被忽略")

            self.genotype_data = self.genotype_data[valid_indices]
            self.phenotypes = self.phenotypes[valid_indices]
            self.n_samples = self.genotype_data.shape[0]

    def _normalize_position_data(self):
        """对位置编码数据进行归一化，特别处理第2、3维语境归一化"""
        position_data = self.position_data.copy()

        unique_chroms = np.unique(position_data[:, 0])

        for chrom in unique_chroms:
            chrom_mask = (position_data[:, 0] == chrom)

            if self.position_encoding_method == "transformer":
                pos_values = position_data[chrom_mask, 1]
                if len(pos_values) > 0:
                    min_pos = np.min(pos_values)
                    max_pos = np.max(pos_values)
                    normalized_pos = (pos_values - min_pos) / max(max_pos - min_pos, 1)
                    position_data[chrom_mask, 1] = np.sin(normalized_pos * (2 * np.pi))

                center_values = position_data[chrom_mask, 2]
                if len(center_values) > 0:
                    min_center = np.min(center_values)
                    max_center = np.max(center_values)
                    normalized_center = (center_values - min_center) / max(max_center - min_center, 1)
                    position_data[chrom_mask, 2] = np.sin(normalized_center * (2 * np.pi))
            else:
                for dim in [1, 2]:
                    values = position_data[chrom_mask, dim]
                    if len(values) > 0:
                        mean = np.mean(values)
                        std = np.std(values)
                        if std > 1e-8:  # Add epsilon for safety
                            position_data[chrom_mask, dim] = (values - mean) / std

        for dim in range(3, self.position_dim):
            if dim == 5:
                self.logger.info(f"Skipping normalization for binary feature dimension {dim} (Is Coding Region).")
                continue

            values = position_data[:, dim]
            mean = np.mean(values)
            std = np.std(values)
            if std > 1e-8:  # Add epsilon for safety
                position_data[:, dim] = (values - mean) / std

        self.normalized_position_data = position_data
        self.logger.info("完成位置编码归一化 (已跳过维度 5)")

    def _prepare_features(self):
        """准备输入特征，将SNP数据和位置编码组合"""
        self.normalized_genotype_data = self.genotype_data

        if self.normalize_phenotype:
            if self.phenotype_norm_method == "standard":
                phenotype_mean = np.mean(self.phenotypes, axis=0)
                phenotype_std = np.std(self.phenotypes, axis=0)
                # Avoid division by zero for phenotypes with no variance
                phenotype_std = np.where(phenotype_std > 1e-8, phenotype_std, 1.0)

                self._phenotype_stats = {"method": "standard", "mean": phenotype_mean, "std": phenotype_std}
                self.normalized_phenotypes = (self.phenotypes - phenotype_mean) / phenotype_std
                self.logger.info(f"表型已使用 'standard' (z-score) 方法归一化。 Mean: {phenotype_mean}, Std: {phenotype_std}")
            elif self.phenotype_norm_method == "minmax":
                phenotype_min = np.min(self.phenotypes, axis=0)
                phenotype_max = np.max(self.phenotypes, axis=0)
                phenotype_range = phenotype_max - phenotype_min
                # Avoid division by zero if range is 0 (all values are the same)
                phenotype_range = np.where(phenotype_range > 1e-8, phenotype_range, 1.0)

                self._phenotype_stats = {"method": "minmax", "min": phenotype_min, "max": phenotype_max}
                # If range is 1.0 (because it was 0), all normalized values will be 0.
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

        if not hasattr(self, 'normalized_position_data') and self.normalize_position:
            self.logger.warning("normalize_position is True, but normalized_position_data not found. Using original position data.")
            self.position_data_to_use = self.position_data
        elif self.normalize_position:
            self.position_data_to_use = self.normalized_position_data
        else:
            self.position_data_to_use = self.position_data

        self.logger.info("输入特征准备完成")

    def __len__(self):
        """返回数据集中的样本数量"""
        return self.n_samples

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        snp_data = self.normalized_genotype_data[idx] # Shape: [n_snps, 10]
        phenotype_data = self.normalized_phenotypes[idx] # Shape: [n_phenotypes]

        position_data = self.position_data_to_use # Shape: [n_snps, position_dim]

        # Remove dimension 4 (SNP density) from position_data
        # Create indices for all dimensions EXCEPT dimension 4
        dims_to_keep = np.arange(position_data.shape[1]) != 4
        position_data_filtered = position_data[:, dims_to_keep] # Shape: [n_snps, position_dim - 1]

        # Concatenate SNP data with the FILTERED position data
        features = np.concatenate([snp_data, position_data_filtered], axis=1) # Shape: [n_snps, 10 + position_dim - 1]

        features_tensor = torch.FloatTensor(features)
        phenotype_tensor = torch.FloatTensor(phenotype_data)

        return {
            "features": features_tensor,
            "phenotype": phenotype_tensor,
            "sample_idx": idx
        }


class WhisperDNADataModule(pl.LightningDataModule):
    """Whisper of DNA 数据模块，用于PyTorch Lightning训练"""

    def __init__(
        self,
        h5_file_path: Union[str, Path],
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        phenotype_names: Optional[List[str]],
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化 WhisperDNADataModule

        Args:
            h5_file_path: HDF5 文件路径
            config: 包含训练配置的字典 (training, data, mic_filtering etc.)
            model_config: 包含模型配置的字典 (embedding, GFI_FormerBLOCKS etc.)
            phenotype_names: 从模型配置中提取的表型名称列表
            seed: 随机种子
            logger: 日志记录器
        """
        super().__init__()
        self.h5_file_path = Path(h5_file_path)
        self.config = config
        self.model_config = model_config
        self.seed = seed
        self.logger = logger or logging.getLogger("WhisperDNADataModule")
        self._phenotype_names_config = phenotype_names

        # --- 从 model_config 获取 Block_length ---
        embedding_config = self.model_config.get('embedding', {})
        self.block_length = embedding_config.get('Block_length')
        if self.block_length is None:
            self.logger.warning("模型配置中未找到 embedding.Block_length，无法执行 SNP 数量对齐。")
        elif not isinstance(self.block_length, int) or self.block_length <= 0:
            self.logger.warning(f"模型配置中的 embedding.Block_length ({self.block_length}) 不是正整数，无法执行 SNP 数量对齐。")
            self.block_length = None # 设为 None 以禁用对齐
        else:
            self.logger.info(f"从模型配置中读取 Block_length: {self.block_length}")
        # --- 结束 ---

        data_config = config.get('data', {})
        self.train_batch_size = data_config.get('batch_size', 32)
        self.val_batch_size = data_config.get('val_batch_size', self.train_batch_size)
        self.test_batch_size = data_config.get('test_batch_size', self.val_batch_size)
        self.num_workers = data_config.get('num_workers', 4)
        self.pin_memory = data_config.get('pin_memory', True)
        self.shuffle_train = data_config.get('shuffle', True)

        self.normalize_position = data_config.get('normalize_position', True)
        self.normalize_phenotype = data_config.get('normalize_phenotype', True)
        self.position_encoding_method = data_config.get('position_encoding_method', "transformer")
        self.phenotype_norm_method = data_config.get('phenotype_norm_method', "standard") # <--- 新增

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
            else:
                self.logger.error("All split ratios are zero. Setting to default 70/15/15.")
                self.train_ratio, self.val_ratio, self.test_ratio = 0.7, 0.15, 0.15

        self.use_cv_folds = split_config.get('use_cv_folds', False)
        self.cv_n_splits = split_config.get('cv_n_splits', 5)
        self.cv_fold_idx = split_config.get('cv_fold_idx', 0)

        mic_config = config.get('mic_filtering', {})
        self.mic_enabled = mic_config.get('enabled', False)
        self.mic_file_path = mic_config.get('mic_file_path', None)
        self.mic_filter_ratio = mic_config.get('filter_ratio', 0.1)
        self.mic_phenotypes = mic_config.get('phenotypes_to_consider', None)

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.dataset = None
        # --- 添加：初始化数据集属性 ---
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # --- 结束添加 ---

    def prepare_data(self):
        """检查数据是否存在，这个方法在单个GPU上运行"""
        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {self.h5_file_path}")
        if self.mic_enabled and (not self.mic_file_path or not Path(self.mic_file_path).exists()):
            self.logger.warning(f"MIC filtering enabled but mic_file_path '{self.mic_file_path}' is invalid or missing. MIC filtering will be skipped.")

    def setup(self, stage: Optional[str] = None):
        """加载数据并设置训练/验证/测试数据集拆分"""
        # --- 修改：确保即使 stage 不同，数据集也会被创建和存储 ---
        if self.dataset is None: # 只有在主数据集未加载时才执行加载和拆分
            self.logger.info(f"首次设置 (stage: {stage}), 加载数据集")

            snp_indices_to_keep = None
            if self.mic_enabled:
                if MICFilter is None:
                    self.logger.error("MIC filtering enabled, but MICFilter class could not be imported. Skipping MIC filtering.")
                elif not self.mic_file_path or not Path(self.mic_file_path).exists():
                    self.logger.error(f"MIC filtering enabled but mic_file_path '{self.mic_file_path}' is invalid or missing. Skipping MIC filtering.")
                else:
                    try:
                        self.logger.info(f"Applying MIC filtering using file: {self.mic_file_path}")
                        mic_filter = MICFilter(mic_file_path=self.mic_file_path, logger=self.logger)
                        phenotypes_for_mic = self.mic_phenotypes if self.mic_phenotypes else self._phenotype_names_config
                        if not phenotypes_for_mic:
                            self.logger.warning("MIC filtering enabled, but no phenotypes specified (mic_filtering.phenotypes_to_consider or model phenotypes). Skipping MIC filtering.")
                        else:
                            self.logger.info(f"Filtering SNPs based on phenotypes: {phenotypes_for_mic} with ratio(s): {self.mic_filter_ratio}")
                            snp_indices_to_keep = mic_filter.filter_snps(
                                phenotype_names_to_use=phenotypes_for_mic,
                                filter_ratios=self.mic_filter_ratio
                            )
                            self.logger.info(f"MIC filtering selected {len(snp_indices_to_keep)} SNPs.")
                    except Exception as e:
                        self.logger.error(f"Error during MIC filtering: {e}. Skipping MIC filtering.")
                        snp_indices_to_keep = None

            self.dataset = WhisperDNADataset(
                h5_file_path=self.h5_file_path,
                phenotype_names=self._phenotype_names_config,
                normalize_position=self.normalize_position,
                normalize_phenotype=self.normalize_phenotype,
                position_encoding_method=self.position_encoding_method,
                phenotype_norm_method=self.phenotype_norm_method, # <--- 新增
                logger=self.logger,
                snp_indices_to_keep=snp_indices_to_keep,
                block_length=self.block_length,
                seed=self.seed
            )

            self._prepare_splits()

            # --- 添加：创建并存储 Subset 对象 ---
            if self.train_indices is not None and len(self.train_indices) > 0:
                self.train_dataset = Subset(self.dataset, self.train_indices)
                self.logger.info(f"已创建并存储 train_dataset (大小: {len(self.train_dataset)})")
            else:
                self.logger.warning("训练索引为空或未定义，train_dataset 将为 None。")
                self.train_dataset = None

            if self.val_indices is not None and len(self.val_indices) > 0:
                self.val_dataset = Subset(self.dataset, self.val_indices)
                self.logger.info(f"已创建并存储 val_dataset (大小: {len(self.val_dataset)})")
            else:
                self.logger.warning("验证索引为空或未定义，val_dataset 将为 None。")
                self.val_dataset = None

            if self.test_indices is not None and len(self.test_indices) > 0:
                self.test_dataset = Subset(self.dataset, self.test_indices)
                self.logger.info(f"已创建并存储 test_dataset (大小: {len(self.test_dataset)})")
            else:
                self.logger.warning("测试索引为空或未定义，test_dataset 将为 None。")
                self.test_dataset = None
            # --- 结束添加 ---
        else:
             self.logger.info(f"数据集已加载，跳过重新加载 (stage: {stage})")
        # --- 结束修改 ---

    def _prepare_splits(self):
        """
        准备数据集拆分。
        1. 获取过滤 NA 后的样本总数。
        2. 生成局部索引 (0 到 n-1)。
        3. 执行随机 Train/Val/Test 划分。
        4. 如果启用 CV，则在 Train+Val 集上执行 K-Fold 划分，并选择指定折。
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized in setup before calling _prepare_splits.")

        n_samples_after_na = len(self.dataset)
        self.logger.info(f"数据集总样本数 (过滤 NA 后): {n_samples_after_na}")

        if n_samples_after_na == 0:
            self.logger.error("过滤 NA 后没有剩余样本，无法进行拆分。")
            self.train_indices = np.array([])
            self.val_indices = np.array([])
            self.test_indices = np.array([])
            return

        local_indices = np.arange(n_samples_after_na)
        np.random.seed(self.seed)
        np.random.shuffle(local_indices)

        test_size = int(np.floor(n_samples_after_na * self.test_ratio))
        val_size = int(np.floor(n_samples_after_na * self.val_ratio))
        train_size = n_samples_after_na - test_size - val_size

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            self.logger.warning(f"计算的拆分大小包含零或负数 (train={train_size}, val={val_size}, test={test_size})。请检查比例或样本数量。将尝试调整。")
            if n_samples_after_na > 0:
                test_size = max(1, test_size) if self.test_ratio > 0 else 0
                val_size = max(1, val_size) if self.val_ratio > 0 else 0
                if n_samples_after_na - test_size - val_size <= 0:
                    test_size = max(0, test_size - 1) if test_size > 0 else 0
                    if n_samples_after_na - test_size - val_size <= 0 and val_size > 0:
                        val_size = max(0, val_size - 1)
                train_size = n_samples_after_na - test_size - val_size
                if train_size <= 0:
                    self.logger.error("无法创建有效的训练集拆分。所有集合将为空。")
                    train_size, val_size, test_size = 0, 0, 0

        initial_train_indices = local_indices[:train_size]
        initial_val_indices = local_indices[train_size: train_size + val_size]
        self.test_indices = local_indices[train_size + val_size:]

        self.logger.info(f"初始随机拆分: 训练 {len(initial_train_indices)}, 验证 {len(initial_val_indices)}, 测试 {len(self.test_indices)}")

        if self.use_cv_folds:
            self.logger.info(f"启用 K 折交叉验证 (K={self.cv_n_splits}), 选择折索引: {self.cv_fold_idx}")

            train_val_indices = np.concatenate([initial_train_indices, initial_val_indices])

            if len(train_val_indices) < self.cv_n_splits:
                self.logger.warning(f"训练+验证集样本数 ({len(train_val_indices)}) 小于 K ({self.cv_n_splits})。无法执行 K 折交叉验证。将使用初始的训练/验证拆分。")
                self.train_indices = initial_train_indices
                self.val_indices = initial_val_indices
            else:
                kf = KFold(n_splits=self.cv_n_splits, shuffle=True, random_state=self.seed)
                folds = list(kf.split(train_val_indices))

                if 0 <= self.cv_fold_idx < self.cv_n_splits:
                    train_fold_local_indices, val_fold_local_indices = folds[self.cv_fold_idx]
                    self.train_indices = train_val_indices[train_fold_local_indices]
                    self.val_indices = train_val_indices[val_fold_local_indices]
                    self.logger.info(f"CV 折 {self.cv_fold_idx}: 训练 {len(self.train_indices)}, 验证 {len(self.val_indices)}")
                else:
                    self.logger.error(f"指定的 CV 折索引 {self.cv_fold_idx} 超出范围 [0, {self.cv_n_splits-1}]。将使用初始的训练/验证拆分。")
                    self.train_indices = initial_train_indices
                    self.val_indices = initial_val_indices
        else:
            self.train_indices = initial_train_indices
            self.val_indices = initial_val_indices
            self.logger.info("未使用 K 折交叉验证。")

        if self.train_indices is None or self.val_indices is None or self.test_indices is None:
            raise RuntimeError("未能成功生成训练、验证或测试索引。")
        if len(self.train_indices) == 0:
            self.logger.warning("最终训练集为空！")
        if len(self.val_indices) == 0:
            self.logger.warning("最终验证集为空！")
        if len(self.test_indices) == 0 and self.test_ratio > 0:
            self.logger.warning("最终测试集为空！")
        if len(set(self.train_indices) & set(self.val_indices)) > 0:
            self.logger.warning("训练集和验证集之间存在重叠索引！")
        if len(set(self.train_indices) & set(self.test_indices)) > 0:
            self.logger.warning("训练集和测试集之间存在重叠索引！")
        if len(set(self.val_indices) & set(self.test_indices)) > 0:
            self.logger.warning("验证集和测试集之间存在重叠索引！")

    def train_dataloader(self):
        """返回训练数据加载器"""
        # --- 修改：使用 self.train_dataset ---
        if self.train_dataset is None:
            self.logger.warning("train_dataset 未设置或为空，返回一个空的 DataLoader。")
            return DataLoader([]) # Return empty DataLoader
        # --- 结束修改 ---
        return DataLoader(
            # --- 修改：使用 self.train_dataset ---
            self.train_dataset,
            # --- 结束修改 ---
            batch_size=self.train_batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        """返回验证数据加载器"""
        # --- 修改：使用 self.val_dataset ---
        if self.val_dataset is None:
            self.logger.warning("val_dataset 未设置或为空，返回一个空的 DataLoader。")
            return DataLoader([]) # Return empty DataLoader
        # --- 结束修改 ---
        return DataLoader(
            # --- 修改：使用 self.val_dataset ---
            self.val_dataset,
            # --- 结束修改 ---
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        """返回测试数据加载器"""
        # --- 修改：使用 self.test_dataset ---
        if self.test_dataset is None:
            self.logger.warning("test_dataset 未设置或为空，返回一个空的 DataLoader。")
            return DataLoader([]) # Return empty DataLoader
        # --- 结束修改 ---
        return DataLoader(
            # --- 修改：使用 self.test_dataset ---
            self.test_dataset,
            # --- 结束修改 ---
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    @property
    def num_phenotypes(self):
        """返回表型数量"""
        if self.dataset is not None and hasattr(self.dataset, 'normalized_phenotypes'):
            if self.dataset.normalized_phenotypes is not None:
                return self.dataset.normalized_phenotypes.shape[1]
        return None

    @property
    def feature_dim(self):
        """返回特征维度 (OneHot SNP(10) + Position - 1)"""
        if self.dataset is not None:
            # Subtract 1 because we removed the SNP density feature (dim 4)
            return 10 + self.dataset.position_dim - 1
        return None

    @property
    def phenotype_names(self):
        """返回表型名称"""
        # Try to get names from the dataset instance first
        if self.dataset is not None and hasattr(self.dataset, 'phenotype_indices') and self.dataset.phenotype_indices is not None:
            # Map selected indices back to names from the full list
            return [self.dataset.all_phenotype_names[i] for i in self.dataset.phenotype_indices]
        # Fallback to the names provided during initialization
        elif hasattr(self, '_phenotype_names_config'):
            return self._phenotype_names_config
        return None

    @phenotype_names.setter
    def phenotype_names(self, names):
        self._phenotype_names_config = names
        # If dataset already exists, potentially re-setup or warn?
        if self.dataset is not None:
            self.logger.warning("Phenotype names set after dataset initialization. Re-run setup() for changes to take effect.")

    def get_normalized_stats(self):
        """获取归一化统计信息，用于推理阶段"""
        if self.dataset is not None:
            stats = {}
            if hasattr(self.dataset, '_phenotype_stats') and self.dataset._phenotype_stats is not None:
                stats["phenotype"] = self.dataset._phenotype_stats
            else:
                self.logger.warning("无法获取表型归一化统计信息，Dataset 或 _phenotype_stats 未初始化。")
            
            # Position stats might be needed if normalization was applied
            # For now, we only focus on phenotype stats as per the request
            # if hasattr(self.dataset, '_position_stats') and self.dataset._position_stats is not None:
            #     stats["position"] = self.dataset._position_stats
            
            return stats if stats else None
        
        self.logger.warning("无法获取归一化统计信息，Dataset 未初始化。")
        return None

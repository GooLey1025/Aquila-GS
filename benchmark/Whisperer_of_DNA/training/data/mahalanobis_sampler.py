import numpy as np
import torch
from torch.utils.data import Sampler
from sklearn.covariance import MinCovDet
from typing import List, Optional, Iterator

class StratifiedMahalanobisSampler(Sampler):  # 移除 [int] 类型参数
    def __init__(self,
                 dataset,  # 不要显式指定类型为 torch.utils.data.Dataset
                 phenotype_data: np.ndarray,
                 num_strata: int,
                 batch_size: int, # 虽然sampler本身不直接用batch_size决定迭代，但可用于启发式
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 phenotype_norm_method: Optional[str] = "standard"):
        """
        基于表型数据的马氏距离进行分层采样。

        Args:
            dataset: PyTorch Dataset 对象。
            phenotype_data: NumPy 数组，形状为 (n_samples, n_phenotypes)，
                            包含用于分层的表型数据。这些数据应对应 `dataset` 中的样本。
            num_strata: 要划分的层数。
            batch_size: DataLoader的批次大小，可用于未来启发式调整。
            shuffle: 是否在每个 epoch 开始时打乱样本顺序。
            seed: 随机种子。
            phenotype_norm_method: 表型归一化方法 ('standard', 'minmax', None)。
        """
        # 不调用 super().__init__(dataset)，而是直接初始化
        self.dataset = dataset
        self.length = len(dataset)  # 保留一个长度属性
        self.phenotype_data = phenotype_data
        self.num_strata = num_strata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.phenotype_norm_method = phenotype_norm_method
        
        if self.phenotype_data.shape[0] != len(self.dataset):
            raise ValueError("phenotype_data 的样本数必须与 dataset 中的样本数一致。")
        if self.num_strata <= 0:
            raise ValueError("num_strata 必须是正整数。")

        self.epoch = 0 # 用于在每个 epoch 改变随机性（如果需要）
        self._strata_indices: List[List[int]] = [] # 存储每个层的样本索引

        self._prepare_strata()

    def _normalize_phenotypes(self) -> np.ndarray:
        """对表型数据进行归一化。"""
        data = self.phenotype_data.astype(np.float32) # 确保是浮点数
        if self.phenotype_norm_method == "standard":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std < 1e-8] = 1.0 # 避免除以零
            return (data - mean) / std
        elif self.phenotype_norm_method == "minmax":
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            range_val = max_val - min_val
            range_val[range_val < 1e-8] = 1.0 # 避免除以零
            return (data - min_val) / range_val
        elif self.phenotype_norm_method is None:
            return data
        else:
            raise ValueError(f"未知的 phenotype_norm_method: {self.phenotype_norm_method}")

    def _prepare_strata(self):
        """计算马氏距离并进行分层。"""
        normalized_phenotypes = self._normalize_phenotypes()
        
        # 计算马氏距离
        # 使用稳健的协方差估计 MinCovDet，对异常值更鲁棒
        try:
            # 如果样本量远大于特征数，直接用 np.cov 可能也行
            # 但如果特征数接近或大于样本数，或存在共线性，需要更稳健的方法
            if normalized_phenotypes.shape[0] > normalized_phenotypes.shape[1]:
                cov_estimator = MinCovDet(random_state=self.seed).fit(normalized_phenotypes)
                inv_cov = cov_estimator.get_precision()
                mean = cov_estimator.location_
            else: # 样本少于特征数，或为了简单起见，使用单位阵（即欧氏距离的平方）或对角阵
                print("警告: 样本数少于表型特征数，或为简化，马氏距离计算可能退化。")
                inv_cov = np.eye(normalized_phenotypes.shape[1])
                mean = np.mean(normalized_phenotypes, axis=0)

            diff = normalized_phenotypes - mean
            # Mahalanobis distance squared: (x-mu)^T * Sigma^-1 * (x-mu)
            mahalanobis_dist_sq = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        except Exception as e:
            print(f"计算马氏距离时出错: {e}。将退回到使用第一个表型特征进行分层。")
            # 回退策略：例如，基于第一个表型特征的值进行分层
            if normalized_phenotypes.shape[1] > 0:
                mahalanobis_dist_sq = normalized_phenotypes[:, 0]
            else: # 如果没有表型数据，则无法分层，退化为随机采样
                print("警告: 没有可用的表型数据进行分层，将执行标准随机采样。")
                self._strata_indices = [list(range(len(self.dataset)))] # 所有样本都在一个层
                self.num_strata = 1
                return


        # 根据马氏距离分层 (例如，使用分位数)
        quantiles = np.linspace(0, 100, self.num_strata + 1)
        strata_bins = np.percentile(mahalanobis_dist_sq, quantiles)
        strata_bins[0] = -np.inf #确保包含最小值
        strata_bins[-1] = np.inf #确保包含最大值
        
        # 避免重复的边界值导致空层
        unique_bins = []
        if len(strata_bins) > 0:
            unique_bins.append(strata_bins[0])
            for i in range(1, len(strata_bins)):
                if strata_bins[i] > strata_bins[i-1]: # 只有当边界值增加时才添加
                    unique_bins.append(strata_bins[i])
        
        if len(unique_bins) <= 1: # 如果所有值都一样，无法分层
            print("警告: 表型数据分布无法有效分层，所有样本归为一层。")
            self._strata_indices = [list(range(len(self.dataset)))]
            self.num_strata = 1
            return

        strata_bins = np.array(unique_bins)
        self.num_strata = len(strata_bins) - 1 # 更新实际层数

        sample_stratum_assignment = np.digitize(mahalanobis_dist_sq, strata_bins[1:-1], right=False)
        
        self._strata_indices = [[] for _ in range(self.num_strata)]
        for i, stratum_idx in enumerate(sample_stratum_assignment):
            self._strata_indices[stratum_idx].append(i)

        # 移除空层
        self._strata_indices = [s_indices for s_indices in self._strata_indices if len(s_indices) > 0]
        self.num_strata = len(self._strata_indices)
        if self.num_strata == 0: # 不太可能发生，但作为保险
             print("警告: 分层后没有有效的层，所有样本归为一层。")
             self._strata_indices = [list(range(len(self.dataset)))]
             self.num_strata = 1


    def __iter__(self) -> Iterator[int]:
        # 使用 self.seed 和 self.epoch 来确保每个 epoch 的随机性是可控和不同的
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed + self.epoch)
        else:
            g.manual_seed(torch.initial_seed() + self.epoch) # Fallback

        indices_in_epoch = []
        
        # 如果 shuffle，打乱每个层内部的样本顺序
        current_strata_indices = []
        for stratum_idx_list in self._strata_indices:
            temp_list = list(stratum_idx_list) # 复制列表
            if self.shuffle:
                # 使用 torch.randperm 实现可复现的打乱
                perm = torch.randperm(len(temp_list), generator=g).tolist()
                current_strata_indices.append([temp_list[i] for i in perm])
            else:
                current_strata_indices.append(temp_list)

        # 轮询从各层抽取样本，直到所有样本都被抽取
        # 这种策略试图在整个 epoch 中均匀混合来自不同层的样本
        stratum_pointers = [0] * self.num_strata
        num_yielded_total = 0
        
        # 如果 shuffle，打乱层的抽取顺序
        strata_fetch_order = torch.randperm(self.num_strata, generator=g).tolist() if self.shuffle else list(range(self.num_strata))

        while num_yielded_total < len(self.dataset):
            made_progress_this_round = False
            for stratum_idx_in_fetch_order in strata_fetch_order:
                # 映射回真实的层索引（如果 fetch_order 被打乱）
                # stratum_id = strata_fetch_order[stratum_idx_in_fetch_order] # 不对，strata_fetch_order 本身就是索引
                stratum_id = stratum_idx_in_fetch_order

                if stratum_pointers[stratum_id] < len(current_strata_indices[stratum_id]):
                    sample_original_idx = current_strata_indices[stratum_id][stratum_pointers[stratum_id]]
                    indices_in_epoch.append(sample_original_idx)
                    stratum_pointers[stratum_id] += 1
                    num_yielded_total += 1
                    made_progress_this_round = True
            
            if not made_progress_this_round and num_yielded_total < len(self.dataset):
                # 如果一轮下来没有取到任何样本（例如所有非空层都已取完，但总数未达到）
                # 这通常不应该发生，除非 self.dataset 长度与各层样本总数不符
                # 或者某些层是空的但 num_strata 没有正确更新
                print(f"警告: 采样轮询中未取得进展，但仍有 {len(self.dataset) - num_yielded_total} 样本未采样。")
                break # 避免死循环

        if len(indices_in_epoch) != len(self.dataset):
            # 如果由于某种原因，生成的索引数量不等于数据集大小，
            # 可能是分层逻辑或上述循环有缺陷。
            # 作为回退，如果启用了 shuffle，则返回一个随机排列。
            print(f"警告: 生成的 epoch 索引数量 ({len(indices_in_epoch)}) 与数据集大小 ({len(self.dataset)}) 不匹配。")
            if self.shuffle:
                indices_in_epoch = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices_in_epoch = list(range(len(self.dataset)))
        
        self.epoch += 1
        return iter(indices_in_epoch)

    def __len__(self) -> int:
        return len(self.dataset)

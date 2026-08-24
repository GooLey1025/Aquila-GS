import logging
from pathlib import Path
from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd


class MICFilterError(Exception):
    """MICFilter specific errors."""
    pass


class MICFilter:
    """
    根据表型与SNP之间的最大互信息系数 (MIC) 筛选SNP。

    该类读取一个包含SNP信息和多个表型MIC值的文件，
    并根据指定的表型和筛选比例，选出与这些表型最相关的SNP子集。
    """

    def __init__(self, mic_file_path: Path, logger: Optional[logging.Logger] = None):
        """
        初始化 MICFilter。

        Args:
            mic_file_path: MIC 文件的路径。文件应包含SNP信息，
                           以及形如 'phenotype_mic' 的列表示MIC值。
                           第一行应为列标题。
            logger: 用于记录日志的日志记录器实例。
        """
        self.mic_file_path = Path(mic_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.mic_dataframe: Optional[pd.DataFrame] = None
        self.phenotype_mic_columns: List[str] = []
        self.n_total_snps: int = 0

        self._load_and_validate_mic_file()

    def _load_and_validate_mic_file(self):
        """加载并验证 MIC 文件。"""
        if not self.mic_file_path.is_file():
            msg = f"MIC 文件未找到: {self.mic_file_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            self.logger.info(f"开始加载 MIC 文件: {self.mic_file_path}")
            # 读取数据，假设第一行是表头
            self.mic_dataframe = pd.read_csv(self.mic_file_path, sep='\t', header=0)
            self.n_total_snps = len(self.mic_dataframe) # 行数即为SNP数

            if self.n_total_snps == 0:
                 msg = f"MIC 文件为空或未能正确解析: {self.mic_file_path}"
                 self.logger.error(msg)
                 raise MICFilterError(msg)

            self.logger.info(f"成功加载 MIC 文件，总共包含 {self.n_total_snps} 个 SNP。")

            # 识别所有可能的表型MIC列
            self.phenotype_mic_columns = [col for col in self.mic_dataframe.columns if col.endswith('_mic')]

            if not self.phenotype_mic_columns:
                msg = f"在 MIC 文件 {self.mic_file_path} 中未找到任何以 '_mic' 结尾的列。"
                self.logger.error(msg)
                raise MICFilterError(msg)

            self.logger.debug(f"在 MIC 文件中找到的 MIC 列: {self.phenotype_mic_columns}")

        except pd.errors.EmptyDataError:
            msg = f"MIC 文件为空: {self.mic_file_path}"
            self.logger.error(msg)
            raise MICFilterError(msg)
        except Exception as e:
            msg = f"加载或解析 MIC 文件时出错 {self.mic_file_path}: {e}"
            self.logger.exception(msg) # 使用 exception 记录堆栈跟踪
            raise MICFilterError(msg) from e

    def filter_snps(self, phenotype_names_to_use: List[str], filter_ratios: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        根据指定的表型和比例筛选 SNP。

        对于每个指定的表型，根据提供的比例选择 MIC 值最高的 SNP。
        最终返回所有被选中的 SNP 索引的并集（去重）。

        Args:
            phenotype_names_to_use: 需要考虑进行筛选的表型名称列表。
            filter_ratios: 每个表型要筛选的 SNP 比例。
                           - 如果是 float: 同一个比例应用于所有有效表型。
                           - 如果是 List[float] 或 np.ndarray: 列表/数组的长度必须
                             与 `phenotype_names_to_use` 中有效表型的数量一致，
                             每个元素对应一个表型的筛选比例。

        Returns:
            一个 NumPy 数组，包含最终筛选出的、去重且排序后的 SNP 索引。

        Raises:
            MICFilterError: 如果输入的表型名称无效或筛选比例无效。
            ValueError: 如果 filter_ratios 中的值超出 [0, 1] 范围，或者列表/数组
                        长度与有效表型数量不匹配。
        """
        if self.mic_dataframe is None:
             msg = "MIC 数据尚未加载，无法执行筛选。"
             self.logger.error(msg)
             raise MICFilterError(msg)

        # 验证输入的表型名称是否有效
        valid_phenotypes = []
        invalid_phenotypes = []
        original_indices_map = {} # 存储有效表型在原始列表中的索引
        for i, name in enumerate(phenotype_names_to_use):
            mic_col = f"{name}_mic"
            if mic_col in self.phenotype_mic_columns:
                valid_phenotypes.append(name)
                original_indices_map[name] = i # 记录原始索引
            else:
                invalid_phenotypes.append(name)

        if invalid_phenotypes:
            self.logger.warning(f"以下请求的表型在 MIC 文件中没有对应的 '_mic' 列，将被忽略: {invalid_phenotypes}")

        if not valid_phenotypes:
            msg = "没有提供任何有效的表型名称用于 MIC 筛选。"
            self.logger.error(msg)
            raise MICFilterError(msg)

        # 处理和验证 filter_ratios
        ratios_to_use = {}
        if isinstance(filter_ratios, float):
            # 单一比例应用于所有有效表型
            if not (0.0 <= filter_ratios <= 1.0):
                msg = f"筛选比例 filter_ratios 必须在 [0.0, 1.0] 之间，但得到的是: {filter_ratios}"
                self.logger.error(msg)
                raise ValueError(msg)
            if filter_ratios == 0.0:
                 self.logger.warning("筛选比例为 0.0，将不会筛选任何 SNP，返回空索引列表。")
                 return np.array([], dtype=int)
            for name in valid_phenotypes:
                ratios_to_use[name] = filter_ratios
            self.logger.info(f"开始基于以下表型进行 MIC 筛选: {valid_phenotypes}，统一筛选比例: {filter_ratios:.2%}")

        elif isinstance(filter_ratios, (list, np.ndarray)):
            # 检查列表/数组长度是否与 *原始请求的* 表型列表匹配
            if len(filter_ratios) != len(phenotype_names_to_use):
                 msg = (f"当 filter_ratios 是列表/数组时，其长度 ({len(filter_ratios)}) "
                        f"必须与请求的表型列表长度 ({len(phenotype_names_to_use)}) 匹配。")
                 self.logger.error(msg)
                 raise ValueError(msg)

            # 检查每个比例值是否有效，并映射到有效表型
            has_zero_ratio = False
            for name in valid_phenotypes:
                original_index = original_indices_map[name]
                ratio = filter_ratios[original_index]
                if not (0.0 <= ratio <= 1.0):
                    msg = (f"筛选比例 filter_ratios[{original_index}] (对应表型 '{name}') "
                           f"必须在 [0.0, 1.0] 之间，但得到的是: {ratio}")
                    self.logger.error(msg)
                    raise ValueError(msg)
                ratios_to_use[name] = ratio
                if ratio == 0.0:
                    has_zero_ratio = True

            if not ratios_to_use: # 可能所有表型都被忽略了
                 msg = "没有有效的表型和对应的非零筛选比例。"
                 self.logger.error(msg)
                 return np.array([], dtype=int) # 或者抛出错误

            if has_zero_ratio:
                 zero_ratio_phenotypes = [name for name, ratio in ratios_to_use.items() if ratio == 0.0]
                 self.logger.warning(f"以下表型的筛选比例为 0.0，将不会为它们选择 SNP: {zero_ratio_phenotypes}")

            self.logger.info(f"开始基于以下表型进行 MIC 筛选: {list(ratios_to_use.keys())}")
            self.logger.info(f"各表型筛选比例: { {name: f'{ratio:.2%}' for name, ratio in ratios_to_use.items()} }")

        else:
            msg = f"filter_ratios 参数类型无效: {type(filter_ratios)}。应为 float, List[float] 或 np.ndarray。"
            self.logger.error(msg)
            raise TypeError(msg)


        self.logger.info(f"总 SNP 数量: {self.n_total_snps}")

        final_selected_indices_set: Set[int] = set()

        # 使用 ratios_to_use 字典进行迭代
        for phenotype_name, current_ratio in ratios_to_use.items():
            if current_ratio == 0.0:
                continue # 跳过比例为 0 的表型

            mic_col = f"{phenotype_name}_mic"
            self.logger.debug(f"处理表型: {phenotype_name} (列: {mic_col}), 比例: {current_ratio:.2%}")

            # 计算当前表型需要筛选的 SNP 数量
            n_snps_per_phenotype = int(self.n_total_snps * current_ratio)
            if current_ratio > 0 and n_snps_per_phenotype == 0 and self.n_total_snps > 0:
                n_snps_per_phenotype = 1
            self.logger.debug(f"表型 {phenotype_name} 将选择 Top {n_snps_per_phenotype} 个 SNP。")

            # 获取 MIC 值
            mic_values = self.mic_dataframe[mic_col]

            # 获取按 MIC 值降序排列的索引
            sorted_indices = np.argsort(mic_values.to_numpy())[::-1]

            # 选取 Top N 的索引
            top_n_indices = sorted_indices[:n_snps_per_phenotype]

            # 添加到最终集合中
            final_selected_indices_set.update(top_n_indices)
            self.logger.debug(f"表型 {phenotype_name} 选出 {len(top_n_indices)} 个 SNP 索引。当前总选中 SNP 数（去重后）: {len(final_selected_indices_set)}")

        # 将集合转换为排序后的 NumPy 数组
        final_selected_indices = np.array(sorted(list(final_selected_indices_set)), dtype=int)

        self.logger.info(f"MIC 筛选完成。总共筛选出 {len(final_selected_indices)} 个唯一的 SNP。")
        if len(final_selected_indices) < self.n_total_snps:
             self.logger.info(f"SNP 数量从 {self.n_total_snps} 降维至 {len(final_selected_indices)}")
        else:
             self.logger.info("筛选后的 SNP 数量与原始数量相同。")


        return final_selected_indices

if __name__ == '__main__':
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("MICFilterTest")

    mic_file = Path("/media/marxin/softs/workspace/Whisper_of_DNA_pl/data/maize1404_genotype/A_cubic_1404_mic_qc_MIC.bimmic") # <--- 修改为你的 MIC 文件路径

    if not mic_file.exists():
         test_logger.error(f"测试所需的 MIC 文件不存在: {mic_file}")
    else:
        try:
            mic_filter = MICFilter(mic_file_path=mic_file, logger=test_logger)

            # --- 测试 1: 单一比例 ---
            test_logger.info("\n--- 测试单一比例 ---")
            phenotypes_to_filter = ['DTA', 'DTS', 'PH'] # <--- 修改为你的目标表型
            ratio_single = 0.1 # 筛选前 10%
            selected_indices_single = mic_filter.filter_snps(phenotypes_to_filter, ratio_single)
            test_logger.info(f"单一比例 ({ratio_single:.1%}) 最终筛选出的 SNP 索引 (前10个): {selected_indices_single[:10]}")
            test_logger.info(f"总共筛选出 {len(selected_indices_single)} 个 SNP。")

            # --- 测试 2: 列表比例 ---
            test_logger.info("\n--- 测试列表比例 ---")
            phenotypes_to_filter_list = ['DTA', 'DTS', 'PH', 'InvalidPheno'] # 包含一个无效表型
            ratios_list = [0.1, 0.05, 0.2, 0.1] # 长度必须与 phenotypes_to_filter_list 匹配
            test_logger.info(f"请求表型: {phenotypes_to_filter_list}")
            test_logger.info(f"对应比例: {ratios_list}")
            selected_indices_list = mic_filter.filter_snps(phenotypes_to_filter_list, ratios_list)
            test_logger.info(f"列表比例最终筛选出的 SNP 索引 (前10个): {selected_indices_list[:10]}")
            test_logger.info(f"总共筛选出 {len(selected_indices_list)} 个 SNP。")

            # --- 测试 3: 列表比例长度不匹配 ---
            test_logger.info("\n--- 测试列表比例长度不匹配 ---")
            try:
                mic_filter.filter_snps(['DTA', 'DTS'], [0.1, 0.2, 0.3])
            except ValueError as e:
                test_logger.info(f"捕获到预期错误: {e}")

            # --- 测试 4: 列表比例值无效 ---
            test_logger.info("\n--- 测试列表比例值无效 ---")
            try:
                mic_filter.filter_snps(['DTA', 'DTS'], [0.1, 1.5])
            except ValueError as e:
                test_logger.info(f"捕获到预期错误: {e}")

            # --- 测试 5: 列表比例全为 0 ---
            test_logger.info("\n--- 测试列表比例全为 0 ---")
            indices_all_zero = mic_filter.filter_snps(['DTA', 'DTS'], [0.0, 0.0])
            test_logger.info(f"列表比例全为 0 时返回索引: {indices_all_zero} (长度: {len(indices_all_zero)})")


        except (FileNotFoundError, MICFilterError, ValueError, TypeError) as e:
            test_logger.error(f"测试过程中发生错误: {e}")
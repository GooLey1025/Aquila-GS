"""
配置处理工具模块，用于加载和处理模型配置信息
"""
import json
import os
import yaml
import warnings
from typing import Dict, Any, Optional, Union, List
import torch
import numpy as np
import random

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，支持 .json 和 .yaml
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    # 根据文件扩展名决定加载方式
    ext = os.path.splitext(config_path)[-1].lower()
    
    if ext == ".json":
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif ext in [".yaml", ".yml"]:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}")
        
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        save_path: 保存路径，支持 .json 和 .yaml
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 根据文件扩展名决定保存方式
    ext = os.path.splitext(save_path)[-1].lower()
    
    if ext == ".json":
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    elif ext in [".yaml", ".yml"]:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}")

def get_device() -> torch.device:
    """
    获取可用的计算设备
    
    Returns:
        torch.device: 可用的计算设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')  # 对于 Apple 芯片
    else:
        return torch.device('cpu')

def get_precision_from_config(config: Dict[str, Any]) -> int:
    """
    从配置中获取精度设置
    
    Args:
        config: 配置字典
    
    Returns:
        precision: PyTorch Lightning 精度设置 (16, 32, 64, 'bf16')
    """
    precision = config.get('precision', 'float32')
    
    if precision == 'float16' or precision == 16:
        return 16
    elif precision == 'bfloat16' or precision == 'bf16':
        return 'bf16'
    elif precision == 'float64' or precision == 64:
        return 64
    else:
        return 32  # 默认使用 float32

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典，override_config 的设置会覆盖 base_config
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
    
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置是否有效，包含对损失函数和表型配置的检查
    
    Args:
        config: 配置字典
    
    Returns:
        bool: 配置是否有效
    
    Raises:
        ValueError: 如果配置无效
        Warning: 如果某些可选配置缺失或格式不符合预期
    """
    # 检查必须的配置项
    required_sections = ["embedding", "GFI_FormerBLOCKS", "output_layer", "loss_config", "phenotype"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置中缺少必要的节: {section}")
    
    # --- Add validation for embedding section ---
    embedding_config = config["embedding"]
    if not isinstance(embedding_config, dict):
        raise ValueError("embedding 必须是一个字典")

    num_chromosomes = embedding_config.get("num_chromosomes")
    if num_chromosomes is None:
        raise ValueError("embedding 配置中缺少必要的键: 'num_chromosomes'")
    if not isinstance(num_chromosomes, int) or num_chromosomes <= 0:
        raise ValueError("embedding.num_chromosomes 必须是大于 0 的整数")

    chromosome_embedding_dim = embedding_config.get("chromosome_embedding_dim")
    if chromosome_embedding_dim is None:
        raise ValueError("embedding 配置中缺少必要的键: 'chromosome_embedding_dim'")
    if not isinstance(chromosome_embedding_dim, int) or chromosome_embedding_dim <= 0:
        raise ValueError("embedding.chromosome_embedding_dim 必须是大于 0 的整数")

    embedding_dim = embedding_config.get("embedding_dim")
    if embedding_dim is None:
         raise ValueError("embedding 配置中缺少必要的键: 'embedding_dim' (FFN 输出维度)")
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
         raise ValueError("embedding.embedding_dim 必须是大于 0 的整数")
    # --- End embedding validation ---
    
    # 检查 GFI_FormerBLOCKS 配置
    blocks_config = config["GFI_FormerBLOCKS"]
    num_blocks = blocks_config.get("num_blocks")
    if num_blocks is None or not isinstance(num_blocks, int) or num_blocks <= 0:
        raise ValueError("GFI_FormerBLOCKS.num_blocks 必须是大于 0 的整数")
        
    blocks = blocks_config.get("blocks", [])
    if not isinstance(blocks, list) or len(blocks) < num_blocks:
        raise ValueError(f"GFI_FormerBLOCKS.blocks 列表长度({len(blocks)})小于 num_blocks({num_blocks})")
    
    # 检查各个块的配置
    for i, block in enumerate(blocks[:num_blocks]):
        if not isinstance(block, dict):
             raise ValueError(f"GFI_FormerBLOCKS.blocks[{i}] 必须是一个字典")
        if "context_length" not in block:
            raise ValueError(f"GFI_FormerBLOCKS.blocks[{i}] 缺少 context_length 配置")
        if not isinstance(block["context_length"], int) or block["context_length"] <= 0:
            raise ValueError(f"GFI_FormerBLOCKS.blocks[{i}].context_length 必须是大于 0 的整数")

    # 检查 output_layer 配置
    output_config = config["output_layer"]
    if not isinstance(output_config, dict):
        raise ValueError("output_layer 必须是一个字典")
    phenotype_dim = output_config.get("phenotype_dim")
    if phenotype_dim is None or not isinstance(phenotype_dim, int) or phenotype_dim <= 0:
        raise ValueError("output_layer.phenotype_dim 必须是大于 0 的整数")
        
    phenotype_name = output_config.get("phenotype_name")
    if phenotype_name is None or not isinstance(phenotype_name, list) or len(phenotype_name) != phenotype_dim:
        raise ValueError(f"output_layer.phenotype_name 必须是一个列表，且长度等于 phenotype_dim ({phenotype_dim})")

    # 检查 loss_config 配置
    loss_config = config["loss_config"]
    if not isinstance(loss_config, dict):
        raise ValueError("loss_config 必须是一个字典")
    if "primary_loss" not in loss_config or not isinstance(loss_config["primary_loss"], dict):
        raise ValueError("loss_config.primary_loss 必须存在且是一个字典")
    
    aux_losses = loss_config.get("auxiliary_losses")
    if aux_losses is not None and not isinstance(aux_losses, dict):
        raise ValueError("loss_config.auxiliary_losses 如果存在，必须是一个字典")
    
    if aux_losses:
        # 检查 distribution_JS 配置
        js_config = aux_losses.get("distribution_JS")
        if js_config is not None:
            if not isinstance(js_config, dict):
                raise ValueError("loss_config.auxiliary_losses.distribution_JS 如果存在，必须是一个字典")
            js_enabled = js_config.get("enabled", False)
            if not isinstance(js_enabled, bool):
                 raise ValueError("loss_config.auxiliary_losses.distribution_JS.enabled 必须是布尔值")
            if js_enabled:
                js_weight = js_config.get("weight")
                if js_weight is None:
                    raise ValueError("如果 distribution_JS enabled，则必须提供 weight")
                if not isinstance(js_weight, list):
                    raise ValueError("loss_config.auxiliary_losses.distribution_JS.weight 必须是一个列表")
                if len(js_weight) != num_blocks:
                    raise ValueError(f"loss_config.auxiliary_losses.distribution_JS.weight 列表长度 ({len(js_weight)}) 必须等于 GFI_FormerBLOCKS.num_blocks ({num_blocks})")
                if not all(isinstance(w, (int, float)) for w in js_weight):
                    raise ValueError("loss_config.auxiliary_losses.distribution_JS.weight 列表中的所有元素必须是数字")

        # 检查 correlation 配置
        corr_config = aux_losses.get("correlation")
        if corr_config is not None:
            if not isinstance(corr_config, dict):
                raise ValueError("loss_config.auxiliary_losses.correlation 如果存在，必须是一个字典")
            corr_enabled = corr_config.get("enabled", False)
            if not isinstance(corr_enabled, bool):
                 raise ValueError("loss_config.auxiliary_losses.correlation.enabled 必须是布尔值")
            if corr_enabled:
                corr_weight = corr_config.get("weight")
                if corr_weight is None:
                    raise ValueError("如果 correlation enabled，则必须提供 weight")
                if not isinstance(corr_weight, (int, float)):
                    raise ValueError("loss_config.auxiliary_losses.correlation.weight 必须是数字")

        # 可以添加对 l1_regularization 的类似检查 (如果需要)
        l1_config = aux_losses.get("l1_regularization")
        if l1_config is not None:
             if not isinstance(l1_config, dict):
                 raise ValueError("loss_config.auxiliary_losses.l1_regularization 如果存在，必须是一个字典")
             l1_enabled = l1_config.get("enabled", False)
             if not isinstance(l1_enabled, bool):
                 raise ValueError("loss_config.auxiliary_losses.l1_regularization.enabled 必须是布尔值")
             if l1_enabled:
                 l1_weight = l1_config.get("weight")
                 if l1_weight is None:
                     raise ValueError("如果 l1_regularization enabled，则必须提供 weight")
                 if not isinstance(l1_weight, (int, float)):
                     raise ValueError("loss_config.auxiliary_losses.l1_regularization.weight 必须是数字")


    # 检查 phenotype 配置
    phenotype_config = config["phenotype"]
    if not isinstance(phenotype_config, dict):
        raise ValueError("phenotype 必须是一个字典")
    
    target_log_var = phenotype_config.get("target_log_variance")
    if target_log_var is None:
        warnings.warn("phenotype.target_log_variance 未在配置中指定，将使用模型默认值")
    elif isinstance(target_log_var, list):
        if len(target_log_var) != phenotype_dim:
            raise ValueError(f"如果 phenotype.target_log_variance 是列表，其长度 ({len(target_log_var)}) 必须等于 output_layer.phenotype_dim ({phenotype_dim})")
        if not all(isinstance(v, (int, float)) for v in target_log_var):
            raise ValueError("phenotype.target_log_variance 列表中的所有元素必须是数字")
    elif not isinstance(target_log_var, (int, float)):
        raise ValueError("phenotype.target_log_variance 必须是数字或数字列表")

    return True

def set_random_seed(seed: int) -> None:
    """
    设置随机种子以确保结果可重现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 额外设置以确保可重现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_precision_dtype(precision: str) -> torch.dtype:
    """
    根据配置获取PyTorch数据类型
    
    Args:
        precision: 精度字符串，如'float32'、'float16'、'bfloat16'
        
    Returns:
        相应的PyTorch数据类型
    """
    precision_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float64': torch.float64
    }
    return precision_map.get(precision, torch.float32)

class ConfigHandler:
    """配置处理器类，用于处理和访问模型配置"""
    
    def __init__(self, config_path: str):
        """
        初始化配置处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        
        # 如果定义了随机种子，则设置它
        if 'random_seed' in self.config:
            set_random_seed(self.config['random_seed'])
            
        # 设置计算精度
        self.precision = get_precision_dtype(self.config.get('precision', 'float32'))
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入层配置"""
        return self.config.get('embedding', {})
    
    def get_gfi_blocks_config(self) -> Dict[str, Any]:
        """获取GFI-Former块配置"""
        return self.config.get('GFI_FormerBLOCKS', {})
    
    def get_output_layer_config(self) -> Dict[str, Any]:
        """获取输出层配置"""
        return self.config.get('output_layer', {})
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        return self.config.get('loss_config', {})
    
    def get_phenotype_config(self) -> Dict[str, Any]:
        """获取表型配置"""
        return self.config.get('phenotype', {})
    
    def get_attention_config(self, block_idx: int, is_encoder: bool = True) -> Dict[str, Any]:
        """
        获取特定块的注意力配置
        
        Args:
            block_idx: 块索引
            is_encoder: 是否为编码器注意力（否则为解码器交叉注意力）
            
        Returns:
            注意力配置字典
        """
        blocks = self.get_gfi_blocks_config().get('blocks', [])
        if block_idx >= len(blocks):
            raise IndexError(f"块索引超出范围: {block_idx}, 总块数: {len(blocks)}")
        
        block_config = blocks[block_idx]
        if is_encoder:
            return block_config.get('encoder', {}).get('attention', {})
        else:
            return block_config.get('decoder', {}).get('cross_attention', {})
    
    def get_pooling_config(self, block_idx: int) -> Dict[str, Any]:
        """
        获取特定块的池化配置
        
        Args:
            block_idx: 块索引
            
        Returns:
            池化配置字典
        """
        blocks = self.get_gfi_blocks_config().get('blocks', [])
        if block_idx >= len(blocks):
            raise IndexError(f"块索引超出范围: {block_idx}, 总块数: {len(blocks)}")
        
        block_config = blocks[block_idx]
        return block_config.get('decoder', {}).get('pooling', {})
    
    def get_context_length(self, block_idx: int) -> int:
        """
        获取特定块的上下文长度
        
        Args:
            block_idx: 块索引
            
        Returns:
            上下文长度
        """
        blocks = self.get_gfi_blocks_config().get('blocks', [])
        if block_idx >= len(blocks):
            raise IndexError(f"块索引超出范围: {block_idx}, 总块数: {len(blocks)}")
        
        block_config = blocks[block_idx]
        return block_config.get('context_length', 300)  
    
    def get_all(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config

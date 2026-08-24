import pytorch_lightning as pl
import time
import torch
from typing import Dict, Any

class DynamicTrainingCallback(pl.Callback):
    """自动调整训练参数以最大化吞吐量的回调"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动态训练回调
        
        Args:
            config: 配置字典，包含动态优化器的参数
        """
        super().__init__()
        self.last_batch_time = None
        self.last_throughput = 0
        self.min_ratio = config.get("min_throughput_ratio", 0.95)
        self.max_accumulate = config.get("max_accumulate_grad_batches", 16)
        self.enabled = config.get("enabled", True)
        self.batch_sizes = []
        self.throughputs = []
        
    def on_train_start(self, trainer, pl_module):
        """训练开始时初始化"""
        if not self.enabled:
            return
        self.last_batch_time = time.time()
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """批处理开始时记录时间"""
        if not self.enabled:
            return
        self.last_batch_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练批次结束后调用"""
        # 记录当前时间
        current_time = time.time()

        # 确保 batch 是字典类型
        if not isinstance(batch, dict):
            self.log.warning(f"Batch is not a dictionary, skipping dynamic adjustment. Batch type: {type(batch)}")
            return

        # 尝试从 batch 字典中获取批次大小
        if 'features' in batch:
            batch_size = batch['features'].size(0)
        elif 'phenotype' in batch:
            batch_size = batch['phenotype'].size(0)
        else:
            # 如果找不到 'features' 或 'phenotype'，尝试获取第一个张量的大小
            try:
                first_tensor = next(item for item in batch.values() if isinstance(item, torch.Tensor))
                batch_size = first_tensor.size(0)
            except (StopIteration, AttributeError):
                self.log.warning("Could not determine batch size from batch dictionary, skipping dynamic adjustment.")
                return

        # 计算批处理时间和吞吐量
        batch_time = current_time - self.last_batch_time
        current_throughput = batch_size / batch_time
        
        # 记录批大小和吞吐量
        self.batch_sizes.append(batch_size)
        self.throughputs.append(current_throughput)
        
        # 记录吞吐量指标
        trainer.logger.log_metrics({
            'throughput': current_throughput,
            'batch_time': batch_time,
            'batch_size': batch_size
        }, step=trainer.global_step)
        
        # 动态调整梯度累积步数
        if batch_idx > 0 and current_throughput < self.last_throughput * self.min_ratio:
            new_accumulate = min(self.max_accumulate, trainer.accumulate_grad_batches * 2)
            if new_accumulate != trainer.accumulate_grad_batches:
                trainer.accumulate_grad_batches = new_accumulate
                print(f"吞吐量下降，增加梯度累积步数到 {new_accumulate}")
                
        self.last_throughput = current_throughput
        self.last_batch_time = current_time  # 重置时间计数
        
    def on_train_epoch_end(self, trainer, pl_module):
        """训练轮次结束时记录平均吞吐量"""
        if not self.enabled or not self.throughputs:
            return
            
        avg_throughput = sum(self.throughputs) / len(self.throughputs)
        trainer.logger.log_metrics({
            'epoch_avg_throughput': avg_throughput,
        }, step=trainer.global_step)
        
        # 重置统计信息
        self.throughputs = []
        self.batch_sizes = []
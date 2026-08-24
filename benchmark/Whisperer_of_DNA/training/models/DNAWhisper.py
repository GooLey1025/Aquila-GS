import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
import warnings
import pandas as pd
import numpy as np
from torch import Tensor
from typing import Optional, Dict, List, Any, Tuple, Union, Literal
from einops import rearrange
from pathlib import Path
from functools import partial
from pytorch_lightning.utilities import rank_zero_only
import traceback
from torch.utils.data import Subset, DataLoader

try:
    import ot
except ImportError:
    ot = None
# --- 修改：导入 DNAWhisperModel 和 GFIFormerBlock ---
from .model import DNAWhisperModel, GFIFormerBlock # GFIFormerBlock 导入可能在 on_after_backward 中用到
# --- 结束修改 ---
from training.utils.config_utils import load_config # 假设 utils 在上一级目录


# +++ 新增 PairwiseCosineSimilarityDistributionLoss 类 +++
class PairwiseCosineSimilarityDistributionLoss(nn.Module):
    """
    Pairwise Cosine Similarity Distribution Loss.

    Encourages the distribution of pairwise cosine similarities among prediction
    vectors (after L2 normalization and feature concatenation) to be similar
    to that of target vectors.
    """
    def __init__(self, reduction='mean', eps=1e-8, gamma=1.0):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction
        self.eps = eps
        self.gamma = gamma # For single-target processing

    def _get_pairwise_cosine_similarity_for_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """ L2 normalize and compute pairwise cosine similarity for predictions. """
        # x is expected to be [B, Total_Feature_Dim]
        if x.ndim != 2 or x.shape[1] == 0:
            warnings.warn(f"PWCosSim Preds: Input not 2D or zero feature dim: {x.shape}. Returning identity if square, else error-like.")
            if x.shape[0] == x.shape[1]: # Batch size == feature dim (unlikely for features)
                 return torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
            # Fallback to avoid crash, but this indicates an issue upstream
            return torch.zeros((x.shape[0], x.shape[0]), device=x.device, dtype=x.dtype)


        x_norm = F.normalize(x, p=2, dim=-1, eps=self.eps)
        cos_sim_matrix = torch.matmul(x_norm, x_norm.transpose(-2, -1))
        return torch.clamp(cos_sim_matrix, -1.0 + self.eps, 1.0 - self.eps)

    def _get_pairwise_similarity_for_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """ Compute pairwise similarity for target vectors. """
        # targets shape [B, E]
        if targets.ndim != 2:
            raise ValueError(f"PWCosSim Targets: Targets must be 2D, got {targets.shape}")

        batch_size, num_phenotypes = targets.shape

        if num_phenotypes > 1: # Multi-target case
            targets_norm = F.normalize(targets, p=2, dim=-1, eps=self.eps)
            cos_sim_matrix = torch.matmul(targets_norm, targets_norm.transpose(-2, -1))
            return torch.clamp(cos_sim_matrix, -1.0 + self.eps, 1.0 - self.eps)
        
        elif num_phenotypes == 1: # Single-phenotype case
            # Calculate pairwise squared differences: [B, 1]
            # (targets.unsqueeze(1) - targets.unsqueeze(0)) gives [B, B, 1]
            diff_sq = (targets.unsqueeze(1) - targets.unsqueeze(0)).squeeze(-1)**2 # -> [B, B]
            
            # Apply Gaussian kernel
            s_base_target = torch.exp(-self.gamma * diff_sq) # Range (0, 1]
            
            # Linearly transform to (-1, 1]
            s_target_proxy = 2 * s_base_target - 1
            
            # Ensure diagonal is 1.0 (self-similarity)
            if s_target_proxy.shape[0] == s_target_proxy.shape[1]:
                 s_target_proxy.fill_diagonal_(1.0)
            return s_target_proxy
        else: # num_phenotypes == 0
            warnings.warn("PWCosSim Targets: Zero target dimension. Returning identity matrix.")
            return torch.eye(batch_size, device=targets.device, dtype=targets.dtype)


    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: [B, Total_Feature_Dim_Concatenated]
        # targets: [B, Phenotype_Dim]
        
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(f"Batch size mismatch: predictions ({predictions.shape[0]}), targets ({targets.shape[0]}).")

        batch_size = predictions.size(0)
        if batch_size <= 1: # Pairwise comparison not meaningful
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype, requires_grad=predictions.requires_grad or targets.requires_grad)

        if predictions.shape[1] == 0: # No features in predictions
            warnings.warn(f"PWCosSimLoss: Predictions have zero feature dimension ({predictions.shape}). Returning 0 loss.")
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype, requires_grad=predictions.requires_grad or targets.requires_grad)
        if targets.shape[1] == 0: # No features in targets
            warnings.warn(f"PWCosSimLoss: Targets have zero phenotype dimension ({targets.shape}). Returning 0 loss.")
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype, requires_grad=predictions.requires_grad or targets.requires_grad)


        cos_sim_preds = self._get_pairwise_cosine_similarity_for_predictions(predictions)
        cos_sim_targets = self._get_pairwise_similarity_for_targets(targets)

        # Use only upper triangular part (excluding diagonal)
        mask = torch.ones_like(cos_sim_preds, dtype=torch.bool, device=cos_sim_preds.device).triu(diagonal=1)
        
        sim_preds_flat = cos_sim_preds[mask]
        sim_targets_flat = cos_sim_targets[mask]

        if sim_preds_flat.numel() == 0: # Should not happen if batch_size > 1
             return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype, requires_grad=predictions.requires_grad or targets.requires_grad)

        loss = F.mse_loss(sim_preds_flat, sim_targets_flat, reduction=self.reduction)
        return loss
# +++ 结束新增 +++


class DNAWhisper(pl.LightningModule):
    """
    DNAWhisper 模型 - 基于 PyTorch Lightning 实现 (使用 DNAWhisperModel)
    """
    def __init__(self,
                 config_path: str = None,
                 config: Dict[str, Any] = None,
                 optimizer_config: Dict[str, Any] = None,
                 scheduler_config: Dict[str, Any] = None):
        super().__init__()

        # --- Config Loading (保持不变) ---
        loaded_config = None
        if config is not None:
            loaded_config = config
        elif config_path is not None:
            self.config_path_arg = config_path # Store for saving hparams
            loaded_config = load_config(config_path)
        else:
            raise ValueError("Either 'config' dictionary or 'config_path' must be provided.")
        # --- End Config Loading ---

        # --- 修改：保存配置，实例化 DNAWhisperModel ---
        hparams_to_save = {
            'config': loaded_config,
            'optimizer_config': optimizer_config if optimizer_config else {},
            'scheduler_config': scheduler_config if scheduler_config else {}
        }
        if hasattr(self, 'config_path_arg'):
             hparams_to_save['config_path'] = self.config_path_arg
        self.save_hyperparameters(hparams_to_save)

        self.random_seed = self.hparams.config.get("random_seed", 42)
        self.model = DNAWhisperModel(config=self.hparams.config, random_seed=self.random_seed)

        self.phenotype_names = self.hparams.config.get("output_layer", {}).get("phenotype_name", None)
        if self.phenotype_names is None:
             raise ValueError("Model config requires 'output_layer.phenotype_name'.")
        # --- 结束修改 ---

        log_config = self.hparams.config.get("logging", {})
        self.log_gradient_norms = log_config.get("log_gradient_norms", False)
        self.gradient_norm_log_interval = log_config.get("gradient_norm_log_interval", 50)

        phenotype_config = self.hparams.config.get("phenotype", {})
        self.phenotype_distribution = phenotype_config.get("distribution", "gaussian")
        self._target_log_variance_config = phenotype_config.get("target_log_variance", -4.6)
        # Initialize target_log_variance, will be moved to device in on_train_start
        self.register_buffer("target_log_variance", torch.tensor(0.0), persistent=False)


        # --- 损失函数配置 (修改) ---
        loss_config = self.hparams.config.get("loss_config", {})

        # Primary Loss (Output Layer Loss)
        primary_loss_config = loss_config.get("primary_loss", {})
        self.primary_loss_type = primary_loss_config.get("type", "mse").lower()
        self.primary_pearson_factor = primary_loss_config.get("pearson_factor", 1.0)
        self.primary_reduction = primary_loss_config.get("reduction", "mean")

        # Auxiliary Losses
        auxiliary_losses_config = loss_config.get("auxiliary_losses", {})

        # Deep Supervision Loss (Previous Primary Loss Logic)
        ds_config = auxiliary_losses_config.get("Deep_Supervision", {})
        self.ds_enabled = ds_config.get("enabled", False)
        self.ds_type = ds_config.get("type", "mse").lower()
        self.ds_weights = ds_config.get("weight", [])
        self.ds_pearson_factor = ds_config.get("pearson_factor", 1.0)
        self.ds_reduction = ds_config.get("reduction", "mean")

        # --- 修改：检查 DS 权重数量时考虑 Embedding 层 ---
        # 期望权重数量 = GFI Block 数量 + 1 (Embedding 层)
        num_expected_ds_weights = self.model.gfi_former.num_blocks + 1
        if self.ds_enabled and len(self.ds_weights) != num_expected_ds_weights:
            raise ValueError(f"Deep Supervision enabled, but number of weights ({len(self.ds_weights)}) "
                             f"does not match number of GFI blocks + 1 ({num_expected_ds_weights}).")
        # --- 结束修改 ---

        # Pairwise Cosine Similarity Loss
        pw_config = auxiliary_losses_config.get("PWCosSim", {}) # 确保键名与 config 文件一致
        self.pw_enabled = pw_config.get("enabled", False)
        self.pw_weights = pw_config.get("weight", []) # Weights for [Embedding, Block1, ..., BlockN]
        self.pw_factor = pw_config.get("PWCosSim_factor", 1.0) # 确保键名与 config 文件一致
        self.pw_reduction_config = pw_config.get("reduction", "mean") # Storing for clarity
        # --- 修改：实例化新的 PWCosSimLoss ---
        self.pw_gamma = pw_config.get("gamma", 1.0) # Get gamma for single-phenotype case
        if self.pw_enabled:
            self.pw_criterion = PairwiseCosineSimilarityDistributionLoss(
                reduction=self.pw_reduction_config,
                gamma=self.pw_gamma
            )
        # --- 结束修改 ---

        # --- 修改：检查 PWCosSim 权重数量 ---
        num_expected_pw_weights = self.model.gfi_former.num_blocks + 1
        if self.pw_enabled and len(self.pw_weights) != num_expected_pw_weights:
             raise ValueError(f"Pairwise CosSim enabled, but number of weights ({len(self.pw_weights)}) "
                              f"does not match number of GFI blocks + 1 ({num_expected_pw_weights}).")
        # --- 结束修改 ---

        l1_reg_config = auxiliary_losses_config.get("l1_regularization", {})
        self.l1_enabled = l1_reg_config.get("enabled", False)
        self.l1_weight = l1_reg_config.get("weight", 0.0)

        correlation_config = auxiliary_losses_config.get("correlation", {})
        self.correlation_enabled = correlation_config.get("enabled", False)
        self.correlation_weight = correlation_config.get("weight", 0.0)
        output_phenotype_dim = self.hparams.config.get("output_layer", {}).get("phenotype_dim", 3)
        # Register as buffer, will be calculated and synced by rank 0
        self.register_buffer('corr_true_target', torch.eye(output_phenotype_dim), persistent=False)


        self.best_val_loss = float('inf')
        self.did_calculate_corr_true = False # Flag to ensure calculation happens only once
        # --- 结束损失函数配置修改 ---

        # --- 初始化用于存储验证步骤输出的列表 ---
        self.validation_step_outputs = []

    # --- 修改：添加 rank_zero_only 装饰器 ---
    @rank_zero_only
    def calculate_target_correlation(self):
        if self.did_calculate_corr_true or not self.correlation_enabled:
            return # Already calculated or not enabled
        logging.info("Calculating target correlation matrix from training data (rank 0)...")

        # --- MODIFIED CHECKS ---
        if not hasattr(self, 'trainer') or self.trainer is None:
            warnings.warn("Cannot calculate target correlation: Trainer not available.")
            return
        if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            warnings.warn("Cannot calculate target correlation: Datamodule not available.")
            return
        # Check for the full dataset and train indices instead of 'train_dataset'
        if not hasattr(self.trainer.datamodule, 'dataset') or self.trainer.datamodule.dataset is None:
            warnings.warn("Cannot calculate target correlation: Datamodule.dataset not available (check datamodule setup).")
            return
        if not hasattr(self.trainer.datamodule, 'train_indices') or self.trainer.datamodule.train_indices is None:
            warnings.warn("Cannot calculate target correlation: Datamodule.train_indices not available (check datamodule setup).")
            return
        # --- END MODIFIED CHECKS ---

        try:
            full_dataset = self.trainer.datamodule.dataset
            train_indices = self.trainer.datamodule.train_indices

            if len(train_indices) == 0:
                 warnings.warn("Cannot calculate target correlation: Training dataset (indices) is empty.")
                 return

            # Create a subset representing the training data
            train_subset = Subset(full_dataset, train_indices)
            logging.info(f"Created training subset for correlation calculation with {len(train_subset)} samples.")

            # Access labels by iterating through the subset using a temporary DataLoader
            all_labels = []
            # Use a reasonable batch size and minimal workers for this internal task
            # Ensure pin_memory is False if running on CPU or if issues arise
            temp_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
            logging.info("Extracting labels using temporary DataLoader...")
            processed_batches = 0
            for batch in temp_loader:
                # Adjust access based on your dataset's __getitem__ structure
                labels = batch.get('phenotype') # Assumes batch is a dict
                if labels is None and isinstance(batch, (list, tuple)) and len(batch) > 1:
                    labels = batch[1] # Fallback if batch is tuple/list

                if labels is not None:
                    all_labels.append(labels.cpu())
                    processed_batches += 1
                else:
                    # Log only once if labels cannot be extracted
                    if not hasattr(self, '_warned_label_extraction'):
                        warnings.warn("Could not extract 'phenotype' or batch[1] from temp_loader batch. Check dataset __getitem__.")
                        self._warned_label_extraction = True
                    # Decide whether to break or continue if one batch fails
                    # break # Option: Stop if label structure is wrong

            logging.info(f"Extracted labels from {processed_batches} batches.")

            if not all_labels:
                  warnings.warn("Could not extract any labels from training subset via DataLoader.")
                  return
            all_labels = torch.cat(all_labels, dim=0)
            logging.info(f"Concatenated labels shape: {all_labels.shape}")


            if all_labels.ndim != 2 or all_labels.shape[0] <= 1 or all_labels.shape[1] == 0:
                 warnings.warn(f"Insufficient data or incorrect label shape ({all_labels.shape}) for correlation calculation.")
                 return

            # Calculate correlation matrix (existing logic)
            labels_float = all_labels.float()
            mean_labels = torch.mean(labels_float, dim=0, keepdim=True)
            centered_labels = labels_float - mean_labels
            # Ensure denominator is not zero for covariance calculation
            n_samples = centered_labels.shape[0]
            if n_samples <= 1:
                 warnings.warn(f"Cannot calculate covariance with {n_samples} samples.")
                 return
            cov_matrix = torch.matmul(centered_labels.T, centered_labels) / (n_samples - 1)

            std_dev = torch.sqrt(torch.diag(cov_matrix))
            # Avoid division by zero for constant features
            std_dev = torch.where(std_dev < 1e-6, torch.tensor(1.0, device=std_dev.device), std_dev) # Use a small epsilon
            corr_matrix = cov_matrix / torch.outer(std_dev, std_dev)
            # Clamp values to handle potential numerical instability
            corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
            # Fill NaNs resulting from 0/0 with 0 (no correlation for constant features)
            corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)


            # Ensure the buffer is on the correct device before assigning
            self.corr_true_target = corr_matrix.to(self.device)
            self.did_calculate_corr_true = True # Set flag after successful calculation
            logging.info(f"Calculated and registered target correlation matrix (shape: {self.corr_true_target.shape}) on device {self.device}.")

        except Exception as e:
            warnings.warn(f"Error calculating target correlation: {e}")
            traceback.print_exc() # Print detailed traceback for debugging

    def setup(self, stage: Optional[str] = None):
        # setup is called per-process, avoid heavy computation here
        pass

    def on_train_start(self):
        # Calculate correlation only on rank 0, PL handles syncing the buffer
        self.calculate_target_correlation()
        # Ensure target_log_variance is on the correct device
        self.target_log_variance = torch.tensor(self._target_log_variance_config).to(self.device)


    # --- on_after_backward: 检查梯度记录的模块是否仍然有效 ---
    # 当前实现记录 embedding, gfi_block_i, encoder, decoder, pooling, output_layer
    # 这些主要模块仍然存在，内部细节变化不影响此处的记录逻辑。
    # 如果需要更细粒度的记录（例如记录新的 projection_for_next_block），则需要修改。
    # 暂时保持不变。
    def on_after_backward(self):
        """计算并记录指定模块的梯度范数。"""
        if not self.log_gradient_norms:
            return

        if self.trainer.global_step % self.gradient_norm_log_interval == 0:
            norms = {}

            def calculate_module_grad_norm(module: nn.Module) -> float:
                """计算单个模块的梯度范数。"""
                total_norm_sq = 0.0
                for p in module.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq += param_norm.item() ** 2
                return total_norm_sq ** 0.5

            # 1. Embedding Layer
            norms['grad_norm/embedding'] = calculate_module_grad_norm(self.model.embedding)
            # --- 添加：记录 Embedding 内部的 Pooling 层 (如果存在) ---
            if hasattr(self.model.embedding, 'block_pooling') and self.model.embedding.block_pooling is not None:
                norms['grad_norm/embedding/pooling'] = calculate_module_grad_norm(self.model.embedding.block_pooling)
            # --- 结束添加 ---

            # 2. GFIFormer Blocks
            for i, block in enumerate(self.model.gfi_former.gfi_blocks):
                if isinstance(block, GFIFormerBlock): # 确保是 GFIFormerBlock 类型
                    block_name = f'grad_norm/gfi_block_{i}'
                    norms[block_name] = calculate_module_grad_norm(block)

                    # 2.1 Encoder part (Input Norm+Projection + Encoder Layers)
                    # --- 修改：包含 norm_input ---
                    encoder_module = nn.ModuleList([block.norm_input, block.projection_from_input, block.encoder_layers])
                    # --- 结束修改 ---
                    norms[f'{block_name}/encoder'] = calculate_module_grad_norm(encoder_module)

                    # 2.2 Decoder part (CrossAttn, Proj, MoE, Pooling, BN, FeatureProcessor, AuxOutput, ProjNext)
                    # --- 修改：包含新的和修改后的层 ---
                    decoder_modules = nn.ModuleList([
                        block.norm_cross_attn,
                        block.cross_attention,
                        block.projection_to_moe_input,
                        block.norm_moe_input,
                        block.moe,
                        block.pooling,
                    ])
                    # --- 结束修改 ---
                    norms[f'{block_name}/decoder'] = calculate_module_grad_norm(decoder_modules)

                    # 2.3 Pooling Layer (Explicitly requested)
                    norms[f'{block_name}/pooling'] = calculate_module_grad_norm(block.pooling)
                else:
                    warnings.warn(f"Skipping gradient norm calculation for module at index {i} in gfi_blocks as it is not a GFIFormerBlock.")

            # 3. Output Layer
            norms['grad_norm/output_layer'] = calculate_module_grad_norm(self.model.output_layer)

            # 4. Total Model Norm (Optional but good for comparison)
            norms['grad_norm/total_model'] = calculate_module_grad_norm(self.model)

            # Log the calculated norms
            self.logger.log_metrics(norms, step=self.trainer.global_step)
    # --- 结束 on_after_backward ---

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool() # Ensure mask is boolean
        return self.model(x, mask)

    # --- 损失函数辅助方法 (_get_loss_function, _log_cosh_loss, _mse_loss, _pearson_loss, l1_regularization_loss) 保持不变 ---
    def _get_loss_function(self, loss_type: str, pearson_factor: float = 1.0, reduction: str = 'mean'):
        """
        Returns the appropriate loss function based on the type, partially filled
        with pearson_factor and reduction strategy.
        """
        if loss_type == "mse":
            # Return a lambda that captures the reduction strategy
            return partial(self._mse_loss, reduction=reduction)
        elif loss_type == "logcosh":
            # LogCosh typically uses mean reduction, ignore reduction parameter for now
            return self._log_cosh_loss
        elif loss_type == "pearson":
             # Return a lambda that captures pearson_factor and reduction
             return partial(self._pearson_loss, pearson_factor=pearson_factor, reduction=reduction)
        else:
             warnings.warn(f"Unsupported loss type: {loss_type}. Using MSE with reduction={reduction}.")
             return partial(self._mse_loss, reduction=reduction)

    def _log_cosh_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        loss = torch.log(torch.cosh(diff) + 1e-12)
        return torch.mean(loss) # LogCosh typically uses mean reduction

    def _mse_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str) -> torch.Tensor:
        """Calculates the Mean Squared Error loss with configurable reduction."""
        if y_pred.shape != y_true.shape:
             raise ValueError(f"MSE Loss: Shape mismatch between prediction {y_pred.shape} and target {y_true.shape}")

        loss = F.mse_loss(y_pred, y_true, reduction='none') # Calculate element-wise loss first

        if reduction == "minimax":
            # Maximize the minimum loss across phenotypes (per sample), then average over batch
            loss_per_phenotype = loss.mean(dim=0) # Average over batch first [E]
            return loss_per_phenotype.max() # Maximize the average loss across phenotypes
        elif reduction == "mean":
            return loss.mean() # Average over all elements
        else: # Default to mean if reduction is unknown
             warnings.warn(f"Unsupported MSE reduction: {reduction}. Using 'mean'.")
             return loss.mean()


    def _pearson_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, pearson_factor: float, reduction: str, eps: float = 1e-9) -> torch.Tensor:
        """
        Calculates a loss based on the Pearson correlation coefficient with configurable reduction.
        Aims to maximize correlation (minimize 1-correlation). Operates per phenotype.
        """
        if y_pred.shape[0] <= 1:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=y_pred.requires_grad) # Cannot compute correlation for batch size <= 1
        if pearson_factor == 0:
             return torch.tensor(0.0, device=y_pred.device, requires_grad=y_pred.requires_grad) # No Pearson loss if factor is 0
        if y_pred.shape != y_true.shape:
             raise ValueError(f"Pearson Loss: Shape mismatch between prediction {y_pred.shape} and target {y_true.shape}")
        if y_pred.ndim != 2 or y_pred.shape[1] == 0:
             warnings.warn(f"Pearson Loss expects 2D input [B, E], got {y_pred.shape}. Returning 0 loss.")
             return torch.tensor(0.0, device=y_pred.device, requires_grad=y_pred.requires_grad)


        # Ensure float32 for stable correlation calculation
        y_pred_f32 = y_pred.float()
        y_true_f32 = y_true.float()

        # Calculate correlation per phenotype (column-wise)
        losses_per_phenotype = []
        for i in range(y_pred_f32.shape[1]):
            pred_col = y_pred_f32[:, i]
            true_col = y_true_f32[:, i]

            vx = pred_col - torch.mean(pred_col)
            vy = true_col - torch.mean(true_col)

            std_vx = torch.sqrt(torch.sum(vx ** 2))
            std_vy = torch.sqrt(torch.sum(vy ** 2))

            if std_vx < eps or std_vy < eps:
                # Handle constant columns: correlation is undefined, assign 0 correlation -> max loss (1.0)
                p = torch.tensor(0.0, device=y_pred.device)
            else:
                p = torch.sum(vx * vy) / (std_vx * std_vy)
                p = torch.clamp(p, -1.0, 1.0) # Clamp for numerical stability

            # Loss: (1 - correlation) / factor, squared to penalize negative correlation more
            # Ensure loss is non-negative
            loss_i = ((1.0 - p) / pearson_factor) ** 2
            losses_per_phenotype.append(loss_i)

        losses_tensor = torch.stack(losses_per_phenotype) # Shape [E]

        if reduction == "minimax":
            # Maximize the minimum loss across phenotypes
            return losses_tensor.max() # Maximize the loss (minimize the worst correlation)
        elif reduction == "mean":
            return losses_tensor.mean() # Average loss across phenotypes
        else:
            warnings.warn(f"Unsupported Pearson reduction: {reduction}. Using 'mean'.")
            return losses_tensor.mean()

    def l1_regularization_loss(self) -> torch.Tensor:
        l1_loss = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            if param.requires_grad:
                l1_loss += torch.norm(param, p=1)
        return l1_loss

    # --- 修改：重写 compute_loss ---
    def compute_loss(self, model_outputs: Dict[str, Any], y_true: torch.Tensor, raw_features: Optional[torch.Tensor] = None) -> Dict[str, Tensor]:
        final_pred = model_outputs['final_pred']
        embed_features = model_outputs['embed_features'] # Shape [B, N_emb, D_emb]
        gfi_block_features = model_outputs['gfi_block_features'] # List of [B, N_i, D_i]
        embed_aux_proj = model_outputs['embed_aux_proj'] # Shape [B, E]
        gfi_aux_projections = model_outputs['gfi_aux_projections'] # List of [B*N_i, E, 1]

        true_batch_size = y_true.shape[0]
        phenotype_dim = y_true.shape[1] 

        primary_loss_fn = self._get_loss_function(
            self.primary_loss_type,
            pearson_factor=self.primary_pearson_factor,
            reduction=self.primary_reduction
        )
        primary_loss = primary_loss_fn(final_pred, y_true)
        loss_requires_grad = primary_loss.requires_grad

        total_deep_supervision_loss = torch.tensor(0.0, device=final_pred.device, requires_grad=loss_requires_grad)
        if self.ds_enabled and self.training:
            ds_loss_fn = self._get_loss_function(
                self.ds_type,
                pearson_factor=self.ds_pearson_factor,
                reduction=self.ds_reduction
            )
            ds_losses = []
            if len(self.ds_weights) > 0 and self.ds_weights[0] > 0:
                if embed_aux_proj is not None and embed_aux_proj.shape == y_true.shape:
                    loss_emb = ds_loss_fn(embed_aux_proj, y_true)
                    ds_losses.append(self.ds_weights[0] * loss_emb)
                elif embed_aux_proj is not None:
                    warnings.warn(f"Deep Supervision: Embedding aux projection shape {embed_aux_proj.shape} "
                                  f"mismatched with target shape {y_true.shape}. Skipping DS loss for embedding.")
            for i, aux_proj_flat in enumerate(gfi_aux_projections):
                weight_idx = i + 1
                if weight_idx < len(self.ds_weights) and self.ds_weights[weight_idx] > 0:
                    if aux_proj_flat is not None:
                        try:
                            num_context_blocks = gfi_block_features[i].shape[1] # N_i
                            expected_shape = (true_batch_size * num_context_blocks, phenotype_dim, 1)
                            if aux_proj_flat.shape == expected_shape:
                                aux_proj_reshaped = aux_proj_flat.view(true_batch_size, num_context_blocks, phenotype_dim, 1)
                                aux_proj_processed = aux_proj_reshaped.mean(dim=1).squeeze(-1)
                                if aux_proj_processed.shape == y_true.shape:
                                    loss_block = ds_loss_fn(aux_proj_processed, y_true)
                                    ds_losses.append(self.ds_weights[weight_idx] * loss_block)
                                else:
                                    warnings.warn(f"Deep Supervision: GFI Block {i} processed aux projection shape {aux_proj_processed.shape} mismatched with target shape {y_true.shape} after processing. Skipping DS loss for block {i}.")
                            else:
                                warnings.warn(f"Deep Supervision: GFI Block {i} raw aux projection shape {aux_proj_flat.shape} "
                                              f"did not match expected shape {expected_shape} (B={true_batch_size}, N_i={num_context_blocks}, E={phenotype_dim}). Skipping DS loss for block {i}.")
                        except IndexError:
                             warnings.warn(f"Deep Supervision: Could not access gfi_block_features for block {i} to determine context blocks. Skipping DS loss.")
                        except Exception as e:
                             warnings.warn(f"Deep Supervision: Error processing aux projection for GFI Block {i}: {e}. Skipping DS loss.")
            if ds_losses:
                total_deep_supervision_loss = torch.stack(ds_losses).sum()


        # --- PWCosSim Loss Calculation (New Implementation) ---
        total_pw_cosim_loss = torch.tensor(0.0, device=final_pred.device, requires_grad=loss_requires_grad)
        if self.pw_enabled and self.training:
            # EMD calculation requires raw integer/boolean marker data. If the input format
            # is incorrect (e.g., float), this auxiliary feature is bypassed.
            if not isinstance(raw_features, torch.Tensor) or raw_features.dtype not in [torch.int8, torch.bool]:
                raw_features = None

            emd_dissimilarity_target = None
            if raw_features is not None:
                emd_dissimilarity_target = self._calculate_emd_dissimilarity_matrix(raw_features)

            pw_losses_terms = []
            
            # 1. Embedding features
            # pw_weights[0] is for embedding features
            if len(self.pw_weights) > 0 and self.pw_weights[0] > 0:
                if embed_features is not None and embed_features.ndim == 3 and embed_features.shape[0] == true_batch_size:
                    # Reshape [B, N_emb, D_emb] to [B, N_emb * D_emb]
                    preds_emb = embed_features.reshape(true_batch_size, -1)
                    if preds_emb.shape[1] > 0: # Check if feature dimension is non-zero
                        loss_emb_pw = self.pw_criterion(preds_emb, y_true)
                        pw_losses_terms.append(self.pw_weights[0] * loss_emb_pw)

                        if emd_dissimilarity_target is not None:
                            sim_preds_emb = self.pw_criterion._get_pairwise_cosine_similarity_for_predictions(preds_emb)
                            dissim_preds_emb = 1.0 - sim_preds_emb
                            emd_target_device = emd_dissimilarity_target.to(dissim_preds_emb.device)
                            emd_reg_loss = F.mse_loss(dissim_preds_emb, emd_target_device)
                            pw_losses_terms.append(self.pw_weights[0] * emd_reg_loss)

                    else:
                        warnings.warn("PWCosSimLoss: Embedding features reshaped to zero feature dimension. Skipping.")
                elif embed_features is not None:
                     warnings.warn(f"PWCosSimLoss: Embedding features have unexpected shape {embed_features.shape} for PWCosSim. Skipping.")


            # 2. GFI Block features
            for i, block_feat in enumerate(gfi_block_features):
                weight_idx = i + 1 # pw_weights[0] is for embedding
                if weight_idx < len(self.pw_weights) and self.pw_weights[weight_idx] > 0:
                    if block_feat is not None and block_feat.ndim == 3 and block_feat.shape[0] == true_batch_size:
                        # Reshape [B, N_i, D_block_i] to [B, N_i * D_block_i]
                        preds_gfi_i = block_feat.reshape(true_batch_size, -1)
                        if preds_gfi_i.shape[1] > 0: # Check if feature dimension is non-zero
                            loss_gfi_pw_i = self.pw_criterion(preds_gfi_i, y_true)
                            pw_losses_terms.append(self.pw_weights[weight_idx] * loss_gfi_pw_i)
                        else:
                            warnings.warn(f"PWCosSimLoss: GFI Block {i} features reshaped to zero feature dimension. Skipping.")
                    elif block_feat is not None:
                        warnings.warn(f"PWCosSimLoss: GFI Block {i} features have unexpected shape {block_feat.shape} for PWCosSim. Skipping.")
            
            if pw_losses_terms:
                total_pw_cosim_loss = torch.stack(pw_losses_terms).sum() * self.pw_factor # Apply overall factor
        # --- End PWCosSim Loss Calculation ---

        l1_loss = torch.tensor(0.0, device=final_pred.device, requires_grad=loss_requires_grad if self.l1_enabled else False)
        if self.l1_enabled:
             l1_loss = self.l1_regularization_loss()

        correlation_loss = torch.tensor(0.0, device=final_pred.device, requires_grad=loss_requires_grad if self.correlation_enabled else False)
        if self.correlation_enabled and true_batch_size > 1 and self.corr_true_target is not None and self.did_calculate_corr_true:
            # ... (Correlation calculation logic remains the same) ...
            pred_f32 = final_pred.float()
            mean_pred = torch.mean(pred_f32, dim=0, keepdim=True)
            centered_pred = pred_f32 - mean_pred
            n_samples_pred = centered_pred.shape[0]
            if n_samples_pred <= 1:
                 warnings.warn("Correlation Loss: Cannot compute covariance with <= 1 sample in batch.")
                 correlation_loss = torch.tensor(0.0, device=final_pred.device, requires_grad=loss_requires_grad)
            else:
                cov_matrix_pred = torch.matmul(centered_pred.T, centered_pred) / (n_samples_pred - 1)
                std_dev_pred = torch.sqrt(torch.diag(cov_matrix_pred))
                std_dev_pred = torch.where(std_dev_pred < 1e-6, torch.tensor(1.0, device=std_dev_pred.device), std_dev_pred)
                corr_matrix_pred = cov_matrix_pred / torch.outer(std_dev_pred, std_dev_pred)
                corr_matrix_pred = torch.clamp(corr_matrix_pred, -1.0, 1.0)
                corr_matrix_pred = torch.nan_to_num(corr_matrix_pred, nan=0.0)
                correlation_loss = F.mse_loss(corr_matrix_pred, self.corr_true_target)
        # --- 结束 PWCosSim, L1, Correlation 计算 ---

        # --- 总损失计算 (不变) ---
        total_loss = primary_loss
        if self.ds_enabled and self.training:
            total_loss = total_loss + total_deep_supervision_loss
        if self.pw_enabled and self.training:
            total_loss = total_loss + total_pw_cosim_loss
        if self.l1_enabled:
             total_loss = total_loss + self.l1_weight * l1_loss
        if self.correlation_enabled:
             total_loss = total_loss + self.correlation_weight * correlation_loss
        # --- 结束总损失计算 ---

        return {
            "total_loss": total_loss,
            "primary_loss": primary_loss,
            "deep_supervision_loss": total_deep_supervision_loss,
            "pw_cosim_loss": total_pw_cosim_loss,
            "l1_loss": l1_loss if self.l1_enabled else torch.tensor(0.0, device=final_pred.device),
            "correlation_loss": correlation_loss if self.correlation_enabled else torch.tensor(0.0, device=final_pred.device)
        }
    # --- End of compute_loss modifications ---

    def _calculate_emd_dissimilarity_matrix(self, marker_data: torch.Tensor, reg: float = 0.1, max_samples: int = 512) -> Optional[torch.Tensor]:
        """
        Calculates a pairwise sample dissimilarity matrix using an
        approximated Earth Mover's Distance (EMD) for binary marker data.

        The dissimilarity D(Si, Sj) between two samples is defined as the EMD
        between the sets of coordinates corresponding to markers unique to each sample.
        This is efficiently approximated using the Sinkhorn algorithm.

        Args:
            marker_data (torch.Tensor): A binary tensor of shape [N, M].
            reg (float): The regularization parameter for the Sinkhorn algorithm.
            max_samples (int): Limit the number of samples to avoid excessive computation time.

        Returns:
            Optional[torch.Tensor]: A dissimilarity matrix of shape [N, N], or None if calculation fails.
        """
        if ot is None:
            warnings.warn("Python Optimal Transport library (`ot`) not found, skipping EMD calculation. Please `pip install pot`.")
            return None

        if marker_data.shape[0] > max_samples:
            warnings.warn(f"Number of samples ({marker_data.shape[0]}) exceeds max_samples ({max_samples}) for EMD calculation. Subsetting to {max_samples} samples.")
            marker_data = marker_data[:max_samples]

        device = marker_data.device
        N, M = marker_data.shape
        dissimilarity_matrix = torch.zeros(N, N, device=device, dtype=torch.float32)

        # Pre-calculate marker coordinates once
        marker_coords = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(1)

        for i in range(N):
            for j in range(i + 1, N):
                sample_i = marker_data[i]
                sample_j = marker_data[j]

                non_identical = (sample_i != sample_j)
                coords_i = marker_coords[non_identical & (sample_i == 1)]
                coords_j = marker_coords[non_identical & (sample_j == 1)]

                num_coords_i = coords_i.numel()
                num_coords_j = coords_j.numel()

                if num_coords_i == 0 or num_coords_j == 0:
                    continue # Dissimilarity is 0 if one set is empty

                # Create uniform distributions
                a = torch.ones(num_coords_i, device=device) / num_coords_i
                b = torch.ones(num_coords_j, device=device) / num_coords_j

                # Cost matrix M_ij = ||coords_i_i - coords_j_j||^2
                cost_matrix = torch.cdist(coords_i, coords_j, p=2)**2

                try:
                    # ot.sinkhorn2 requires numpy arrays on CPU
                    a_np = a.cpu().numpy()
                    b_np = b.cpu().numpy()
                    cost_matrix_np = cost_matrix.cpu().numpy()

                    emd_val = ot.sinkhorn2(a_np, b_np, cost_matrix_np, reg).item()
                    dissimilarity_matrix[i, j] = emd_val
                    dissimilarity_matrix[j, i] = emd_val
                except Exception as e:
                    warnings.warn(f"Sinkhorn calculation failed for sample pair ({i}, {j}): {e}")
                    continue

        # Normalize the matrix to a [0, 1] range
        max_val = dissimilarity_matrix.max()
        if max_val > 1e-9:
            dissimilarity_matrix /= max_val

        return dissimilarity_matrix

    def training_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["phenotype"]
        mask = batch.get("mask", None)

        model_outputs = self(x, mask)
        losses = self.compute_loss(model_outputs, y, raw_features=x)

        # --- Log training losses ---
        # Use batch_size for logging step losses, PL handles epoch averages
        batch_size = x.shape[0]
        self.log("train_loss", losses["total_loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train_primary_loss", losses["primary_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if self.ds_enabled:
            self.log("train_ds_loss", losses["deep_supervision_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if self.pw_enabled:
            self.log("train_pw_loss", losses["pw_cosim_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if self.l1_enabled:
            self.log("train_l1_loss", losses["l1_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if self.correlation_enabled:
            self.log("train_corr_loss", losses["correlation_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # --- End log training losses ---

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["phenotype"]
        mask = batch.get("mask", None)

        model_outputs = self(x, mask)
        losses = self.compute_loss(model_outputs, y) # Use compute_loss for consistency

        # --- Log validation losses (averaged over epoch by PL) ---
        batch_size = x.shape[0]
        self.log("val_loss", losses["total_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_primary_loss", losses["primary_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # Log other val losses if needed
        # self.log("val_ds_loss", losses["deep_supervision_loss"], ...)
        # self.log("val_pw_loss", losses["pw_cosim_loss"], ...)
        # self.log("val_l1_loss", losses["l1_loss"], ...)
        # self.log("val_corr_loss", losses["correlation_loss"], ...)

        # Store predictions and labels for epoch-end calculation
        output = {"preds": model_outputs['final_pred'], "labels": y}
        self.validation_step_outputs.append(output)
        return output # Return dict for epoch end


    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
             return # Skip if no outputs

        # Concatenate all predictions and labels from the validation steps
        all_preds = torch.cat([out['preds'] for out in self.validation_step_outputs], dim=0)
        all_labels = torch.cat([out['labels'] for out in self.validation_step_outputs], dim=0)

        # Clear the stored outputs
        self.validation_step_outputs.clear()

        # Calculate Pearson correlation per phenotype
        if all_preds.shape[0] > 1 and all_preds.ndim == 2 and all_labels.ndim == 2 and all_preds.shape == all_labels.shape:
            num_phenotypes = all_preds.shape[1]
            phenotype_names = self.phenotype_names if self.phenotype_names and len(self.phenotype_names) == num_phenotypes else [f"pheno_{i}" for i in range(num_phenotypes)]

            total_pearson = 0.0
            valid_phenotypes = 0
            for i in range(num_phenotypes):
                pred_col = all_preds[:, i].float()
                label_col = all_labels[:, i].float()

                vx = pred_col - torch.mean(pred_col)
                vy = label_col - torch.mean(label_col)
                std_vx = torch.sqrt(torch.sum(vx ** 2))
                std_vy = torch.sqrt(torch.sum(vy ** 2))

                if std_vx > 1e-9 and std_vy > 1e-9:
                    pearson_r = torch.sum(vx * vy) / (std_vx * std_vy)
                    pearson_r = torch.clamp(pearson_r, -1.0, 1.0)
                    # --- 修改：添加 sync_dist=True ---
                    self.log(f'val_pearson_{phenotype_names[i]}', pearson_r, on_epoch=True, sync_dist=True, batch_size=all_preds.shape[0])
                    total_pearson += pearson_r.item()
                    valid_phenotypes += 1
                else:
                    # Log 0 or NaN for constant columns? Let's log 0.
                    self.log(f'val_pearson_{phenotype_names[i]}', 0.0, on_epoch=True, sync_dist=True, batch_size=all_preds.shape[0])

            if valid_phenotypes > 0:
                avg_pearson = total_pearson / valid_phenotypes
                # --- 修改：添加 sync_dist=True ---
                self.log('val_pearson_corr_epoch', avg_pearson, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=all_preds.shape[0])
            else:
                self.log('val_pearson_corr_epoch', 0.0, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=all_preds.shape[0])
        else:
             warnings.warn("Could not calculate validation Pearson correlation due to insufficient data or shape mismatch.")
             self.log('val_pearson_corr_epoch', 0.0, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=all_preds.shape[0])
        # --- 结束修改 ---


    def test_step(self, batch, batch_idx):
        # Similar to validation_step, but log test metrics
        x = batch["features"]
        y = batch["phenotype"]
        mask = batch.get("mask", None)

        model_outputs = self(x, mask)
        losses = self.compute_loss(model_outputs, y) # Use compute_loss

        batch_size = x.shape[0]
        self.log("test_loss", losses["total_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test_primary_loss", losses["primary_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # Optionally calculate and log test Pearson correlation similarly to validation
        # Remember to handle potential distributed testing if applicable
        return {"preds": model_outputs['final_pred'], "labels": y}


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        # Assumes batch is a dictionary containing 'features' and potentially 'mask'
        if isinstance(batch, dict):
            x = batch["features"]
            mask = batch.get("mask", None)
            labels = batch.get("phenotype", None) # Include labels if available
        else:
            # Fallback if batch is not a dict (e.g., tuple)
            x = batch[0]
            mask = batch[2] if len(batch) > 2 else None
            labels = batch[1] if len(batch) > 1 else None
            warnings.warn("Predict step received non-dict batch, assuming structure (features, labels, mask).")

        model_outputs = self(x, mask) # self.model.training will be False here
        
        predictions_dict = {"preds": model_outputs['final_pred']}
        if labels is not None:
            predictions_dict["labels"] = labels # Return labels alongside predictions

        # Add true batch size
        predictions_dict['batch_size'] = torch.tensor(x.shape[0], device=x.device) # Add B_true

        # Extract pooling weights if model is in eval mode (which it is during predict)
        # These keys are added in DNAWhisperModel.forward when not training
        if not self.model.training:
            predictions_dict['embedding_pooling_weights'] = model_outputs.get('embedding_pooling_weights')
            predictions_dict['num_embedding_blocks'] = model_outputs.get('num_embedding_blocks')

            # Derive num_gfi_blocks_list from the shape of gfi_block_features
            gfi_block_features = model_outputs.get('gfi_block_features')
            if gfi_block_features and isinstance(gfi_block_features, list):
                num_gfi_blocks_list = []
                for block_feat_tensor in gfi_block_features:
                    if isinstance(block_feat_tensor, torch.Tensor) and block_feat_tensor.ndim >= 2:
                        # N_i is the second dimension of [B, N_i, Dim_i]
                        num_gfi_blocks_list.append(block_feat_tensor.shape[1]) 
                    else:
                        num_gfi_blocks_list.append(0) # Placeholder if a feature tensor is not as expected
                        warnings.warn(
                            f"Predict_step: Could not determine num_blocks for a GFI block feature. "
                            f"Feature: {type(block_feat_tensor)}, Shape: {block_feat_tensor.shape if isinstance(block_feat_tensor, torch.Tensor) else 'N/A'}"
                        )
                predictions_dict['num_gfi_blocks_list'] = num_gfi_blocks_list
            else:
                predictions_dict['num_gfi_blocks_list'] = []
                if self.model.gfi_former.num_blocks > 0 : # Only warn if GFI blocks are expected
                    warnings.warn("Predict_step: 'gfi_block_features' not found or not a list in model_outputs, "
                                  "num_gfi_blocks_list will be empty.")

            gfi_attention_weights_full = model_outputs.get('attention_weights_full', [])
            gfi_pooling_weights_list = []
            if isinstance(gfi_attention_weights_full, list):
                for block_weights_dict in gfi_attention_weights_full:
                    if isinstance(block_weights_dict, dict):
                        gfi_pooling_weights_list.append(block_weights_dict.get('pooling'))
                    else:
                        gfi_pooling_weights_list.append(None) # Placeholder if structure is unexpected
            predictions_dict['gfi_pooling_weights_list'] = gfi_pooling_weights_list
            
        return predictions_dict


    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_config.get("name", "adamw").lower()
        optimizer_params = self.hparams.optimizer_config.get("params", {})
        lr = optimizer_params.get("lr", 1e-4)
        weight_decay = optimizer_params.get("weight_decay", 0.01)

        # Filter out parameters that don't require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay) # Adam has weight decay too
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(params_to_optimize, lr=lr, weight_decay=weight_decay,
                                        momentum=optimizer_params.get("momentum", 0.9)) # Add momentum for SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler_name = self.hparams.scheduler_config.get("name", "cosine").lower()
        scheduler_params = self.hparams.scheduler_config.get("params", {})

        if scheduler_name == "cosine":
            # 支持两种格式：T_max 和 t_max
            t_max = scheduler_params.get("T_max") or scheduler_params.get("t_max")
            if t_max is None:
                # Estimate T_max based on trainer settings if possible
                if self.trainer and self.trainer.estimated_stepping_batches:
                    t_max = self.trainer.estimated_stepping_batches
                else:
                    # Fallback or raise error if T_max cannot be determined
                    raise ValueError("CosineAnnealingLR requires 'T_max' or 't_max' in scheduler params or trainer access.")

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=scheduler_params.get("eta_min", 0)
            )
            lr_scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params.get("mode", "min"),
                factor=scheduler_params.get("factor", 0.1),
                patience=scheduler_params.get("patience", 10),
                verbose=True # Log when LR changes
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "monitor": scheduler_params.get("monitor", "val_loss"), # Monitor validation loss
                "interval": "epoch",
                "frequency": 1
            }
        elif scheduler_name == "none" or scheduler_name is None:
             return optimizer # No scheduler
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

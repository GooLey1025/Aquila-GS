"""
Metrics for evaluating genomic prediction models.
Includes both regression and classification metrics with support for missing labels.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss with masking for missing labels.
    
    Args:
        predictions: (batch, num_tasks) predicted values
        targets: (batch, num_tasks) target values
        mask: (batch, num_tasks) boolean mask (True = valid, False = missing)
        
    Returns:
        Scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask.float()
    return masked_error.sum() / mask.sum()


def masked_mae_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MAE loss with masking for missing labels.
    
    Args:
        predictions: (batch, num_tasks) predicted values
        targets: (batch, num_tasks) target values
        mask: (batch, num_tasks) boolean mask (True = valid, False = missing)
        
    Returns:
        Scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    absolute_error = torch.abs(predictions - targets)
    masked_error = absolute_error * mask.float()
    return masked_error.sum() / mask.sum()


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute binary cross-entropy loss with masking for missing labels.
    
    Args:
        logits: (batch, num_tasks) predicted logits
        targets: (batch, num_tasks) target labels (0 or 1)
        mask: (batch, num_tasks) boolean mask (True = valid, False = missing)
        
    Returns:
        Scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    masked_bce = bce * mask.float()
    return masked_bce.sum() / mask.sum()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty weighting.
    Automatically learns task weights based on homoscedastic uncertainty.
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(
        self,
        num_regression_tasks: int = 0,
        num_classification_tasks: int = 0,
        loss_type: str = 'mse',  # 'mse' or 'mae' for regression
        uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.num_regression_tasks = num_regression_tasks
        self.num_classification_tasks = num_classification_tasks
        self.loss_type = loss_type
        self.uncertainty_weighting = uncertainty_weighting
        
        # Learnable uncertainty parameters (log variance)
        if uncertainty_weighting:
            if num_regression_tasks > 0:
                self.log_vars_reg = nn.Parameter(torch.zeros(num_regression_tasks))
            if num_classification_tasks > 0:
                self.log_vars_cls = nn.Parameter(torch.zeros(num_classification_tasks))
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dict with 'regression' and/or 'classification' predictions
            targets: Dict with 'regression' and/or 'classification' targets
            masks: Dict with 'regression' and/or 'classification' masks
            
        Returns:
            total_loss: Weighted sum of all task losses
            loss_dict: Dictionary with individual task losses
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Regression tasks
        if 'regression' in predictions and self.num_regression_tasks > 0:
            reg_pred = predictions['regression']
            reg_target = targets['regression']
            reg_mask = masks['regression']
            
            for i in range(self.num_regression_tasks):
                pred_i = reg_pred[:, i]
                target_i = reg_target[:, i]
                mask_i = reg_mask[:, i]
                
                if self.loss_type == 'mse':
                    loss_i = masked_mse_loss(pred_i, target_i, mask_i)
                else:  # mae
                    loss_i = masked_mae_loss(pred_i, target_i, mask_i)
                
                loss_dict[f'regression_{i}'] = loss_i.item()
                
                if self.uncertainty_weighting:
                    # Uncertainty weighting: L = (1/2σ²)L_i + log(σ)
                    precision = torch.exp(-self.log_vars_reg[i])
                    weighted_loss = 0.5 * precision * loss_i + 0.5 * self.log_vars_reg[i]
                else:
                    weighted_loss = loss_i
                
                total_loss = total_loss + weighted_loss
        
        # Classification tasks
        if 'classification' in predictions and self.num_classification_tasks > 0:
            cls_logits = predictions['classification']
            cls_target = targets['classification']
            cls_mask = masks['classification']
            
            for i in range(self.num_classification_tasks):
                logits_i = cls_logits[:, i]
                target_i = cls_target[:, i]
                mask_i = cls_mask[:, i]
                
                loss_i = masked_bce_loss(logits_i, target_i, mask_i)
                loss_dict[f'classification_{i}'] = loss_i.item()
                
                if self.uncertainty_weighting:
                    precision = torch.exp(-self.log_vars_cls[i])
                    weighted_loss = 0.5 * precision * loss_i + 0.5 * self.log_vars_cls[i]
                else:
                    weighted_loss = loss_i
                
                total_loss = total_loss + weighted_loss
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


class MetricsCalculator:
    """Calculate evaluation metrics for regression and classification tasks."""
    
    @staticmethod
    def compute_regression_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute regression metrics (MSE, MAE, R², Pearson correlation).
        
        Args:
            predictions: (n_samples, n_tasks) array
            targets: (n_samples, n_tasks) array
            mask: (n_samples, n_tasks) boolean array
            
        Returns:
            Dictionary of metrics per task and averaged
        """
        n_tasks = predictions.shape[1]
        metrics = {}
        
        for i in range(n_tasks):
            valid_idx = mask[:, i]
            if valid_idx.sum() == 0:
                continue
            
            pred_i = predictions[valid_idx, i]
            target_i = targets[valid_idx, i]
            
            mse = mean_squared_error(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            r2 = r2_score(target_i, pred_i)
            
            # Pearson correlation
            pearson = np.corrcoef(pred_i, target_i)[0, 1] if len(pred_i) > 1 else 0.0
            
            metrics[f'task_{i}_mse'] = mse
            metrics[f'task_{i}_rmse'] = np.sqrt(mse)
            metrics[f'task_{i}_mae'] = mae
            metrics[f'task_{i}_r2'] = r2
            metrics[f'task_{i}_pearson'] = pearson
        
        # Average metrics
        if n_tasks > 0:
            metrics['avg_mse'] = np.mean([metrics.get(f'task_{i}_mse', 0) for i in range(n_tasks)])
            metrics['avg_rmse'] = np.mean([metrics.get(f'task_{i}_rmse', 0) for i in range(n_tasks)])
            metrics['avg_mae'] = np.mean([metrics.get(f'task_{i}_mae', 0) for i in range(n_tasks)])
            metrics['avg_r2'] = np.mean([metrics.get(f'task_{i}_r2', 0) for i in range(n_tasks)])
            metrics['avg_pearson'] = np.mean([metrics.get(f'task_{i}_pearson', 0) for i in range(n_tasks)])
        
        return metrics
    
    @staticmethod
    def compute_classification_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute classification metrics (accuracy, precision, recall, F1, AUC).
        
        Args:
            logits: (n_samples, n_tasks) array of logits
            targets: (n_samples, n_tasks) array of labels (0 or 1)
            mask: (n_samples, n_tasks) boolean array
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics per task and averaged
        """
        n_tasks = logits.shape[1]
        metrics = {}
        
        # Convert logits to probabilities
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs >= threshold).astype(int)
        
        for i in range(n_tasks):
            valid_idx = mask[:, i]
            if valid_idx.sum() == 0:
                continue
            
            pred_i = preds[valid_idx, i]
            target_i = targets[valid_idx, i].astype(int)
            prob_i = probs[valid_idx, i]
            
            # Basic metrics
            acc = accuracy_score(target_i, pred_i)
            
            # Handle edge cases for precision/recall
            try:
                prec = precision_score(target_i, pred_i, zero_division=0)
                rec = recall_score(target_i, pred_i, zero_division=0)
                f1 = f1_score(target_i, pred_i, zero_division=0)
            except:
                prec = rec = f1 = 0.0
            
            # AUC metrics (if both classes present)
            try:
                if len(np.unique(target_i)) > 1:
                    auc = roc_auc_score(target_i, prob_i)
                    ap = average_precision_score(target_i, prob_i)
                else:
                    auc = ap = 0.0
            except:
                auc = ap = 0.0
            
            metrics[f'task_{i}_accuracy'] = acc
            metrics[f'task_{i}_precision'] = prec
            metrics[f'task_{i}_recall'] = rec
            metrics[f'task_{i}_f1'] = f1
            metrics[f'task_{i}_auc'] = auc
            metrics[f'task_{i}_ap'] = ap
        
        # Average metrics
        if n_tasks > 0:
            metrics['avg_accuracy'] = np.mean([metrics.get(f'task_{i}_accuracy', 0) for i in range(n_tasks)])
            metrics['avg_f1'] = np.mean([metrics.get(f'task_{i}_f1', 0) for i in range(n_tasks)])
            metrics['avg_auc'] = np.mean([metrics.get(f'task_{i}_auc', 0) for i in range(n_tasks)])
        
        return metrics


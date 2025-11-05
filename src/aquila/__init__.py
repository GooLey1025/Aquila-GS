"""
Aquila: Deep Learning Framework for Genomic Selection

A powerful PyTorch-based framework for genomic prediction using SNP data.
"""

__version__ = "0.1.0"

from .snpnn import SNPNeuralNetwork, create_model_from_config
from .trainer import SNPTrainer, EarlyStopping
from .metrics import (
    MultiTaskLoss,
    MetricsCalculator,
    masked_mse_loss,
    masked_mae_loss,
    masked_bce_loss,
)
from .layers import (
    SNPEmbedding,
    GlobalPooling,
    RegressionHead,
    ClassificationHead,
)
from .data_utils import (
    SNPDataset,
    create_data_loaders,
    parse_genotype_file,
    parse_phenotype_file,
)
from .utils import (
    set_seed,
    save_config,
    load_config,
    print_model_summary,
)

__all__ = [
    # Main components
    "SNPNeuralNetwork",
    "create_model_from_config",
    "SNPTrainer",
    "EarlyStopping",
    
    # Metrics and losses
    "MultiTaskLoss",
    "MetricsCalculator",
    "masked_mse_loss",
    "masked_mae_loss",
    "masked_bce_loss",
    
    # Layers
    "SNPEmbedding",
    "TransformerBlock",
    "GlobalPooling",
    "RegressionHead",
    "ClassificationHead",
    
    # Data utilities
    "SNPDataset",
    "create_data_loaders",
    "parse_genotype_file",
    "parse_phenotype_file",
    
    # Utilities
    "set_seed",
    "save_config",
    "load_config",
    "print_model_summary",
]


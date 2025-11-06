"""
Aquila: Deep Learning Framework for Genomic Selection

A powerful PyTorch-based framework for genomic prediction using SNP data.
"""

__version__ = "0.1.0"

from .varnn import VariantsNeuralNetwork, create_model_from_config
# Backward compatibility alias
SNPNeuralNetwork = VariantsNeuralNetwork
from .trainer import VarTrainer, EarlyStopping
# Backward compatibility alias
SNPTrainer = VarTrainer
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
    VariantsDataset,
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
    "VariantsNeuralNetwork",
    "SNPNeuralNetwork",  # Backward compatibility
    "create_model_from_config",
    "VarTrainer",
    "SNPTrainer",  # Backward compatibility
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
    "VariantsDataset",
    "create_data_loaders",
    "parse_genotype_file",
    "parse_phenotype_file",
    
    # Utilities
    "set_seed",
    "save_config",
    "load_config",
    "print_model_summary",
]


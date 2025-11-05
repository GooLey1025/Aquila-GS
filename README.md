# Aquila: Deep Learning for Genomic Selection

Aquila is a powerful deep learning framework for genomic prediction using SNP (Single Nucleotide Polymorphism) data. It supports multi-task learning with both regression and classification tasks, handles missing data elegantly, and uses state-of-the-art transformer architectures.

## Features

- **Multi-task Learning**: Simultaneously predict multiple phenotypic traits (regression and/or classification)
- **Missing Data Handling**: Built-in support for missing SNP genotypes and phenotype labels
- **Flexible Architecture**: Choose between Transformer or MLP backbone networks
- **Uncertainty Weighting**: Automatic task weighting based on learned uncertainty
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Easy Configuration**: YAML-based configuration system

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/aquila.git
cd aquila

# Install in editable mode
pip3 install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- PyYAML >= 5.4.0
- tqdm >= 4.62.0

## Quick Start

### 1. Prepare Your Data

**Genotype File Format** (`data.geno`):
```
#CHROM  POS     REF  ALT  Sample001  Sample002  Sample003  ...
1       96804   C    T    C          C          C          ...
1       161524  A    G    A          A          A          ...
```

- Values: `A`, `C`, `G`, `T` (nucleotides), `H` (heterozygous)
- Missing values are automatically detected
- Converted to: -1 (homozygous ref), 0 (heterozygous), 1 (homozygous alt), 3 (missing)

**Phenotype File Format** (`data.pheno`):
```
LINE            Plant_height  Seed_length  Protein_content  ...
Sample001       139.33        9.49         8.4              ...
Sample002       128.89        8.33         7.75             ...
```

- First column: Sample IDs (must match genotype file)
- Other columns: Phenotypic traits
- Missing values (NA) are supported

### 2. Create Configuration File

Create `params.yaml`:

```yaml
data:
  geno_file: "data/data.geno"
  pheno_file: "data/data.pheno"
  # Specify which traits are for regression vs classification
  regression_tasks:
    - "Plant_height"
    - "Seed_length"
    - "Protein_content"
  classification_tasks: []  # Optional: binary classification tasks

model:
  embed_dim: 128                    # Embedding dimension
  num_transformer_layers: 4         # Number of transformer blocks
  num_heads: 8                      # Number of attention heads
  d_ff: 512                         # Feed-forward dimension
  dropout: 0.1                      # Dropout rate
  trunk_type: "transformer"         # "transformer" or "mlp"
  pool_type: "attention"            # "mean", "max", or "attention"
  regression_hidden_dim: 256        # Hidden layer size for regression head
  classification_hidden_dim: 256    # Hidden layer size for classification head

train:
  batch_size: 32                    # Batch size
  num_epochs: 100                   # Maximum number of epochs
  learning_rate: 1.0e-4             # Learning rate
  weight_decay: 1.0e-5              # L2 regularization
  loss_type: "mse"                  # "mse" or "mae" for regression
  uncertainty_weighting: true       # Use uncertainty-based task weighting
  early_stopping_patience: 20       # Early stopping patience
  gradient_clip_norm: 1.0           # Gradient clipping norm
  val_split: 0.2                    # Validation split ratio
  test_split: 0.0                   # Test split ratio
  num_workers: 4                    # Data loading workers
```

### 3. Train the Model

```bash
# Using the command-line interface
aquila-train --config params.yaml --output ./outputs

# Or using Python
python -m aquila.scripts.aquila_snp_train --config params.yaml --output ./outputs
```

### 4. Use in Python Code

```python
from aquila import SNPNeuralNetwork
from aquila.trainer import SNPTrainer
from aquila.data_utils import create_data_loaders
from aquila.utils import set_seed

# Set random seed
set_seed(42)

# Load data
train_loader, val_loader, _ = create_data_loaders(
    geno_path="data/data.geno",
    pheno_path="data/data.pheno",
    regression_tasks=["Plant_height", "Seed_length"],
    batch_size=32,
    val_split=0.2
)

# Create model
model = SNPNeuralNetwork(
    seq_length=10000,  # Number of SNPs
    embed_dim=128,
    num_transformer_layers=4,
    num_heads=8,
    regression_tasks=["Plant_height", "Seed_length"],
    classification_tasks=[]
)

# Train
trainer = SNPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_regression_tasks=2,
    learning_rate=1e-4
)

history = trainer.train(num_epochs=100)
```

## Architecture

Aquila uses a modern deep learning architecture designed for genomic data:

```
Input SNPs → Embedding → Positional Encoding → Transformer Layers → Pooling → Task Heads
  {-1,0,1,3}    (128d)         (learned)           (4 layers)      (attention)  (regression/classification)
```

### Key Components

1. **SNP Embedding**: Converts discrete genotypes to dense vectors
2. **Transformer Blocks**: Captures long-range dependencies between SNPs
3. **Global Pooling**: Aggregates sequence information (attention-based by default)
4. **Task-Specific Heads**: Separate output layers for each task
5. **Uncertainty Weighting**: Learns optimal task weights during training

## Advanced Usage

### Custom Data Loading

```python
from aquila.data_utils import parse_genotype_file, parse_phenotype_file, SNPDataset

# Parse files
snp_matrix, sample_ids, snp_ids = parse_genotype_file("data/data.geno")
pheno_df, reg_cols, cls_cols = parse_phenotype_file("data/data.pheno")

# Create dataset
dataset = SNPDataset(snp_matrix, pheno_df, sample_ids, reg_cols, cls_cols)
```

### Model Evaluation

```python
from aquila.metrics import MetricsCalculator

# After training, evaluate on test set
metrics_calc = MetricsCalculator()

# For regression
reg_metrics = metrics_calc.compute_regression_metrics(
    predictions, targets, mask
)
print(f"R²: {reg_metrics['avg_r2']:.4f}")
print(f"Pearson: {reg_metrics['avg_pearson']:.4f}")

# For classification
cls_metrics = metrics_calc.compute_classification_metrics(
    logits, targets, mask
)
print(f"AUC: {cls_metrics['avg_auc']:.4f}")
```

### Load Trained Model

```python
import torch
from aquila import SNPNeuralNetwork

# Load checkpoint
checkpoint = torch.load("outputs/checkpoints/best_checkpoint.pt")

# Recreate model and load weights
model = SNPNeuralNetwork(...)  # Use same architecture
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(snp_tensor)
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster training
   ```bash
   aquila-train --config params.yaml --device cuda
   ```

2. **Batch Size**: Increase for better GPU utilization (if memory allows)

3. **Number of Workers**: Increase `num_workers` for faster data loading

4. **Mixed Precision**: For even faster training (requires code modification)

5. **Gradient Checkpointing**: For training with very long sequences

## Citation

If you use Aquila in your research, please cite:

```bibtex
@software{aquila2025,
  title={Aquila: Deep Learning Framework for Genomic Selection},
  author={Aquila Team},
  year={2025},
  url={https://github.com/yourusername/aquila}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact us at aquila@example.com.


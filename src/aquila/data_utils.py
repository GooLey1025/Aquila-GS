"""
Data utilities for loading and preprocessing SNP and phenotype data.
"""

import time
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
import json
import hashlib
import pickle


def parse_genotype_file(geno_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse genotype file in the format shown in the image.
    
    File format:
    #CHROM POS REF ALT Sample001 Sample002 ...
    Values: A, C, G, T, H (heterozygous)
    
    Convert to SNP encoding:
    - Homozygous reference (e.g., C/C matches REF): 1
    - Heterozygous (H or different from both REF and ALT): 0
    - Homozygous alternate (e.g., T/T matches ALT): 1
    - Missing: 3
    
    Args:
        geno_path: Path to genotype file
        
    Returns:
        snp_matrix: (n_samples, n_snps) array with values {0, 1, 2, 3}
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs (CHROM_POS)
    """
    print(f"Loading genotype file: {geno_path}")
    
    # Read the file
    df = pd.read_csv(geno_path, sep='\s+', dtype=str)
    
    # Extract metadata columns
    metadata_cols = ['#CHROM', 'POS', 'REF', 'ALT']
    sample_ids = [col for col in df.columns if col not in metadata_cols]
    
    n_snps = len(df)
    n_samples = len(sample_ids)
    
    print(f"Found {n_snps} SNPs and {n_samples} samples")
    
    # Create SNP IDs
    snp_ids = [f"{row['#CHROM']}_{row['POS']}" for _, row in df.iterrows()]
    
    # Initialize SNP matrix
    snp_matrix = np.zeros((n_samples, n_snps), dtype=np.int8)
    
    # Convert genotypes to SNP encoding
    for snp_idx, (_, row) in enumerate(df.iterrows()):
        ref = row['REF']
        alt = row['ALT']
        
        for sample_idx, sample_id in enumerate(sample_ids):
            genotype = row[sample_id]
            
            if pd.isna(genotype) or genotype == '.' or genotype == 'N':
                # Missing
                snp_matrix[sample_idx, snp_idx] = 3
            elif genotype == 'H':
                # Heterozygous
                snp_matrix[sample_idx, snp_idx] = 2
            elif genotype == ref:
                # Homozygous reference
                snp_matrix[sample_idx, snp_idx] = 1
            elif genotype == alt:
                # Homozygous alternate
                snp_matrix[sample_idx, snp_idx] = 0
            else:
                raise ValueError(f"Invalid genotype: {genotype} for SNP {snp_idx} in sample {sample_idx}")
    
    return snp_matrix, sample_ids, snp_ids


def parse_phenotype_file(
    pheno_path: str,
    classification_tasks: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Parse phenotype file with simple logic:
    - Default: All traits are regression tasks
    - If classification_tasks specified: Those become classification, others remain regression
    
    Args:
        pheno_path: Path to phenotype file
        regression_tasks: Ignored (kept for API compatibility)
        classification_tasks: List of column names for binary classification tasks
        
    Returns:
        pheno_df: DataFrame with phenotypes
        regression_cols: List of regression column names
        classification_cols: List of classification column names
    """
    print(f"Loading phenotype file: {pheno_path}")
    
    # Read phenotype file
    pheno_df = pd.read_csv(pheno_path, sep='\t')
    
    # First column should be sample IDs
    sample_col = pheno_df.columns[0]
    pheno_df = pheno_df.rename(columns={sample_col: 'sample_id'})
    
    # Get all trait columns (exclude sample_id)
    trait_cols = [col for col in pheno_df.columns if col != 'sample_id']
    
    # Task assignment: Default all as regression, specified ones as classification
    if classification_tasks is None:
        classification_cols = []
    else:
        classification_cols = classification_tasks
    
    # All non-classification traits are regression
    regression_cols = [col for col in trait_cols if col not in classification_cols]
    
    # Validate that all specified classification tasks exist
    if classification_cols:
        missing_tasks = set(classification_cols) - set(trait_cols)
        if missing_tasks:
            raise ValueError(f"Classification tasks not found in phenotype file: {missing_tasks}")
    
    print(f"    Found {len(pheno_df)} samples with {len(trait_cols)} traits")
    print(f"    Regression tasks ({len(regression_cols)}): {regression_cols}")
    print(f"    Classification tasks ({len(classification_cols)}): {classification_cols}")
    
    return pheno_df, regression_cols, classification_cols


class VariantsDataset(Dataset):
    """PyTorch Dataset for variant data with multi-task phenotypes.
    
    Supports both single-branch (tensor) and multi-branch (dict) inputs.
    """
    
    def __init__(
        self,
        snp_matrix,  # Can be np.ndarray or dict
        phenotype_df: pd.DataFrame,
        sample_ids: List[str],
        regression_cols: List[str],
        classification_cols: List[str],
        normalize_regression: bool = True,
        regression_means: Optional[np.ndarray] = None,
        regression_stds: Optional[np.ndarray] = None,
    ):
        """
        Args:
            snp_matrix: (n_samples, n_variants) array or dict of arrays for multi-branch
                       Dict format: {'snp': array, 'indel': array, 'sv': array}
            phenotype_df: DataFrame with phenotypes
            sample_ids: List of sample IDs matching matrix rows
            regression_cols: List of regression task columns
            classification_cols: List of classification task columns
            normalize_regression: Whether to apply z-score normalization to regression targets
            regression_means: Pre-computed means for normalization (for val/test sets)
            regression_stds: Pre-computed stds for normalization (for val/test sets)
        """
        # Handle multi-branch input (dict) or single-branch (array)
        self.is_multi_branch = isinstance(snp_matrix, dict)
        
        if self.is_multi_branch:
            # Multi-branch: store each variant type separately
            self.variant_matrices = {}
            for variant_type, matrix in snp_matrix.items():
                if matrix.ndim == 2:
                    self.variant_matrices[variant_type] = torch.from_numpy(matrix).long()
                elif matrix.ndim == 3:
                    self.variant_matrices[variant_type] = torch.from_numpy(matrix).float()
                else:
                    raise ValueError(f"Unexpected matrix shape for {variant_type}: {matrix.shape}")
            self.snp_matrix = None
        else:
            # Single-branch: handle both 2D (token) and 3D (diploid_onehot) encodings
            if snp_matrix.ndim == 2:
                # Token encoding: (n_samples, n_snps) -> convert to long
                self.snp_matrix = torch.from_numpy(snp_matrix).long()
            elif snp_matrix.ndim == 3:
                # Diploid one-hot encoding: (n_samples, n_snps, 8) -> keep as float
                self.snp_matrix = torch.from_numpy(snp_matrix).float()
            else:
                raise ValueError(f"Unexpected snp_matrix shape: {snp_matrix.shape}")
            self.variant_matrices = None
        
        self.sample_ids = sample_ids
        self.regression_cols = regression_cols
        self.classification_cols = classification_cols
        self.normalize_regression = normalize_regression
        self.regression_means = regression_means
        self.regression_stds = regression_stds
        
        # Match samples between genotype and phenotype
        self.valid_indices = []
        self.regression_targets = []
        self.regression_masks = []
        self.classification_targets = []
        self.classification_masks = []
        
        for idx, sample_id in enumerate(sample_ids):
            if sample_id in phenotype_df['sample_id'].values:
                self.valid_indices.append(idx)
                
                # Get phenotype row
                pheno_row = phenotype_df[phenotype_df['sample_id'] == sample_id].iloc[0]
                
                # Regression targets
                if regression_cols:
                    reg_values = []
                    reg_mask = []
                    for col in regression_cols:
                        val = pheno_row[col]
                        if pd.isna(val):
                            reg_values.append(0.0)  # Placeholder
                            reg_mask.append(False)
                        else:
                            reg_values.append(float(val))
                            reg_mask.append(True)
                    self.regression_targets.append(reg_values)
                    self.regression_masks.append(reg_mask)
                
                # Classification targets
                if classification_cols:
                    cls_values = []
                    cls_mask = []
                    for col in classification_cols:
                        val = pheno_row[col]
                        if pd.isna(val):
                            cls_values.append(0.0)  # Placeholder
                            cls_mask.append(False)
                        else:
                            cls_values.append(float(val))
                            cls_mask.append(True)
                    self.classification_targets.append(cls_values)
                    self.classification_masks.append(cls_mask)
        
        # Convert to tensors
        if regression_cols:
            self.regression_targets = torch.tensor(self.regression_targets, dtype=torch.float32)
            self.regression_masks = torch.tensor(self.regression_masks, dtype=torch.bool)
        
            # Apply z-score normalization for regression targets
            if normalize_regression:
                if regression_means is None or regression_stds is None:
                    # Compute statistics from this dataset (training set)
                    self.regression_means = torch.zeros(len(regression_cols))
                    self.regression_stds = torch.ones(len(regression_cols))
                    
                    for i in range(len(regression_cols)):
                        valid_mask = self.regression_masks[:, i]
                        if valid_mask.sum() > 0:
                            valid_values = self.regression_targets[valid_mask, i]
                            self.regression_means[i] = valid_values.mean()
                            self.regression_stds[i] = valid_values.std()
                            if self.regression_stds[i] < 1e-6:  # Avoid division by zero
                                self.regression_stds[i] = 1.0
                else:
                    # Use provided statistics (for validation/test sets)
                    self.regression_means = torch.tensor(regression_means, dtype=torch.float32)
                    self.regression_stds = torch.tensor(regression_stds, dtype=torch.float32)
                
                # Normalize: z = (x - mean) / std
                for i in range(len(regression_cols)):
                    valid_mask = self.regression_masks[:, i]
                    if valid_mask.sum() > 0:
                        self.regression_targets[valid_mask, i] = (
                            self.regression_targets[valid_mask, i] - self.regression_means[i]
                        ) / self.regression_stds[i]
                
        
        if classification_cols:
            self.classification_targets = torch.tensor(self.classification_targets, dtype=torch.float32)
            self.classification_masks = torch.tensor(self.classification_masks, dtype=torch.bool)
        
        print(f"\nCreated dataset with {len(self.valid_indices)} valid samples")
        if regression_cols:
            valid_reg = self.regression_masks.sum(dim=0)
            print(f"Regression tasks - valid samples per task: {valid_reg.tolist()}")
        if classification_cols:
            valid_cls = self.classification_masks.sum(dim=0)
            print(f"Classification tasks - valid samples per task: {valid_cls.tolist()}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        item = {'sample_id': self.sample_ids[real_idx]}
        
        # Add variant data (single-branch or multi-branch)
        if self.is_multi_branch:
            # Multi-branch: return dict of variant types
            for variant_type, matrix in self.variant_matrices.items():
                item[variant_type] = matrix[real_idx]
        else:
            # Single-branch: return single tensor
            item['snp'] = self.snp_matrix[real_idx]
        
        # Add phenotype targets
        if self.regression_cols:
            item['regression_targets'] = self.regression_targets[idx]
            item['regression_mask'] = self.regression_masks[idx]
        
        if self.classification_cols:
            item['classification_targets'] = self.classification_targets[idx]
            item['classification_mask'] = self.classification_masks[idx]
        
        return item


def _generate_cache_key(geno_path: str, pheno_path: str, encoding_type: str, 
                        classification_tasks: Optional[List[str]], val_split: float, 
                        test_split: float, normalize_regression: bool, 
                        skew_threshold: float = 2.0) -> str:
    """Generate unique cache key based on data configuration."""
    config_dict = {
        'geno_path': str(geno_path),
        'geno_mtime': os.path.getmtime(geno_path) if os.path.exists(geno_path) else 0,
        'pheno_path': str(pheno_path),
        'pheno_mtime': os.path.getmtime(pheno_path) if os.path.exists(pheno_path) else 0,
        'encoding_type': encoding_type,
        'classification_tasks': classification_tasks,
        'val_split': val_split,
        'test_split': test_split,
        'normalize_regression': normalize_regression,
        'skew_threshold': skew_threshold,
    }
    
    key_string = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def _save_data_cache(cache_dir: Path, train_loader: DataLoader, 
                     val_loader: Optional[DataLoader], test_loader: Optional[DataLoader], 
                     normalization_stats: Optional[Dict], cache_key: str):
    """Save data loaders and stats to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving data cache to: {cache_dir}")
    
    # Extract dataset from DataLoader and save
    with open(cache_dir / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_loader.dataset, f)
    print(f"  ✓ Saved train dataset ({len(train_loader.dataset)} samples)")
    
    if val_loader:
        with open(cache_dir / 'val_dataset.pkl', 'wb') as f:
            pickle.dump(val_loader.dataset, f)
        print(f"  ✓ Saved val dataset ({len(val_loader.dataset)} samples)")
    
    if test_loader:
        with open(cache_dir / 'test_dataset.pkl', 'wb') as f:
            pickle.dump(test_loader.dataset, f)
        print(f"  ✓ Saved test dataset ({len(test_loader.dataset)} samples)")
    
    # Save normalization stats
    with open(cache_dir / 'normalization_stats.pkl', 'wb') as f:
        pickle.dump(normalization_stats, f)
    print(f"  ✓ Saved normalization statistics")
    
    # Save cache config
    with open(cache_dir / 'data_config.json', 'w') as f:
        json.dump({'cache_key': cache_key}, f)
    print(f"  ✓ Cache key: {cache_key}\n")


def _load_cached_loader(cache_path: Path, batch_size: int, num_workers: int = 4, 
                        shuffle: bool = True) -> Optional[DataLoader]:
    """Load dataset from cache and create DataLoader."""
    if not cache_path.exists():
        return None
    
    with open(cache_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return DataLoader(dataset, batch_size=batch_size, 
                     shuffle=shuffle, num_workers=num_workers)


def detect_skewed_phenotypes(pheno_df: pd.DataFrame, regression_tasks: List[str], 
                            skew_threshold: float = 2.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detect highly skewed phenotypes and apply log transformation.
    
    Args:
        pheno_df: DataFrame with phenotype data
        regression_tasks: List of regression task column names
        skew_threshold: Threshold for absolute skewness to trigger log transform
    
    Returns:
        transformed_df: DataFrame with log-transformed phenotypes
        log_transformed_tasks: List of task names that were log-transformed
    """
    from scipy.stats import skew
    
    log_transformed_tasks = []
    transformed_df = pheno_df.copy()
    
    print(f"\nDetecting skewed phenotypes (threshold: |skewness| > {skew_threshold}):")
    
    for task in regression_tasks:
        values = pheno_df[task].dropna()
        if len(values) < 3:
            continue
        
        skewness = skew(values)
        if abs(skewness) > skew_threshold:
            # Apply log(x + 1) transformation
            transformed_df[task] = np.log1p(pheno_df[task])
            log_transformed_tasks.append(task)
            
            print(f"  [{task}] Skewness={skewness:.3f} -> Applied log(x+1) transform")
    
    if not log_transformed_tasks:
        print(f"  No phenotypes required log transformation")
    
    return transformed_df, log_transformed_tasks


def create_data_loaders(
    geno_path: str,
    pheno_path: str,
    classification_tasks: Optional[List[str]] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    normalize_regression: bool = True,
    encoding_type: str = 'token',
    cache_dir: Optional[str] = None,
    data_restart: bool = False,
    skew_threshold: float = 2.0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Optional[Dict]]:
    """
    Create train/val/test data loaders with z-score normalization for regression targets.
    
    Args:
        geno_path: Path to genotype file
        pheno_path: Path to phenotype file
        classification_tasks: List of classification task column names
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of data loading workers
        normalize_regression: Apply z-score normalization to regression targets
        encoding_type: Encoding strategy ('token' or 'diploid_onehot')
        cache_dir: Directory to save/load data cache (None = no caching)
        data_restart: If True, ignore cache and re-process data
        skew_threshold: Threshold for absolute skewness to trigger log(x+1) transformation
        
    Returns:
        train_loader, val_loader, test_loader, normalization_stats
    """
    # Check for cached data if cache_dir is provided
    if cache_dir is not None and not data_restart:
        cache_path = Path(cache_dir) / 'data_cache'
        cache_config_file = cache_path / 'data_config.json'
        
        # Generate cache key
        cache_key = _generate_cache_key(
            geno_path, pheno_path, encoding_type, classification_tasks,
            val_split, test_split, normalize_regression, skew_threshold
        )
        
        # Check if valid cache exists
        if cache_path.exists() and cache_config_file.exists():
            with open(cache_config_file, 'r') as f:
                cached_config = json.load(f)
            
            if cached_config.get('cache_key') == cache_key:
                print("\n🔄 Found valid data cache, loading from disk...")
                print(f"   Cache directory: {cache_path}")
                
                # Load from cache
                train_loader = _load_cached_loader(
                    cache_path / 'train_dataset.pkl', batch_size, num_workers, shuffle=True
                )
                val_loader = _load_cached_loader(
                    cache_path / 'val_dataset.pkl', batch_size, num_workers, shuffle=False
                )
                test_loader = _load_cached_loader(
                    cache_path / 'test_dataset.pkl', batch_size, num_workers, shuffle=False
                )
                
                with open(cache_path / 'normalization_stats.pkl', 'rb') as f:
                    normalization_stats = pickle.load(f)
                
                print(f"   ✓ Loaded train: {len(train_loader.dataset)} samples")
                if val_loader:
                    print(f"   ✓ Loaded val: {len(val_loader.dataset)} samples")
                if test_loader:
                    print(f"   ✓ Loaded test: {len(test_loader.dataset)} samples")
                print(f"   ✓ Cache loading complete!\n")
                
                return train_loader, val_loader, test_loader, normalization_stats
            else:
                print(f"\n⚠️  Cache key mismatch, will re-process data")
        else:
            print(f"\n📦 No cache found, will process data and create cache")
    
    # Load data using specified encoding
    from aquila.encoding import parse_genotype_file as parse_geno_with_encoding
    
    # Check if multi-branch encoding
    multi_branch_encodings = ['snp_indel_vcf', 'snp_indel_sv_vcf']
    is_multi_branch = encoding_type in multi_branch_encodings
    
    if is_multi_branch:
        # Multi-branch VCF: returns dict
        variant_data = parse_geno_with_encoding(geno_path, encoding_type)
        # Extract sample IDs from first variant type
        first_variant_type = list(variant_data.keys())[0]
        _, sample_ids, _ = variant_data[first_variant_type]
        
        # Create dict of matrices (n_samples, n_variants, 8)
        snp_matrix = {vtype: data[0] for vtype, data in variant_data.items()}
    else:
        # Single-branch: returns tuple
        result = parse_geno_with_encoding(geno_path, encoding_type)
        if encoding_type == 'snp_vcf':
            # snp_vcf returns tuple
            snp_matrix, sample_ids, snp_ids = result
        else:
            # token or diploid_onehot
            snp_matrix, sample_ids, snp_ids = result

    pheno_df, regression_cols, classification_cols = parse_phenotype_file(
        pheno_path, classification_tasks
    )
    
    # Detect and transform skewed phenotypes
    log_transformed_tasks = []
    if regression_cols and skew_threshold > 0:
        pheno_df, log_transformed_tasks = detect_skewed_phenotypes(
            pheno_df, regression_cols, skew_threshold
        )
    
    # First create dataset WITHOUT normalization to get valid indices
    dataset_unnormalized = VariantsDataset(
        snp_matrix, pheno_df, sample_ids,
        regression_cols, classification_cols,
        normalize_regression=False  # Don't normalize yet
    )
    
    # Split dataset
    n_samples = len(dataset_unnormalized)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * val_split)
    train_size = n_samples - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Compute normalization statistics from training set only
    normalization_stats = None
    regression_means_np = None
    regression_stds_np = None
    
    if normalize_regression and regression_cols:
        # Get training sample IDs
        train_sample_ids = [dataset_unnormalized.sample_ids[i] for i in train_indices]
        
        # Compute statistics from original phenotype data for training samples only
        regression_means_np = np.zeros(len(regression_cols))
        regression_stds_np = np.ones(len(regression_cols))
        
        for i, col in enumerate(regression_cols):
            # Get original values for training samples
            train_pheno_subset = pheno_df[pheno_df['sample_id'].isin(train_sample_ids)]
            valid_values = train_pheno_subset[col].dropna().values
            
            if len(valid_values) > 0:
                regression_means_np[i] = np.mean(valid_values)
                regression_stds_np[i] = np.std(valid_values)
                if regression_stds_np[i] < 1e-6:  # Avoid division by zero
                    regression_stds_np[i] = 1.0
        
        normalization_stats = {
            'regression_means': regression_means_np,
            'regression_stds': regression_stds_np,
            'regression_tasks': regression_cols,
            'log_transformed_tasks': log_transformed_tasks,
            'skew_threshold': skew_threshold,
        }
        
        print(f"Computed normalization statistics from training set:")
        print(f"  Means: {regression_means_np}")
        print(f"  Stds: {regression_stds_np}")
    
    # Now create normalized datasets using the training statistics
    # Training set: compute statistics from training data
    train_dataset = VariantsDataset(
        snp_matrix, pheno_df, sample_ids,
        regression_cols, classification_cols,
        normalize_regression=normalize_regression,
        regression_means=regression_means_np,
        regression_stds=regression_stds_np
    )
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_loader = None
    if test_size > 0:
        test_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"\nData split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Save to cache if cache_dir is provided
    if cache_dir is not None:
        cache_path = Path(cache_dir) / 'data_cache'
        cache_key = _generate_cache_key(
            geno_path, pheno_path, encoding_type, classification_tasks,
            val_split, test_split, normalize_regression, skew_threshold
        )
        _save_data_cache(cache_path, train_loader, val_loader, test_loader, 
                        normalization_stats, cache_key)
    
    return train_loader, val_loader, test_loader, normalization_stats


def create_kfold_data_loaders(
    geno_path: str,
    pheno_path: str,
    classification_tasks: Optional[List[str]] = None,
    n_folds: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize_regression: bool = True,
    encoding_type: str = 'token',
    random_seed: int = 42,
):
    """
    Create k-fold cross-validation data loaders.
    
    Yields data loaders for each fold with proper normalization.
    For k-fold CV: each fold uses 9 parts as training and 1 part as validation.
    No separate test set is used.
    
    Args:
        geno_path: Path to genotype file
        pheno_path: Path to phenotype file
        classification_tasks: List of classification task column names
        n_folds: Number of folds for cross-validation
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize_regression: Apply z-score normalization to regression targets
        encoding_type: Encoding strategy ('token' or 'diploid_onehot')
        random_seed: Random seed for fold splitting
        
    Yields:
        (fold_idx, train_loader, val_loader, normalization_stats, seq_length)
    """
    from sklearn.model_selection import KFold
    
    # Load data once using specified encoding
    from aquila.encoding import parse_genotype_file as parse_geno_with_encoding
    snp_matrix, sample_ids, snp_ids = parse_geno_with_encoding(geno_path, encoding_type)
    
    pheno_df, regression_cols, classification_cols = parse_phenotype_file(
        pheno_path, classification_tasks
    )
    
    # Get sequence length for model initialization
    if snp_matrix.ndim == 2:
        seq_length = snp_matrix.shape[1]
    elif snp_matrix.ndim == 3:
        seq_length = snp_matrix.shape[1]
    else:
        raise ValueError(f"Unexpected snp_matrix shape: {snp_matrix.shape}")
    
    # First create dataset WITHOUT normalization to get valid indices
    dataset_unnormalized = VariantsDataset(
        snp_matrix, pheno_df, sample_ids,
        regression_cols, classification_cols,
        normalize_regression=False
    )
    
    # Get all valid sample indices
    n_samples = len(dataset_unnormalized)
    indices = np.arange(n_samples)
    
    # Create k-fold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    print(f"\nPreparing {n_folds}-fold cross-validation")
    print(f"Total samples: {n_samples}")
    print(f"Each fold: {n_folds-1} parts for training, 1 part for validation")
    
    # Generate folds
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"\n{'='*80}")
        print(f"Preparing Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*80}")
        
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val samples: {len(val_indices)}")
        
        # Compute normalization statistics from training set only
        normalization_stats = None
        regression_means_np = None
        regression_stds_np = None
        
        if normalize_regression and regression_cols:
            # Get training sample IDs
            train_sample_ids = [dataset_unnormalized.sample_ids[i] for i in train_indices]
            
            # Compute statistics from original phenotype data for training samples only
            regression_means_np = np.zeros(len(regression_cols))
            regression_stds_np = np.ones(len(regression_cols))
            
            for i, col in enumerate(regression_cols):
                # Get original values for training samples
                train_pheno_subset = pheno_df[pheno_df['sample_id'].isin(train_sample_ids)]
                valid_values = train_pheno_subset[col].dropna().values
                
                if len(valid_values) > 0:
                    regression_means_np[i] = np.mean(valid_values)
                    regression_stds_np[i] = np.std(valid_values)
                    if regression_stds_np[i] < 1e-6:  # Avoid division by zero
                        regression_stds_np[i] = 1.0
            
            normalization_stats = {
                'regression_means': regression_means_np,
                'regression_stds': regression_stds_np,
                'regression_tasks': regression_cols
            }
            
            print(f"  Computed normalization stats from training set")
        
        # Create normalized dataset using the training statistics
        fold_dataset = VariantsDataset(
            snp_matrix, pheno_df, sample_ids,
            regression_cols, classification_cols,
            normalize_regression=normalize_regression,
            regression_means=regression_means_np,
            regression_stds=regression_stds_np
        )
        
        # Create data loaders for this fold
        train_loader = DataLoader(
            torch.utils.data.Subset(fold_dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            torch.utils.data.Subset(fold_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        yield fold_idx, train_loader, val_loader, normalization_stats, seq_length


def normalize_phenotypes(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute normalization statistics from training data.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        
    Returns:
        means: Dictionary of means per task
        stds: Dictionary of stds per task
    """
    # Collect all regression targets from training set
    all_targets = []
    all_masks = []
    
    for batch in train_loader:
        if 'regression_targets' in batch:
            all_targets.append(batch['regression_targets'].numpy())
            all_masks.append(batch['regression_mask'].numpy())
    
    if not all_targets:
        return {}, {}
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute statistics per task
    n_tasks = all_targets.shape[1]
    means = {}
    stds = {}
    
    for i in range(n_tasks):
        valid_values = all_targets[all_masks[:, i], i]
        if len(valid_values) > 0:
            means[f'task_{i}'] = float(np.mean(valid_values))
            stds[f'task_{i}'] = float(np.std(valid_values))
        else:
            means[f'task_{i}'] = 0.0
            stds[f'task_{i}'] = 1.0
    
    return means, stds


"""
Genotype encoding strategies for genomic data.

This module provides different encoding schemes for SNP genotype data:
- Additive encoding: Maps genotypes to discrete tokens {0,1,2,3} for embedding
- Diploid one-hot encoding: Maps genotypes to 8-dimensional one-hot vectors
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List


###############################################################################
# Token Encoding (Default - for use with snp_embedding)
###############################################################################

def parse_genotype_token(geno_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse genotype file to discrete tokens for embedding lookup.
    
    File format:
    #CHROM POS REF ALT Sample001 Sample002 ...
    Values: A, C, G, T, H (heterozygous), N/. (missing)
    
    Encoding:
    - 0: Homozygous alternate (ALT/ALT)
    - 1: Homozygous reference (REF/REF)
    - 2: Heterozygous (H)
    - 3: Missing (N, .)
    
    This encoding is designed to work with snp_embedding in params.yaml.
    
    Args:
        geno_path: Path to genotype file
        
    Returns:
        snp_matrix: (n_samples, n_snps) array with values {0, 1, 2, 3}
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs (CHROM_POS)
    """
    print(f"Loading genotype file: {geno_path}")
    print(f"Using encoding: token (for snp_embedding)")
    
    # Read the file
    df = pd.read_csv(geno_path, sep=r'\s+', dtype=str)
    
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
    
    # Convert genotypes to token encoding
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
                raise ValueError(
                    f"Invalid genotype: {genotype} (REF={ref}, ALT={alt}) "
                    f"at SNP {snp_idx}, sample {sample_idx}"
                )
    
    return snp_matrix, sample_ids, snp_ids


###############################################################################
# Diploid One-Hot Encoding
###############################################################################

def parse_genotype_diploid_onehot(geno_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse genotype file using diploid one-hot encoding.
    
    File format:
    #CHROM POS REF ALT Sample001 Sample002 ...
    Values: A, C, G, T, H (heterozygous), N/. (missing)
    
    Encoding scheme (8-dimensional):
    Each SNP is represented as two alleles, each encoded as 4-dim one-hot:
    - Position 0: A
    - Position 1: T
    - Position 2: C
    - Position 3: G
    
    Examples:
    - A (AA) → [1,0,0,0, 1,0,0,0]
    - T (TT) → [0,1,0,0, 0,1,0,0]
    - C (CC) → [0,0,1,0, 0,0,1,0]
    - G (GG) → [0,0,0,1, 0,0,0,1]
    - H (REF=A, ALT=T) → [1,0,0,0, 0,1,0,0]  (order preserved: AT ≠ TA)
    - Missing (N, .) → [0,0,0,0, 0,0,0,0]
    
    Args:
        geno_path: Path to genotype file
        
    Returns:
        snp_matrix: (n_samples, n_snps, 8) array with one-hot encoded values
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs (CHROM_POS)
    """
    print(f"Loading genotype file: {geno_path}")
    print(f"Using encoding: diploid_onehot (8-dimensional)")
    
    # Read the file
    df = pd.read_csv(geno_path, sep=r'\s+', dtype=str)
    
    # Extract metadata columns
    metadata_cols = ['#CHROM', 'POS', 'REF', 'ALT']
    sample_ids = [col for col in df.columns if col not in metadata_cols]
    
    n_snps = len(df)
    n_samples = len(sample_ids)
    
    print(f"Found {n_snps} SNPs and {n_samples} samples")
    
    # Create SNP IDs
    snp_ids = [f"{row['#CHROM']}_{row['POS']}" for _, row in df.iterrows()]
    
    # Initialize SNP matrix with 8 dimensions per SNP
    snp_matrix = np.zeros((n_samples, n_snps, 8), dtype=np.float32)
    
    # Mapping from nucleotide to one-hot position
    nucleotide_to_onehot = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
    }
    
    # Convert genotypes to diploid one-hot encoding
    for snp_idx, (_, row) in enumerate(df.iterrows()):
        ref = row['REF']
        alt = row['ALT']
        
        for sample_idx, sample_id in enumerate(sample_ids):
            genotype = row[sample_id]
            
            if pd.isna(genotype) or genotype == '.' or genotype == 'N':
                # Missing: all zeros [0,0,0,0, 0,0,0,0]
                pass  # Already initialized to zeros
            
            elif genotype == 'H':
                # Heterozygous: REF/ALT
                if ref not in nucleotide_to_onehot or alt not in nucleotide_to_onehot:
                    raise ValueError(
                        f"Invalid REF/ALT alleles: REF={ref}, ALT={alt} "
                        f"at SNP {snp_idx}"
                    )
                # First allele: REF, Second allele: ALT
                allele1_onehot = nucleotide_to_onehot[ref]
                allele2_onehot = nucleotide_to_onehot[alt]
                snp_matrix[sample_idx, snp_idx, :4] = allele1_onehot
                snp_matrix[sample_idx, snp_idx, 4:] = allele2_onehot
            
            elif genotype in nucleotide_to_onehot:
                # Homozygous: genotype/genotype (e.g., A → AA)
                allele_onehot = nucleotide_to_onehot[genotype]
                snp_matrix[sample_idx, snp_idx, :4] = allele_onehot
                snp_matrix[sample_idx, snp_idx, 4:] = allele_onehot
            
            else:
                raise ValueError(
                    f"Invalid genotype: {genotype} (REF={ref}, ALT={alt}) "
                    f"at SNP {snp_idx}, sample {sample_idx}"
                )
    
    return snp_matrix, sample_ids, snp_ids



###############################################################################
# Factory Function
###############################################################################

def parse_genotype_file(geno_path: str, encoding_type: str = 'token') -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse genotype file using specified encoding strategy.
    
    Args:
        geno_path: Path to genotype file
        encoding_type: Type of encoding ('token' or 'diploid_onehot')
            - 'token': Returns (n_samples, n_snps) with values {0,1,2,3} for use with snp_embedding
            - 'diploid_onehot': Returns (n_samples, n_snps, 8) with one-hot encoded diploid genotypes
        
    Returns:
        snp_matrix: Encoded SNP matrix (shape depends on encoding type)
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs
        
    Raises:
        ValueError: If encoding_type is not recognized
    """
    if encoding_type == 'token':
        return parse_genotype_token(geno_path)
    elif encoding_type == 'diploid_onehot':
        return parse_genotype_diploid_onehot(geno_path)
    else:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Available options: ['token', 'diploid_onehot']"
        )


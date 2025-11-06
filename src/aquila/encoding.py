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
    
    # Mapping from nucleotide to one-hot position (alphabetical order: A C G T)
    nucleotide_to_onehot = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
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

###############################################################################
# VCF Encoding (for multi-variant-type analysis)
###############################################################################

def parse_genotype_vcf(
    vcf_path: str,
    variant_types: List[str] = None
) -> dict:
    """
    Parse VCF file with variant type filtering for multi-branch architectures.
    
    File format:
    Standard VCF with phased genotypes (GT field: 0|0, 0|1, 1|0, 1|1, .|.)
    ID column contains variant type prefix (SNP-, pdSNP-, INDEL-, SV-)
    
    Args:
        vcf_path: Path to VCF file
        variant_types: List of variant types to extract (e.g., ["SNP", "INDEL", "SV"])
                      If None, extract all variants
                      
    Returns:
        Dictionary mapping variant_type to (matrix, sample_ids, variant_ids):
        {
            'SNP': (snp_matrix, sample_ids, snp_ids),
            'INDEL': (indel_matrix, sample_ids, indel_ids),
            'SV': (sv_matrix, sample_ids, sv_ids)
        }
        Each matrix has shape (n_samples, n_variants, 8) with diploid one-hot encoding
        
    Encoding scheme:
    - 0|0 (REF/REF): [ref_onehot, ref_onehot]
    - 0|1 (REF/ALT): [ref_onehot, alt_onehot]
    - 1|0 (ALT/REF): [alt_onehot, ref_onehot]
    - 1|1 (ALT/ALT): [alt_onehot, alt_onehot]
    - .|. or ./. (missing): [0,0,0,0, 0,0,0,0]
    """
    print(f"Loading VCF file: {vcf_path}")
    if variant_types:
        print(f"Filtering for variant types: {variant_types}")
    
    # Nucleotide to one-hot mapping (alphabetical order: A C G T)
    nucleotide_to_onehot = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
    }
    
    # Read VCF file
    variants_by_type = {vtype: [] for vtype in (variant_types or [])}
    sample_ids = None
    
    with open(vcf_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse header to get sample IDs
            if line.startswith('#CHROM'):
                fields = line.split('\t')
                # Standard VCF columns: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, then samples
                sample_ids = fields[9:]  # All columns after FORMAT
                print(f"Found {len(sample_ids)} samples in VCF")
                continue
            
            # Skip other header lines
            if line.startswith('#'):
                continue
            
            # Parse variant line
            fields = line.split('\t')
            chrom = fields[0]
            pos = fields[1]
            variant_id = fields[2]
            ref = fields[3]
            alt = fields[4]
            format_field = fields[8]
            genotypes = fields[9:]
            
            # Determine variant type from ID
            variant_type = None
            if variant_types:
                for vtype in variant_types:
                    if vtype in variant_id:
                        variant_type = vtype
                        break
                # Skip if not in requested types
                if variant_type is None:
                    continue
            else:
                # If no filtering, determine type from ID
                if 'SNP' in variant_id:
                    variant_type = 'SNP'
                elif 'INDEL' in variant_id:
                    variant_type = 'INDEL'
                elif 'SV' in variant_id:
                    variant_type = 'SV'
                else:
                    variant_type = 'OTHER'
                
                if variant_type not in variants_by_type:
                    variants_by_type[variant_type] = []
            
            # Parse GT field (assume GT is first in FORMAT)
            gt_idx = format_field.split(':').index('GT') if 'GT' in format_field else 0
            
            # Encode genotypes for this variant
            variant_encodings = []
            for gt_field in genotypes:
                gt = gt_field.split(':')[gt_idx]
                
                # Parse phased/unphased genotypes
                if '|' in gt:
                    alleles = gt.split('|')
                elif '/' in gt:
                    alleles = gt.split('/')
                else:
                    alleles = ['.', '.']
                
                # Encode diploid genotype
                encoding = np.zeros(8, dtype=np.float32)
                
                # Check for missing
                if alleles[0] == '.' or alleles[1] == '.':
                    # Missing: all zeros
                    pass
                else:
                    # Parse allele indices
                    try:
                        allele1_idx = int(alleles[0])
                        allele2_idx = int(alleles[1])
                        
                        # Get nucleotides (0=REF, 1=ALT)
                        allele1_nuc = ref if allele1_idx == 0 else alt
                        allele2_nuc = ref if allele2_idx == 0 else alt
                        
                        # Encode if valid nucleotides
                        if allele1_nuc in nucleotide_to_onehot and allele2_nuc in nucleotide_to_onehot:
                            encoding[:4] = nucleotide_to_onehot[allele1_nuc]
                            encoding[4:] = nucleotide_to_onehot[allele2_nuc]
                    except (ValueError, IndexError):
                        # Invalid genotype, leave as zeros
                        pass
                
                variant_encodings.append(encoding)
            
            # Store variant data
            variants_by_type[variant_type].append({
                'id': variant_id,
                'encodings': np.array(variant_encodings)  # Shape: (n_samples, 8)
            })
    
    if sample_ids is None:
        raise ValueError("No sample IDs found in VCF file. Make sure file has #CHROM header line.")
    
    # Convert to matrices
    result = {}
    for vtype, variants in variants_by_type.items():
        if len(variants) == 0:
            print(f"Warning: No {vtype} variants found")
            continue
        
        n_variants = len(variants)
        n_samples = len(sample_ids)
        
        # Create matrix: (n_samples, n_variants, 8)
        matrix = np.zeros((n_samples, n_variants, 8), dtype=np.float32)
        variant_ids = []
        
        for i, variant in enumerate(variants):
            matrix[:, i, :] = variant['encodings']
            variant_ids.append(variant['id'])
        
        result[vtype] = (matrix, sample_ids, variant_ids)
        print(f"  {vtype}: {n_variants} variants × {n_samples} samples")
    
    return result


def parse_genotype_snp_vcf(vcf_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse VCF file extracting only SNP variants.
    
    Returns:
        snp_matrix: (n_samples, n_snps, 8) array
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs
    """
    result = parse_genotype_vcf(vcf_path, variant_types=['SNP'])
    if 'SNP' not in result:
        raise ValueError("No SNP variants found in VCF file")
    return result['SNP']


def parse_genotype_snp_indel_vcf(vcf_path: str) -> dict:
    """
    Parse VCF file extracting SNP and INDEL variants.
    
    Returns:
        Dictionary with 'SNP' and 'INDEL' keys, each mapping to (matrix, sample_ids, variant_ids)
    """
    return parse_genotype_vcf(vcf_path, variant_types=['SNP', 'INDEL'])


def parse_genotype_snp_indel_sv_vcf(vcf_path: str) -> dict:
    """
    Parse VCF file extracting SNP, INDEL, and SV variants.
    
    Returns:
        Dictionary with 'SNP', 'INDEL', and 'SV' keys, each mapping to (matrix, sample_ids, variant_ids)
    """
    return parse_genotype_vcf(vcf_path, variant_types=['SNP', 'INDEL', 'SV'])


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
            - 'snp_vcf': Returns (n_samples, n_snps, 8) from VCF with only SNPs
            - 'snp_indel_vcf': Returns dict with SNP and INDEL matrices
            - 'snp_indel_sv_vcf': Returns dict with SNP, INDEL, and SV matrices
        
    Returns:
        snp_matrix: Encoded SNP matrix (shape depends on encoding type)
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs
        
        For multi-branch VCF types, returns dict instead of tuple
        
    Raises:
        ValueError: If encoding_type is not recognized
    """
    if encoding_type == 'token':
        return parse_genotype_token(geno_path)
    elif encoding_type == 'diploid_onehot':
        return parse_genotype_diploid_onehot(geno_path)
    elif encoding_type == 'snp_vcf':
        return parse_genotype_snp_vcf(geno_path)
    elif encoding_type == 'snp_indel_vcf':
        return parse_genotype_snp_indel_vcf(geno_path)
    elif encoding_type == 'snp_indel_sv_vcf':
        return parse_genotype_snp_indel_sv_vcf(geno_path)
    else:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Available options: ['token', 'diploid_onehot', 'snp_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf']"
        )


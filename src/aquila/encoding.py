"""
Genotype encoding strategies for genomic data.

This module provides different encoding schemes for SNP genotype data:
- Additive encoding: Maps genotypes to discrete tokens {0,1,2,3} for embedding
- Diploid one-hot encoding: Maps genotypes to 8-dimensional one-hot vectors
- Classic one-hot (onehot): Biallelic SNP as 3-class {REF/REF, het, ALT/ALT}
- 10-classed one-hot (10classed_onehot): unordered diploid
  {AA, AT, AC, AG, TT, TC, TG, CC, CG, GG}
- Genotype-class one-hot (4-way): INDEL/SV as 4-class {REF/REF, REF/ALT, ALT/REF, ALT/ALT}
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import gzip
import subprocess
import tempfile
from typing import Tuple, List, Sequence


def parse_id_prefix_spec(spec: str | None) -> tuple[str, ...] | None:
    """Parse ``--id-prefix`` into startswith tokens.

    ``"SNP-"`` keeps IDs that start with ``SNP-``.
    ``"SNP- | INDEL-"`` keeps either prefix. ``None`` / empty disables filtering.
    """
    if spec is None:
        return None
    prefixes: list[str] = []
    for part in str(spec).split("|"):
        token = part.strip()
        if token.startswith("^"):
            token = token[1:].strip()
        if token:
            prefixes.append(token)
    if not prefixes:
        raise ValueError(f"id_prefix {spec!r} did not contain any prefixes")
    return tuple(prefixes)


def variant_id_matches_prefixes(
    variant_id: str | None, prefixes: Sequence[str] | None
) -> bool:
    """Return True when the VCF ID starts with any configured prefix."""
    if not prefixes:
        return True
    vid = "" if variant_id is None else str(variant_id)
    return any(vid.startswith(prefix) for prefix in prefixes)


_ACGT = frozenset("ACGT")


def _is_star_allele(ref: str | None, alt: str | None) -> bool:
    """True when REF or ALT is VCF ``*`` (spanning deletion); encode as all-zero."""
    if ref is None or alt is None:
        return False
    ref_u = str(ref).upper()
    alt_u = str(alt).upper()
    if "," in alt_u:
        return False
    return ref_u == "*" or alt_u == "*"


def _is_allowed_snp_alleles(ref: str | None, alt: str | None) -> bool:
    """SNP mode: biallelic ACGT, or ``*`` which is stored as all-zero."""
    return _is_biallelic_acgt_snp(ref, alt) or _is_star_allele(ref, alt)


def _is_biallelic_acgt_snp(ref: str | None, alt: str | None) -> bool:
    """True when REF/ALT are a single-nucleotide biallelic ACGT pair."""
    if ref is None or alt is None:
        return False
    ref_u = str(ref).upper()
    alt_u = str(alt).upper()
    if "," in alt_u:
        return False
    return len(ref_u) == 1 and len(alt_u) == 1 and ref_u in _ACGT and alt_u in _ACGT


def _all_zero_site_indices(matrix: np.ndarray) -> np.ndarray:
    """Return site indices whose encoding is all-zero for every sample."""
    arr = np.asarray(matrix)
    if arr.ndim < 2:
        return np.zeros((0,), dtype=int)
    per_sample = arr.reshape(arr.shape[0], arr.shape[1], -1).sum(axis=-1)
    return np.flatnonzero(per_sample.sum(axis=0) == 0)


def ensure_snp_sites_are_acgt(parsed: dict) -> dict:
    """Reject SNP-mode sites that are neither ACGT SNPs nor VCF ``*``.

    ``*`` (spanning deletion) is allowed and stored as all-zero. Other non-ACGT
    alleles still fail. All-zero columns on ACGT SNPs still fail.
    """
    if not isinstance(parsed, dict) or "variant_ids" not in parsed:
        return parsed
    variant_ids = list(parsed["variant_ids"])
    refs = list(parsed.get("refs") or [])
    alts = list(parsed.get("alts") or [])
    bad: list[str] = []
    seen: set[str] = set()

    n = len(variant_ids)
    for i in range(n):
        ref = refs[i] if i < len(refs) else None
        alt = alts[i] if i < len(alts) else None
        if not _is_allowed_snp_alleles(ref, alt):
            vid = variant_ids[i]
            if vid not in seen:
                seen.add(vid)
                bad.append(vid)

    if "matrix" in parsed:
        for i in _all_zero_site_indices(parsed["matrix"]):
            idx = int(i)
            ref = refs[idx] if idx < len(refs) else None
            alt = alts[idx] if idx < len(alts) else None
            if _is_star_allele(ref, alt):
                continue
            vid = variant_ids[idx] if idx < len(variant_ids) else f"site_{idx}"
            if vid not in seen:
                seen.add(vid)
                bad.append(vid)

    if not bad:
        return parsed
    examples = ", ".join(str(v) for v in bad[:8])
    more = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
    raise ValueError(
        f"--variant-type snp (SNP mode) found {len(bad)} non-ACGT site(s) "
        f"(REF/ALT are not single-base A/C/G/T, or the site encodes as all-zero): "
        f"{examples}{more}."
    )


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

    return {
        'matrix': snp_matrix,
        'sample_ids': sample_ids,
        'variant_ids': snp_ids,
        'refs': [row['REF'] for _, row in df.iterrows()],
        'alts': [row['ALT'] for _, row in df.iterrows()],
        'chroms': [row['#CHROM'] for _, row in df.iterrows()],
        'positions': [row['POS'] for _, row in df.iterrows()],
    }


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

    return {
        'matrix': snp_matrix,
        'sample_ids': sample_ids,
        'variant_ids': snp_ids,
        'refs': [row['REF'] for _, row in df.iterrows()],
        'alts': [row['ALT'] for _, row in df.iterrows()],
        'chroms': [row['#CHROM'] for _, row in df.iterrows()],
        'positions': [row['POS'] for _, row in df.iterrows()],
    }


###############################################################################
# Factory Function
###############################################################################

###############################################################################
# VCF Encoding (for multi-variant-type analysis)
###############################################################################

def parse_genotype_vcf(
    vcf_path: str,
    variant_types: List[str] = None,
    assume_variant_type: str = None,
    id_prefixes: Sequence[str] | None = None,
) -> dict:
    """
    Parse VCF file with variant type filtering for multi-branch architectures.

    File format:
    Standard VCF with phased genotypes (GT field: 0|0, 0|1, 1|0, 1|1, .|.)
    ID column contains variant type prefix (SNP-, pdSNP-, INDEL-, SV-)

    Args:
        vcf_path: Path to VCF file
        variant_types: Variant types to retain after automatic classification.
            If None, retain every automatically classified type.
        assume_variant_type: Assign every VCF record to this type without
            inspecting the ID or alleles. Intended for explicitly typed inputs.
        id_prefixes: If set, keep only records whose VCF ID starts with one of
            these strings (see ``parse_id_prefix_spec``).

    Returns:
        Dictionary mapping variant_type to (matrix, sample_ids, variant_ids):
        {
            'SNP': (snp_matrix, sample_ids, snp_ids),
            'INDEL': (indel_matrix, sample_ids, indel_ids),
            'SV': (sv_matrix, sample_ids, sv_ids)
        }

        SNP encoding (8-dimensional diploid nucleotide one-hot):
        - 0|0 (REF/REF): [A/C/G/T one-hot for REF, A/C/G/T one-hot for REF]
        - 0|1 (REF/ALT): [A/C/G/T one-hot for REF, A/C/G/T one-hot for ALT]
        - 1|0 (ALT/REF): [A/C/G/T one-hot for ALT, A/C/G/T one-hot for REF]
        - 1|1 (ALT/ALT): [A/C/G/T one-hot for ALT, A/C/G/T one-hot for ALT]
        - .|. or ./. (missing): [0,0,0,0, 0,0,0,0]

        INDEL/SV encoding (4-dimensional genotype-class one-hot):
        - 0|0 or 0/0 (REF/REF): [1, 0, 0, 0]
        - 0|1 or 0/1 (REF/ALT): [0, 1, 0, 0]  (phase preserved)
        - 1|0 or 1/0 (ALT/REF): [0, 0, 1, 0]  (phase preserved, different from 0|1)
        - 1|1 or 1/1 (ALT/ALT): [0, 0, 0, 1]
        - .|. or ./. (missing): [0, 0, 0, 0]
    """
    print(f"Loading VCF file: {vcf_path}")
    if assume_variant_type is not None:
        assume_variant_type = str(assume_variant_type).upper()
        if assume_variant_type not in {"SNP", "INDEL", "SV"}:
            raise ValueError(
                "assume_variant_type must be SNP, INDEL, or SV"
            )
        print(f"Treating all VCF records as {assume_variant_type}")
    elif variant_types:
        print(f"Filtering for variant types: {variant_types}")
    if id_prefixes:
        print(f"Keeping variant IDs starting with: {list(id_prefixes)}")

    # Nucleotide to one-hot mapping (alphabetical order: A C G T)
    nucleotide_to_onehot = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
    }

    # Genotype-class one-hot mapping for INDEL/SV (4-way)
    # Index: 0=REF/REF, 1=REF/ALT, 2=ALT/REF, 3=ALT/ALT
    genotype_class_to_onehot = {
        (0, 0): [1, 0, 0, 0],  # 0|0 or 0/0: Homozygous REF
        (0, 1): [0, 1, 0, 0],  # 0|1 or 0/1: Heterozygous (REF first)
        (1, 0): [0, 0, 1, 0],  # 1|0 or 1/0: Heterozygous (ALT first) - phase preserved
        (1, 1): [0, 0, 0, 1],  # 1|1 or 1/1: Homozygous ALT
    }

    # Read VCF file (support both .vcf and .vcf.gz)
    initial_types = (
        [assume_variant_type]
        if assume_variant_type is not None
        else (variant_types or [])
    )
    variants_by_type = {vtype: [] for vtype in initial_types}
    sample_ids = None

    # Detect if file is gzipped
    is_gzipped = vcf_path.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    open_mode = 'rt' if is_gzipped else 'r'

    with open_func(vcf_path, open_mode) as f:
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

            if not variant_id_matches_prefixes(variant_id, id_prefixes):
                continue

            # Explicitly typed inputs do not require an ID naming convention.
            variant_type = assume_variant_type
            if variant_type is not None:
                pass
            elif variant_types:
                for vtype in variant_types:
                    if vtype in variant_id:
                        variant_type = vtype
                        break
                # Skip if not in requested types
                if variant_type is None:
                    continue
            else:
                # Prefer the established ID convention, then infer from alleles.
                if 'SNP' in variant_id:
                    variant_type = 'SNP'
                elif 'INDEL' in variant_id:
                    variant_type = 'INDEL'
                elif 'SV' in variant_id:
                    variant_type = 'SV'
                elif (
                    alt.startswith("<")
                    or "[" in alt
                    or "]" in alt
                    or alt == "*"
                ):
                    variant_type = 'SV'
                elif (
                    len(ref) == 1
                    and len(alt) == 1
                    and "," not in alt
                ):
                    variant_type = 'SNP'
                else:
                    variant_type = 'INDEL'

                if variant_type not in variants_by_type:
                    variants_by_type[variant_type] = []

            is_snp = variant_type == 'SNP'
            encoding_dim = 8 if is_snp else 4

            variant_encodings = []
            if is_snp and _is_star_allele(ref, alt):
                variant_encodings = [
                    np.zeros(encoding_dim, dtype=np.float32)
                    for _ in genotypes
                ]
            else:
                gt_idx = format_field.split(':').index(
                    'GT') if 'GT' in format_field else 0
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
                    encoding = np.zeros(encoding_dim, dtype=np.float32)

                    # Check for missing
                    if alleles[0] == '.' or alleles[1] == '.':
                        # Missing: all zeros
                        pass
                    else:
                        # Parse allele indices
                        try:
                            allele1_idx = int(alleles[0])
                            allele2_idx = int(alleles[1])

                            if is_snp:
                                # SNP: 8-dim diploid nucleotide one-hot encoding
                                # Get nucleotides (0=REF, 1=ALT)
                                allele1_nuc = ref if allele1_idx == 0 else alt
                                allele2_nuc = ref if allele2_idx == 0 else alt

                                # Encode if both are single nucleotides
                                if allele1_nuc in nucleotide_to_onehot and allele2_nuc in nucleotide_to_onehot:
                                    encoding[:4] = nucleotide_to_onehot[allele1_nuc]
                                    encoding[4:] = nucleotide_to_onehot[allele2_nuc]
                            else:
                                # INDEL/SV: 4-dim genotype-class one-hot encoding
                                # Preserve phase: 0|1 (REF/ALT) vs 1|0 (ALT/REF) are different
                                genotype_class = (allele1_idx, allele2_idx)
                                if genotype_class in genotype_class_to_onehot:
                                    encoding = np.array(
                                        genotype_class_to_onehot[genotype_class],
                                        dtype=np.float32
                                    )
                        except (ValueError, IndexError):
                            # Invalid genotype, leave as zeros
                            pass

                    variant_encodings.append(encoding)

            # Store variant data
            variants_by_type[variant_type].append({
                'id': variant_id,
                'chrom': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                # Shape: (n_samples, encoding_dim) - varies by variant type
                'encodings': np.array(variant_encodings)
            })

    if sample_ids is None:
        raise ValueError(
            "No sample IDs found in VCF file. Make sure file has #CHROM header line.")

    # Convert to matrices
    result = {}
    for vtype, variants in variants_by_type.items():
        if len(variants) == 0:
            print(f"Warning: No {vtype} variants found")
            # Still include empty entry so callers can distinguish "type not requested"
            # from "type requested but none found"
            result[vtype] = None
            continue

        n_variants = len(variants)
        n_samples = len(sample_ids)

        # Determine encoding dimension based on variant type
        # SNP: 8-dim diploid nucleotide one-hot; INDEL/SV: 4-dim genotype-class one-hot
        if vtype == 'SNP':
            encoding_dim = 8
        else:
            encoding_dim = 4

        # Create matrix: (n_samples, n_variants, encoding_dim)
        matrix = np.zeros((n_samples, n_variants, encoding_dim), dtype=np.float32)
        variant_ids = []
        variant_refs = []
        variant_alts = []
        variant_chroms = []
        variant_positions = []

        for i, variant in enumerate(variants):
            variant_enc = variant['encodings']
            # Handle case where encoding_dim might differ from stored (shouldn't happen but safety check)
            if variant_enc.shape[1] != encoding_dim:
                # Resize if necessary (take first encoding_dim columns)
                if variant_enc.shape[1] > encoding_dim:
                    variant_enc = variant_enc[:, :encoding_dim]
                else:
                    # Pad with zeros if smaller
                    temp = np.zeros((variant_enc.shape[0], encoding_dim), dtype=np.float32)
                    temp[:, :variant_enc.shape[1]] = variant_enc
                    variant_enc = temp
            matrix[:, i, :] = variant_enc
            variant_ids.append(variant['id'])
            variant_refs.append(variant['ref'])
            variant_alts.append(variant['alt'])
            variant_chroms.append(variant['chrom'])
            variant_positions.append(variant['pos'])

        result[vtype] = {
            'matrix': matrix,
            'sample_ids': sample_ids,
            'variant_ids': variant_ids,
            'refs': variant_refs,
            'alts': variant_alts,
            'chroms': variant_chroms,
            'positions': variant_positions,
        }
        encoding_name = 'diploid_onehot (8-dim)' if vtype == 'SNP' else 'genotype_class (4-dim)'
        print(f"  {vtype}: {n_variants} variants × {n_samples} samples [{encoding_name}]")

    return result


def parse_genotype_snp_vcf(
    vcf_path: str,
    *,
    assume_all_variants: bool = False,
    id_prefixes: Sequence[str] | None = None,
):
    """
    Parse VCF file extracting only SNP variants.

    Returns:
        Dict with keys: matrix, sample_ids, variant_ids, refs, alts, chroms, positions.
    """
    result = parse_genotype_vcf(
        vcf_path,
        variant_types=None if assume_all_variants else ['SNP'],
        assume_variant_type='SNP' if assume_all_variants else None,
        id_prefixes=id_prefixes,
    )
    if 'SNP' not in result or result['SNP'] is None:
        raise ValueError("No SNP variants found in VCF file")
    return result['SNP']


def _gt_to_classic_snp_onehot(
    gt_field: str, format_field: str, ref: str, alt: str
) -> np.ndarray:
    """
    Map VCF GT to classic 3-way one-hot for a biallelic SNP.

    - REF/REF (0/0): [1, 0, 0]
    - Heterozygous (0/1 or 1/0): [0, 1, 0]
    - ALT/ALT (1/1): [0, 0, 1]
    - Missing or non-biallelic: [0, 0, 0]

    Only sites with single-nucleotide REF and ALT (no comma in ALT) and
    allele indices in {0, 1} are encoded; otherwise returns zeros.
    """
    enc = np.zeros(3, dtype=np.float32)
    if _is_star_allele(ref, alt):
        return enc
    if ',' in alt or len(ref) != 1 or len(alt) != 1:
        return enc

    gt_idx = format_field.split(':').index('GT') if 'GT' in format_field else 0
    gt = gt_field.split(':')[gt_idx]

    if '|' in gt:
        alleles = gt.split('|')
    elif '/' in gt:
        alleles = gt.split('/')
    else:
        return enc

    if len(alleles) < 2:
        return enc
    if alleles[0] == '.' or alleles[1] == '.':
        return enc
    try:
        a1 = int(alleles[0])
        a2 = int(alleles[1])
    except ValueError:
        return enc

    if a1 not in (0, 1) or a2 not in (0, 1):
        return enc

    if a1 == 0 and a2 == 0:
        enc[0] = 1.0
    elif a1 == 1 and a2 == 1:
        enc[2] = 1.0
    else:
        enc[1] = 1.0
    return enc


def parse_genotype_snp_vcf_onehot(
    vcf_path: str,
    *,
    assume_all_variants: bool = False,
    id_prefixes: Sequence[str] | None = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse VCF file (SNP rows only) with classic 3-dimensional one-hot encoding.

    Same SNP row selection as ``parse_genotype_snp_vcf`` (variant ID contains 'SNP').

    Returns:
        snp_matrix: (n_samples, n_snps, 3) float32
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs
    """
    print(f"Loading VCF file: {vcf_path}")
    if assume_all_variants:
        print("Treating all VCF records as SNP")
    else:
        print("Filtering for variant types: ['SNP']")
    print(f"Using encoding: onehot (3-dimensional classic REF/HET/ALT)")
    if id_prefixes:
        print(f"Keeping variant IDs starting with: {list(id_prefixes)}")

    variants_by_type: dict = {'SNP': []}
    sample_ids = None

    is_gzipped = vcf_path.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    open_mode = 'rt' if is_gzipped else 'r'

    with open_func(vcf_path, open_mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#CHROM'):
                fields = line.split('\t')
                sample_ids = fields[9:]
                print(f"Found {len(sample_ids)} samples in VCF")
                continue

            if line.startswith('#'):
                continue

            fields = line.split('\t')
            chrom = fields[0]
            pos = fields[1]
            variant_id = fields[2]
            ref = fields[3]
            alt = fields[4]
            format_field = fields[8]
            genotypes = fields[9:]

            variant_type = 'SNP'
            if not assume_all_variants and 'SNP' not in variant_id:
                continue
            if not variant_id_matches_prefixes(variant_id, id_prefixes):
                continue

            variant_encodings = []
            for gt_field in genotypes:
                variant_encodings.append(
                    _gt_to_classic_snp_onehot(gt_field, format_field, ref, alt)
                )

            variants_by_type[variant_type].append({
                'id': variant_id,
                'chrom': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'encodings': np.array(variant_encodings),
            })

    if sample_ids is None:
        raise ValueError(
            "No sample IDs found in VCF file. Make sure file has #CHROM header line."
        )

    variants = variants_by_type['SNP']
    if len(variants) == 0:
        raise ValueError("No SNP variants found in VCF file")

    n_variants = len(variants)
    n_samples = len(sample_ids)
    matrix = np.zeros((n_samples, n_variants, 3), dtype=np.float32)
    snp_ids = []
    refs = []
    alts = []
    chroms = []
    positions = []

    for i, variant in enumerate(variants):
        matrix[:, i, :] = variant['encodings']
        snp_ids.append(variant['id'])
        refs.append(variant['ref'])
        alts.append(variant['alt'])
        chroms.append(variant['chrom'])
        positions.append(variant['pos'])

    print(f"  SNP: {n_variants} variants × {n_samples} samples")
    return {
        'matrix': matrix,
        'sample_ids': sample_ids,
        'variant_ids': snp_ids,
        'refs': refs,
        'alts': alts,
        'chroms': chroms,
        'positions': positions,
    }


# Unordered diploid genotypes over {A,C,G,T}. Heterozygotes ignore phase:
# AT == TA, TC == CT, TG == GT, CG == GC.
TEN_CLASSED_GENOTYPE_ORDER = (
    "AA", "AT", "AC", "AG", "TT", "TC", "TG", "CC", "CG", "GG",
)
TEN_CLASSED_INDEX = {}
for _idx, _gt in enumerate(TEN_CLASSED_GENOTYPE_ORDER):
    TEN_CLASSED_INDEX[_gt] = _idx
    TEN_CLASSED_INDEX[_gt[1] + _gt[0]] = _idx


def _nucleotides_to_10class_onehot(allele1: str, allele2: str) -> np.ndarray:
    """Map two nucleotides to a 10-class unordered diploid one-hot."""
    enc = np.zeros(10, dtype=np.float32)
    key = f"{allele1}{allele2}".upper()
    idx = TEN_CLASSED_INDEX.get(key)
    if idx is None:
        return enc
    enc[idx] = 1.0
    return enc


def _gt_to_10class_onehot(
    gt_field: str, format_field: str, ref: str, alt: str
) -> np.ndarray:
    """Map VCF GT to 10-class unordered diploid one-hot; missing stays zeros."""
    enc = np.zeros(10, dtype=np.float32)
    if _is_star_allele(ref, alt):
        return enc
    if "," in alt:
        return enc

    gt_idx = format_field.split(":").index("GT") if "GT" in format_field else 0
    gt = gt_field.split(":")[gt_idx]
    if "|" in gt:
        alleles = gt.split("|")
    elif "/" in gt:
        alleles = gt.split("/")
    else:
        return enc
    if len(alleles) < 2 or alleles[0] == "." or alleles[1] == ".":
        return enc
    try:
        a1 = int(alleles[0])
        a2 = int(alleles[1])
    except ValueError:
        return enc

    bases = [ref, alt]
    if a1 not in (0, 1) or a2 not in (0, 1):
        return enc
    n1 = bases[a1]
    n2 = bases[a2]
    if len(n1) != 1 or len(n2) != 1:
        return enc
    return _nucleotides_to_10class_onehot(n1, n2)


def parse_genotype_snp_vcf_10classed_onehot(
    vcf_path: str,
    *,
    assume_all_variants: bool = False,
    id_prefixes: Sequence[str] | None = None,
) -> dict:
    """Parse VCF SNPs as 10-class unordered diploid nucleotide one-hot.

    Classes: AA, AT, AC, AG, TT, TC, TG, CC, CG, GG (phase-insensitive).
    Missing or non-ACGT diploid calls are all-zero.

    Returns:
        Dict with matrix of shape (n_samples, n_snps, 10).
    """
    print(f"Loading VCF file: {vcf_path}")
    if assume_all_variants:
        print("Treating all VCF records as SNP")
    else:
        print("Filtering for variant types: ['SNP']")
    print(
        "Using encoding: 10classed_onehot "
        "(AA, AT, AC, AG, TT, TC, TG, CC, CG, GG)"
    )
    if id_prefixes:
        print(f"Keeping variant IDs starting with: {list(id_prefixes)}")

    variants_by_type: dict = {"SNP": []}
    sample_ids = None
    is_gzipped = vcf_path.endswith(".gz")
    open_func = gzip.open if is_gzipped else open
    open_mode = "rt" if is_gzipped else "r"

    with open_func(vcf_path, open_mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#CHROM"):
                fields = line.split("\t")
                sample_ids = fields[9:]
                print(f"Found {len(sample_ids)} samples in VCF")
                continue
            if line.startswith("#"):
                continue

            fields = line.split("\t")
            chrom = fields[0]
            pos = fields[1]
            variant_id = fields[2]
            ref = fields[3]
            alt = fields[4]
            format_field = fields[8]
            genotypes = fields[9:]
            if not assume_all_variants and "SNP" not in variant_id:
                continue
            if not variant_id_matches_prefixes(variant_id, id_prefixes):
                continue

            variant_encodings = [
                _gt_to_10class_onehot(gt_field, format_field, ref, alt)
                for gt_field in genotypes
            ]
            variants_by_type["SNP"].append(
                {
                    "id": variant_id,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "encodings": np.array(variant_encodings),
                }
            )

    if sample_ids is None:
        raise ValueError(
            "No sample IDs found in VCF file. Make sure file has #CHROM header line."
        )
    variants = variants_by_type["SNP"]
    if len(variants) == 0:
        raise ValueError("No SNP variants found in VCF file")

    n_variants = len(variants)
    n_samples = len(sample_ids)
    matrix = np.zeros((n_samples, n_variants, 10), dtype=np.float32)
    snp_ids = []
    refs = []
    alts = []
    chroms = []
    positions = []
    for i, variant in enumerate(variants):
        matrix[:, i, :] = variant["encodings"]
        snp_ids.append(variant["id"])
        refs.append(variant["ref"])
        alts.append(variant["alt"])
        chroms.append(variant["chrom"])
        positions.append(variant["pos"])

    print(f"  SNP: {n_variants} variants × {n_samples} samples")
    return {
        "matrix": matrix,
        "sample_ids": sample_ids,
        "variant_ids": snp_ids,
        "refs": refs,
        "alts": alts,
        "chroms": chroms,
        "positions": positions,
    }


def parse_genotype_indel_vcf(
    vcf_path: str,
    *,
    assume_all_variants: bool = False,
    id_prefixes: Sequence[str] | None = None,
):
    """
    Parse VCF file extracting only INDEL variants.

    Returns:
        Dict with keys: matrix, sample_ids, variant_ids, refs, alts, chroms, positions.
    """
    result = parse_genotype_vcf(
        vcf_path,
        variant_types=None if assume_all_variants else ['INDEL'],
        assume_variant_type='INDEL' if assume_all_variants else None,
        id_prefixes=id_prefixes,
    )
    if 'INDEL' not in result or result['INDEL'] is None:
        raise ValueError("No INDEL variants found in VCF file")
    return result['INDEL']


def parse_genotype_sv_vcf(
    vcf_path: str,
    *,
    assume_all_variants: bool = False,
    id_prefixes: Sequence[str] | None = None,
):
    """
    Parse VCF file extracting only SV (structural variant) variants.

    Returns:
        Dict with keys: matrix, sample_ids, variant_ids, refs, alts, chroms, positions.
    """
    result = parse_genotype_vcf(
        vcf_path,
        variant_types=None if assume_all_variants else ['SV'],
        assume_variant_type='SV' if assume_all_variants else None,
        id_prefixes=id_prefixes,
    )
    if 'SV' not in result or result['SV'] is None:
        raise ValueError("No SV variants found in VCF file")
    return result['SV']


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

def parse_genotype_file(
    geno_path: str,
    encoding_type: str = 'diploid_onehot',
    variant_type: str = None,
    id_prefix: str | None = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse genotype file using specified encoding strategy.

    Args:
        geno_path: Path to genotype file
        encoding_type: Type of encoding
            - 'token': Returns (n_samples, n_snps) with values {0,1,2,3} for use with snp_embedding
            - 'diploid_onehot': Returns (n_samples, n_snps, 8) with one-hot encoded diploid genotypes
            - 'onehot': Returns (n_samples, n_snps, 3) classic REF/HET/ALT one-hot from VCF (biallelic SNPs)
            - '10classed_onehot': Returns (n_samples, n_snps, 10) unordered diploid nucleotide classes
        variant_type: How to encode VCF records (does not filter by ID)
            - 'snp': Treat every kept record as SNP
            - 'indel': Treat every kept record as INDEL
            - 'sv': Treat every kept record as SV
            If None, variant types are detected automatically.
        id_prefix: Optional VCF ID prefix filter, e.g. ``"SNP-"`` or ``"SNP- | INDEL-"``.

    Returns:
        snp_matrix: Encoded SNP matrix (shape depends on encoding type)
        sample_ids: List of sample IDs
        snp_ids: List of SNP IDs

        Automatic detection returns one branch per detected type.

    Raises:
        ValueError: If encoding_type or variant_type is not recognized
    """
    # Validate inputs
    if encoding_type not in ['token', 'diploid_onehot', 'onehot', '10classed_onehot']:
        raise ValueError(
            "encoding_type must be 'token', 'diploid_onehot', 'onehot', or "
            f"'10classed_onehot', got '{encoding_type}'"
        )

    if variant_type is not None and variant_type not in ['snp', 'indel', 'sv']:
        raise ValueError(
            f"variant_type must be 'snp', 'indel', or 'sv', got '{variant_type}'"
        )

    if encoding_type in ('onehot', '10classed_onehot') and variant_type not in (None, 'snp'):
        raise ValueError(
            f"encoding_type '{encoding_type}' is only supported for single-branch SNP VCF "
            "(variant_type 'snp' or omitted)."
        )

    id_prefixes = parse_id_prefix_spec(id_prefix)

    # Handle single-variant types (non-multi-branch)
    if variant_type is None:
        if encoding_type == 'token':
            return parse_genotype_token(geno_path)
        if encoding_type == 'onehot':
            return parse_genotype_snp_vcf_onehot(
                geno_path, id_prefixes=id_prefixes
            )
        if encoding_type == '10classed_onehot':
            return parse_genotype_snp_vcf_10classed_onehot(
                geno_path, id_prefixes=id_prefixes
            )
        return parse_genotype_vcf(geno_path, id_prefixes=id_prefixes)

    if variant_type == 'snp':
        if encoding_type == 'token':
            return parse_genotype_token(geno_path)
        if encoding_type == 'onehot':
            return ensure_snp_sites_are_acgt(parse_genotype_snp_vcf_onehot(
                geno_path,
                assume_all_variants=True,
                id_prefixes=id_prefixes,
            ))
        if encoding_type == '10classed_onehot':
            return ensure_snp_sites_are_acgt(parse_genotype_snp_vcf_10classed_onehot(
                geno_path,
                assume_all_variants=True,
                id_prefixes=id_prefixes,
            ))
        # diploid_onehot
        return ensure_snp_sites_are_acgt(parse_genotype_snp_vcf(
            geno_path,
            assume_all_variants=True,
            id_prefixes=id_prefixes,
        ))

    if variant_type == 'indel':
        return parse_genotype_indel_vcf(
            geno_path,
            assume_all_variants=True,
            id_prefixes=id_prefixes,
        )
    if variant_type == 'sv':
        return parse_genotype_sv_vcf(
            geno_path,
            assume_all_variants=True,
            id_prefixes=id_prefixes,
        )
    # Fallback (should not reach here)
    raise ValueError(f"Unknown variant_type: {variant_type}")


###############################################################################
# VCF Writing Utilities (for directed evolution output)
###############################################################################

def onehot_to_gt_diploid(
    encoding: np.ndarray, ref_allele: str, alt_allele: str
) -> Tuple[int, int]:
    """
    Convert 8-dimensional diploid nucleotide one-hot encoding back to VCF GT allele indices.

    Args:
        encoding: (8,) array where [:4] is allele1 A/C/G/T, [4:] is allele2 A/C/G/T.
        ref_allele: The REF allele nucleotide (e.g. 'A', 'T').
        alt_allele: The ALT allele nucleotide (e.g. 'T', 'G').

    Returns:
        Tuple of (allele1_idx, allele2_idx) where 0=REF, 1=ALT in VCF terms,
        or (-1, -1) if the encoding is all-zeros (missing).

    Encoding map (alphabetical order A,C,G,T -> indices 0,1,2,3):
        Position 0=A, 1=C, 2=G, 3=T
        First 4 positions: haplotype 1
        Last 4 positions: haplotype 2

    Example (REF=A, ALT=T):
        - [0,0,0,1, 1,0,0,0] → haplotype1=T(ALT), haplotype2=A(REF) → (1, 0) → "1|0"
        - [1,0,0,0, 0,0,0,1] → haplotype1=A(REF), haplotype2=T(ALT) → (0, 1) → "0|1"
    """
    if encoding.sum() == 0:
        return (-1, -1)  # Missing

    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ref_idx = nuc_to_idx.get(ref_allele.upper(), 0)
    alt_idx = nuc_to_idx.get(alt_allele.upper(), 1)

    hap1_nuc_idx = int(np.argmax(encoding[:4]))
    hap2_nuc_idx = int(np.argmax(encoding[4:]))

    allele1 = 0 if hap1_nuc_idx == ref_idx else (1 if hap1_nuc_idx == alt_idx else 0)
    allele2 = 0 if hap2_nuc_idx == ref_idx else (1 if hap2_nuc_idx == alt_idx else 0)

    return (allele1, allele2)


def onehot_to_gt_genotype_class(encoding: np.ndarray) -> Tuple[int, int]:
    """
    Convert 4-dimensional genotype-class one-hot encoding back to GT allele indices.

    Args:
        encoding: (4,) array: [REF/REF, REF/ALT, ALT/REF, ALT/ALT].

    Returns:
        Tuple of (allele1_idx, allele2_idx) where 0=REF, 1=ALT, or (-1,-1) if missing.
    """
    if encoding.sum() == 0:
        return (-1, -1)  # Missing

    idx = np.argmax(encoding)
    # Map: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    return [(0, 0), (0, 1), (1, 0), (1, 1)][idx]


def onehot_to_classic_3way(encoding: np.ndarray) -> int:
    """
    Convert 3-dimensional classic REF/HET/ALT one-hot encoding back to class index.

    Args:
        encoding: (3,) array: [REF/REF, HET, ALT/ALT].

    Returns:
        0 for REF/REF, 1 for HET, 2 for ALT/ALT, -1 for missing.
    """
    if encoding.sum() == 0:
        return -1  # Missing
    return int(np.argmax(encoding))


def token_to_gt(token: int) -> Tuple[int, int]:
    """
    Convert token encoding {0,1,2,3} to GT allele indices.

    Token encoding:
        0: Homozygous alternate (ALT/ALT) -> (1, 1)
        1: Homozygous reference (REF/REF) -> (0, 0)
        2: Heterozygous (REF/ALT) -> (0, 1)
        3: Missing -> (-1, -1)

    Returns:
        Tuple of (allele1_idx, allele2_idx) where 0=REF, 1=ALT.
    """
    return [(1, 1), (0, 0), (0, 1), (-1, -1)][token]


def build_vcf_header_lines(vcf_path: str) -> List[str]:
    """
    Extract all header lines from a VCF file (everything before the first non-comment line).

    Args:
        vcf_path: Path to the input VCF file.

    Returns:
        List of header lines (including the #CHROM line).
    """
    is_gzipped = vcf_path.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    open_mode = 'rt' if is_gzipped else 'r'

    header_lines = []
    with open_func(vcf_path, open_mode) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                header_lines.append(line)
            else:
                break

    return header_lines


def parse_vcf_variant_rows(vcf_path: str) -> Tuple[List[dict], List[str]]:
    """
    Parse all variant rows from a VCF file, extracting metadata needed for writing.

    Args:
        vcf_path: Path to the input VCF file.

    Returns:
        Tuple of (variant_rows, header_lines) where variant_rows is a list of dicts:
        {
            'chrom': str,
            'pos': str,
            'id': str,
            'ref': str,
            'alt': str,
            'qual': str,
            'filter': str,
            'info': str,
            'format': str,
            'genotypes': list of str,  # raw GT fields per sample
        }
        and header_lines are all lines starting with '#'.
    """
    is_gzipped = vcf_path.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    open_mode = 'rt' if is_gzipped else 'r'

    variant_rows = []
    header_lines = []

    with open_func(vcf_path, open_mode) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                header_lines.append(line)
            else:
                fields = line.split('\t')
                if len(fields) < 10:
                    continue
                variant_rows.append({
                    'chrom': fields[0],
                    'pos': fields[1],
                    'id': fields[2],
                    'ref': fields[3],
                    'alt': fields[4],
                    'qual': fields[5],
                    'filter': fields[6],
                    'info': fields[7],
                    'format': fields[8],
                    'genotypes': fields[9:],
                })

    return variant_rows, header_lines


def write_evolved_vcf(
    original_vcf_path: str,
    output_vcf_path: str,
    evolved_genotypes: dict,
    evolved_sample_name: str,
    phased: bool = True,
    verbose: bool = False,
) -> None:
    """
    Write an evolved VCF file by replacing the genotype of the original sample.

    This reverses the VCF parsing: takes the evolved genotype matrices and writes them
    back to VCF format with a new sample name.

    Args:
        original_vcf_path: Path to the original input VCF file.
        output_vcf_path: Path to write the evolved VCF.
        evolved_genotypes: Dict mapping variant_type to evolved genotype array:
            - For SNP (8-dim one-hot): (n_snps, 8) array with 8-dim vectors per SNP
            - For INDEL/SV (4-dim class one-hot): (n_variants, 4) array
            - For token (1-dim): (n_snps,) array with token values {0,1,2,3}
            Keys should be lowercase ('snp', 'indel', 'sv').
        evolved_sample_name: New sample name (e.g. "Teqing__SAMN04505840__evolve").
        phased: If True, use '|' (phased, default); if False, use '/' (unphased).
    """
    variant_rows, header_lines = parse_vcf_variant_rows(original_vcf_path)

    # Build lookup dict: vid -> (vtype, encoding_row)
    # Flatten all variant type arrays into a single lookup by variant ID
    vid_to_info = {}  # vid -> {'vtype': str, 'encoding': array, 'idx': int}

    for vtype, arr in evolved_genotypes.items():
        for idx in range(arr.shape[0]):
            # Try to get variant ID from the VCF rows by matching position/chrom if available
            # Since we don't have a direct VID->encoding mapping, we use row-by-row lookup
            pass  # Will do per-row lookup below

    # Determine GT separator
    sep = '|' if phased else '/'

    # Build variant-type -> index tracker (for sequential matching)
    vtype_idx = {vtype: 0 for vtype in evolved_genotypes}

    # Build encoding dim lookup
    vtype_dims = {}
    for vtype, arr in evolved_genotypes.items():
        vtype_dims[vtype] = arr.shape[-1] if arr.ndim > 1 else 1

    # Write to a plain-text VCF first, then convert to proper BGZF via bcftools.
    # Writing with Python gzip produces non-BGZF deflate blocks that pysam cannot seek.
    if output_vcf_path.endswith('.gz'):
        fd, tmp_path = tempfile.mkstemp(suffix='.vcf', dir=os.path.dirname(output_vcf_path) or '.')
        os.close(fd)
        out_handle = open(tmp_path, 'w')
    else:
        tmp_path = None
        out_handle = open(output_vcf_path, 'w')

    with out_handle as f:
        # Write updated header
        for hl in header_lines:
            if hl.startswith('#CHROM'):
                # Update sample name in header
                fields = hl.split('\t')
                if len(fields) > 9:
                    fields[9] = evolved_sample_name
                    f.write('\t'.join(fields) + '\n')
                else:
                    f.write(hl + '\n')
            elif hl.startswith('##'):
                f.write(hl + '\n')
            else:
                f.write(hl + '\n')

        # Write variant rows with evolved genotypes
        for row in variant_rows:
            vid = row['id']
            orig_genotypes = row['genotypes']
            original_sample_gt = orig_genotypes[0] if orig_genotypes else '.'

            # Determine variant type from ID (case-insensitive check)
            vtype = None
            for key in evolved_genotypes:
                if key.upper() in vid.upper():
                    vtype = key
                    break
            if vtype is None:
                # Default to 'snp' for unknown types
                vtype = 'snp'

            # Get evolved genotype using sequential index for this vtype
            gt_str = original_sample_gt  # Default to original

            if vtype in evolved_genotypes:
                arr = evolved_genotypes[vtype]
                dim = vtype_dims.get(vtype, 1)

                # Check if we have this variant in our evolved genotypes
                idx = vtype_idx[vtype]
                if idx < arr.shape[0]:
                    encoding = arr[idx]
                    vtype_idx[vtype] += 1  # Advance index

                    if dim == 1:
                        token = int(encoding)
                        allele1, allele2 = token_to_gt(token)
                    elif dim == 8:
                        allele1, allele2 = onehot_to_gt_diploid(encoding, row['ref'], row['alt'])
                    elif dim == 4:
                        allele1, allele2 = onehot_to_gt_genotype_class(encoding)
                    elif dim == 3:
                        cls = onehot_to_classic_3way(encoding)
                        if cls == -1:
                            allele1, allele2 = -1, -1
                        elif cls == 0:
                            allele1, allele2 = 0, 0
                        elif cls == 1:
                            allele1, allele2 = 0, 1
                        else:
                            allele1, allele2 = 1, 1
                    else:
                        allele1, allele2 = -1, -1

                    if allele1 == -1 or allele2 == -1:
                        gt_str = '.'
                    else:
                        gt_str = f'{allele1}{sep}{allele2}'

            # Write the row with evolved genotype
            new_genotypes = [gt_str] + list(orig_genotypes[1:])
            new_line = '\t'.join([
                row['chrom'], row['pos'], row['id'], row['ref'], row['alt'],
                row['qual'], row['filter'], row['info'], row['format'],
            ] + new_genotypes) + '\n'
            f.write(new_line)

    # Convert plain VCF to BGZF-compressed VCF via bcftools (indexable by pysam).
    if tmp_path is not None:
        bgzf_tmp = tmp_path + '.bgzf.vcf.gz'
        try:
            subprocess.run(
                ['bcftools', 'view', '--output-type', 'z',
                 '--output', bgzf_tmp, tmp_path],
                check=True, capture_output=True, text=True
            )
            os.replace(bgzf_tmp, output_vcf_path)
            if verbose:
                print(f"  Evolved VCF (BGZF) written to: {output_vcf_path}")

            # Build CSI index so downstream tools (bcftools, sliding_divergence.py) work.
            subprocess.run(
                ['bcftools', 'index', '-f', output_vcf_path],
                check=True, capture_output=True, text=True
            )
            if verbose:
                print(f"  CSI index built for: {output_vcf_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"bcftools BGZF conversion failed.\nStdout: {e.stdout}\nStderr: {e.stderr}"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        if verbose:
            print(f"  Evolved VCF written to: {output_vcf_path}")

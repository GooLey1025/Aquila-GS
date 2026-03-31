"""
Genomic Data Augmentation for Genotype-to-Phenotype Prediction.

This module provides data augmentation strategies for genomic data that are
biologically meaningful and designed for diploid one-hot encoded genotypes.

Encoding format: (n_samples, n_snps, 8)
- First 4 dims = allele 1 one-hot (A, C, G, T)
- Last 4 dims = allele 2 one-hot (A, C, G, T)
- Missing = [0,0,0,0, 0,0,0,0]
"""

import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any


class SNPMasking:
    """
    Randomly mask SNPs by setting to zero vector (simulating missing data).

    This is the most stable augmentation with minimal biological violation.
    Recommended mask_rate: 0.02~0.08 (start with 0.03)

    Args:
        mask_rate: Probability of masking each SNP position (default: 0.03)
    """

    def __init__(self, mask_rate: float = 0.03):
        assert 0.0 <= mask_rate <= 1.0, f"mask_rate must be in [0, 1], got {mask_rate}"
        self.mask_rate = mask_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random SNP masking.

        Args:
            x: Genotype tensor of shape (n_snps, 8)

        Returns:
            Augmented tensor with randomly masked SNPs
        """
        if self.mask_rate == 0:
            return x

        # Create a copy to avoid modifying original data
        x = x.clone()

        n_snps = x.shape[0]

        # Generate mask for each SNP position
        mask = torch.rand(n_snps) < self.mask_rate

        # Set masked positions to zero vector
        x[mask] = 0.0

        return x

    def __repr__(self) -> str:
        return f"SNPMasking(mask_rate={self.mask_rate})"


class Cutout:
    """
    Mask contiguous genomic regions (like Cutout in computer vision).

    Forces the model to not rely solely on local positions or LD blocks.
    Recommended: 1-3 segments per sample, each segment = 0.5%~2% of total SNPs

    Args:
        num_segments: Number of contiguous segments to mask (default: 2)
        segment_ratio: Fraction of total SNPs for each segment length (default: 0.01)
        segment_length: Fixed segment length in SNPs (overrides segment_ratio if set)
    """

    def __init__(
        self,
        num_segments: int = 2,
        segment_ratio: float = 0.01,
        segment_length: Optional[int] = None
    ):
        assert num_segments >= 1, f"num_segments must be >= 1, got {num_segments}"
        assert 0.0 < segment_ratio <= 0.5, f"segment_ratio must be in (0, 0.5], got {segment_ratio}"

        self.num_segments = num_segments
        self.segment_ratio = segment_ratio
        self.segment_length = segment_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply contiguous region masking (Cutout).

        Args:
            x: Genotype tensor of shape (n_snps, 8)

        Returns:
            Augmented tensor with contiguous regions masked
        """
        # Create a copy to avoid modifying original data
        x = x.clone()

        n_snps = x.shape[0]

        # Determine segment length
        if self.segment_length is not None:
            seg_len = min(self.segment_length, n_snps // 2)
        else:
            seg_len = max(1, int(n_snps * self.segment_ratio))

        # Apply multiple cutout segments
        for _ in range(self.num_segments):
            # Random start position
            if n_snps <= seg_len:
                start = 0
                end = n_snps
            else:
                start = torch.randint(0, n_snps - seg_len + 1, (1,)).item()
                end = start + seg_len

            # Mask the segment
            x[start:end] = 0.0

        return x

    def __repr__(self) -> str:
        if self.segment_length is not None:
            return f"Cutout(num_segments={self.num_segments}, segment_length={self.segment_length})"
        return f"Cutout(num_segments={self.num_segments}, segment_ratio={self.segment_ratio})"


class AlleleSwap:
    """
    Swap allele order in heterozygous sites (0|1 <-> 1|0).

    This is essentially "free" augmentation when phase information is unreliable
    (common in most genomic selection studies). Only affects heterozygous sites.

    Supports two encoding schemes:
    1. Diploid nucleotide one-hot (8-dim): First 4 dims = allele 1, last 4 dims = allele 2
       - Heterozygous: first 4 dims != last 4 dims (both non-zero)
    2. Genotype-class one-hot (4-dim): INDEL/SV encoding
       - Heterozygous: [0,1,0,0] or [0,0,1,0] (index 1 or 2)

    Two modes:
    1. Per-site swap: Each heterozygous site has swap_prob chance to swap
    2. Whole-sample swap: With swap_prob chance, swap ALL heterozygous sites

    Args:
        swap_prob: Probability of swapping (default: 0.5)
        whole_sample: If True, swap all het sites together; if False, per-site (default: False)
    """

    def __init__(self, swap_prob: float = 0.5, whole_sample: bool = False):
        assert 0.0 <= swap_prob <= 1.0, f"swap_prob must be in [0, 1], got {swap_prob}"
        self.swap_prob = swap_prob
        self.whole_sample = whole_sample

    def _is_heterozygous(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect heterozygous sites based on encoding type.

        For 8-dim diploid nucleotide one-hot (SNP):
            Heterozygous if first 4 dims != last 4 dims (both non-zero)

        For 4-dim genotype-class one-hot (INDEL/SV):
            Heterozygous if index 1 or 2 is active ([0,1,0,0] or [0,0,1,0])

        Args:
            x: Genotype tensor of shape (n_variants, 8) or (n_variants, 4)

        Returns:
            Boolean tensor of shape (n_variants,) indicating heterozygous sites
        """
        if x.shape[-1] == 8:
            # SNP: 8-dim diploid nucleotide one-hot
            allele1 = x[:, :4]
            allele2 = x[:, 4:]

            # Heterozygous if alleles differ (and not missing)
            # Check if any position differs AND both alleles are non-zero
            is_valid = (allele1.sum(dim=1) > 0) & (allele2.sum(dim=1) > 0)
            is_different = (allele1 != allele2).any(dim=1)

            return is_valid & is_different

        elif x.shape[-1] == 4:
            # INDEL/SV: 4-dim genotype-class one-hot
            # Heterozygous if index 1 ([0,1,0,0]) or index 2 ([0,0,1,0]) is active
            # This is equivalent to: sum > 0 AND sum < 1 (not REF/REF and not ALT/ALT)
            # REF/REF: [1,0,0,0] -> sum=1
            # REF/ALT: [0,1,0,0] -> sum=1
            # ALT/REF: [0,0,1,0] -> sum=1
            # ALT/ALT: [0,0,0,1] -> sum=1
            # Missing: [0,0,0,0] -> sum=0
            # All non-missing have sum=1, so we check: is index 1 or 2 active?
            is_index1 = x[:, 1] == 1  # REF/ALT (0|1)
            is_index2 = x[:, 2] == 1  # ALT/REF (1|0)
            is_missing = x.sum(dim=1) == 0

            return (is_index1 | is_index2) & ~is_missing

        else:
            # Unknown encoding, assume no heterozygous sites
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply allele swapping for heterozygous sites.

        Args:
            x: Genotype tensor of shape (n_variants, 8) or (n_variants, 4)

        Returns:
            Augmented tensor with swapped alleles at heterozygous sites
        """
        if self.swap_prob == 0:
            return x

        # Create a copy to avoid modifying original data
        x = x.clone()

        # Find heterozygous sites
        het_mask = self._is_heterozygous(x)

        if not het_mask.any():
            return x

        if x.shape[-1] == 8:
            # SNP: swap first 4 dims with last 4 dims
            if self.whole_sample:
                if torch.rand(1).item() < self.swap_prob:
                    het_indices = het_mask.nonzero(as_tuple=True)[0]
                    temp = x[het_indices, :4].clone()
                    x[het_indices, :4] = x[het_indices, 4:]
                    x[het_indices, 4:] = temp
            else:
                het_indices = het_mask.nonzero(as_tuple=True)[0]
                swap_decisions = torch.rand(len(het_indices)) < self.swap_prob
                sites_to_swap = het_indices[swap_decisions]

                if len(sites_to_swap) > 0:
                    temp = x[sites_to_swap, :4].clone()
                    x[sites_to_swap, :4] = x[sites_to_swap, 4:]
                    x[sites_to_swap, 4:] = temp

        elif x.shape[-1] == 4:
            # INDEL/SV: swap REF/ALT (index 1) with ALT/REF (index 2)
            if self.whole_sample:
                if torch.rand(1).item() < self.swap_prob:
                    het_indices = het_mask.nonzero(as_tuple=True)[0]
                    # Swap index 1 <-> index 2
                    temp = x[het_indices, 1].clone()
                    x[het_indices, 1] = x[het_indices, 2]
                    x[het_indices, 2] = temp
            else:
                het_indices = het_mask.nonzero(as_tuple=True)[0]
                swap_decisions = torch.rand(len(het_indices)) < self.swap_prob
                sites_to_swap = het_indices[swap_decisions]

                if len(sites_to_swap) > 0:
                    temp = x[sites_to_swap, 1].clone()
                    x[sites_to_swap, 1] = x[sites_to_swap, 2]
                    x[sites_to_swap, 2] = temp

        return x

    def __repr__(self) -> str:
        return f"AlleleSwap(swap_prob={self.swap_prob}, whole_sample={self.whole_sample})"


class GenomicAugmentation:
    """
    Compose multiple genomic augmentations into a pipeline.

    Args:
        augmentations: List of augmentation instances to apply
        p: Probability of applying the entire augmentation pipeline (default: 1.0)
             Set to < 1.0 to randomly skip augmentation for some samples
    """

    def __init__(
        self,
        augmentations: Optional[List] = None,
        p: float = 1.0
    ):
        assert 0.0 <= p <= 1.0, f"p must be in [0, 1], got {p}"
        self.augmentations = augmentations or []
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation pipeline.

        Args:
            x: Genotype tensor of shape (n_snps, 8)

        Returns:
            Augmented tensor
        """
        # Skip augmentation with probability (1 - p)
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return x

        # Apply each augmentation in sequence
        for aug in self.augmentations:
            x = aug(x)

        return x

    def __repr__(self) -> str:
        aug_strs = [repr(aug) for aug in self.augmentations]
        return f"GenomicAugmentation(p={self.p}, augmentations=[{', '.join(aug_strs)}])"


def create_augmentation_from_config(config: Optional[Dict[str, Any]]) -> Optional[GenomicAugmentation]:
    """
    Create a GenomicAugmentation pipeline from a configuration dictionary.

    Args:
        config: Augmentation configuration dictionary. Example:
            {
                'enabled': True,
                'p': 1.0,  # probability of applying augmentation
                'snp_masking': {'mask_rate': 0.03},
                'cutout': {'num_segments': 2, 'segment_ratio': 0.01},
                'allele_swap': {'swap_prob': 0.5, 'whole_sample': False}
            }

    Returns:
        GenomicAugmentation instance or None if disabled/not configured
    """
    if config is None:
        return None

    if not config.get('enabled', True):
        return None

    augmentations = []

    # SNP Masking
    if 'snp_masking' in config:
        snp_config = config['snp_masking']
        if snp_config is not None:
            augmentations.append(SNPMasking(
                mask_rate=snp_config.get('mask_rate', 0.03)
            ))

    # Cutout
    if 'cutout' in config:
        cutout_config = config['cutout']
        if cutout_config is not None:
            augmentations.append(Cutout(
                num_segments=cutout_config.get('num_segments', 2),
                segment_ratio=cutout_config.get('segment_ratio', 0.01),
                segment_length=cutout_config.get('segment_length', None)
            ))

    # Allele Swap
    if 'allele_swap' in config:
        swap_config = config['allele_swap']
        if swap_config is not None:
            augmentations.append(AlleleSwap(
                swap_prob=swap_config.get('swap_prob', 0.5),
                whole_sample=swap_config.get('whole_sample', False)
            ))

    if not augmentations:
        return None

    return GenomicAugmentation(
        augmentations=augmentations,
        p=config.get('p', 1.0)
    )


# Convenience function to create default augmentation
def default_augmentation() -> GenomicAugmentation:
    """
    Create a default augmentation pipeline with recommended settings.

    Returns:
        GenomicAugmentation with SNPMasking, Cutout, and AlleleSwap
    """
    return GenomicAugmentation(
        augmentations=[
            SNPMasking(mask_rate=0.03),
            Cutout(num_segments=2, segment_ratio=0.01),
            AlleleSwap(swap_prob=0.5, whole_sample=False),
        ],
        p=1.0
    )

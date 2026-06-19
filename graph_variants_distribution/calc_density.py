#!/usr/bin/env python3
"""
Calculate INDEL, SV, and SNP density from variant_positions.tsv
Optimized version using vectorized operations
"""

import pandas as pd
import numpy as np

# Read karyotype to get chromosome lengths
karyotype = pd.read_csv('karyotype.txt', sep='\t')
print("Karyotype:")
print(karyotype.head())

# Create window size (100kb)
window_size = 100000

# Read variant positions
print("\nReading variant_positions.tsv...")
variants = pd.read_csv('variant_positions.tsv', sep='\t')
print(f"Total variants: {len(variants)}")
print("\nVariant types:", variants['Type'].value_counts())

# Process each chromosome
results_indel = []
results_sv = []
results_snp = []

for _, row in karyotype.iterrows():
    chr_num = row['Chr']
    chr_len = row['End']
    
    print(f"\nProcessing Chr {chr_num} (length: {chr_len})...")
    
    # Filter variants for this chromosome
    chr_variants = variants[variants['Chr'] == chr_num]
    
    # INDELs, SVs, SNPs
    indels = chr_variants[chr_variants['Type'] == 'INDEL']
    svs = chr_variants[chr_variants['Type'] == 'SV']
    snps = chr_variants[chr_variants['Type'] == 'SNP']
    
    print(f"  SNPs: {len(snps)}, INDELs: {len(indels)}, SVs: {len(svs)}")
    
    # Calculate density using binning (much faster)
    def calc_density_fast(df, chrom_len, window_size):
        # Calculate window index for each position
        window_idx = (df['Start'].values - 1) // window_size
        # Count by window
        counts = np.bincount(window_idx, minlength=(chrom_len // window_size + 1))
        # Create windows (starting from 1 to match original variant_density.tsv)
        starts = np.arange(1, chrom_len + 1, window_size)
        ends = np.minimum(starts + window_size - 1, chrom_len)
        # Trim to valid windows
        valid = counts > 0
        return pd.DataFrame({
            'Chr': chr_num,
            'Start': starts[valid],
            'End': ends[valid],
            'Value': counts[valid]
        })
    
    indel_density = calc_density_fast(indels, chr_len, window_size)
    results_indel.append(indel_density)
    
    sv_density = calc_density_fast(svs, chr_len, window_size)
    results_sv.append(sv_density)
    
    snp_density = calc_density_fast(snps, chr_len, window_size)
    results_snp.append(snp_density)

# Combine results
indel_df = pd.concat(results_indel, ignore_index=True)
sv_df = pd.concat(results_sv, ignore_index=True)
snp_df = pd.concat(results_snp, ignore_index=True)

# Rename columns
indel_df = indel_df.rename(columns={'Value': 'Value_1'})
sv_df = sv_df.rename(columns={'Value': 'Value_2'})
snp_df = snp_df.rename(columns={'Value': 'Value'})

# Merge all three types
merged = snp_df.merge(indel_df[['Chr', 'Start', 'End', 'Value_1']], 
                      on=['Chr', 'Start', 'End'], 
                      how='outer')
merged = merged.merge(sv_df[['Chr', 'Start', 'End', 'Value_2']], 
                      on=['Chr', 'Start', 'End'], 
                      how='outer')
merged = merged.fillna(0)

# Add colors (SNP: blue #1f78b4, INDEL: green #33a02c, SV: orange #ff7f00)
merged['Color'] = '1f78b4'
merged['Color_1'] = '33a02c'
merged['Color_2'] = 'ff7f00'

# Reorder columns for dual-density format (INDEL and SV)
indel_sv = merged[['Chr', 'Start', 'End', 'Value_1', 'Color_1', 'Value_2', 'Color_2']]
indel_sv = indel_sv.sort_values(['Chr', 'Start']).reset_index(drop=True)

# SNP density alone
snp_only = merged[['Chr', 'Start', 'End', 'Value', 'Color']]
snp_only = snp_only.sort_values(['Chr', 'Start']).reset_index(drop=True)

print("\nINDEL+SV density sample:")
print(indel_sv.head(20))

print("\nSNP density sample:")
print(snp_only.head(20))

# Save files
indel_sv.to_csv('indel_sv_density.tsv', sep='\t', index=False)
print("\nSaved indel_sv_density.tsv")

snp_only.to_csv('snp_density.tsv', sep='\t', index=False)
print("Saved snp_density.tsv")

print("\nDone!")

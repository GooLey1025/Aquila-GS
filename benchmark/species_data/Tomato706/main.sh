#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
P=Tomato706
THREADS="${THREADS:-$(nproc)}"
LD_WINDOW=1000
LD_STEP=50
LD_R2=0.005
FILTER_TAG="max_missing_0.5.maf_0.05.biallelic.filter"

VCF_IN="${P}.merge_snp.vcf.gz"
VCF_KEEP="${P}.merge_snp.keep.vcf.gz"
KEEP_SAMPLES="${P}.keep.samples.txt"
VCF_FILT="${P}.merge_snp.keep.${FILTER_TAG}.vcf.gz"
VCF_IMPUTE="results/${P}.merge_snp.keep.${FILTER_TAG}.impute.biallelic.vcf.gz"
VCF_PRUNED="${P}.LD.vcf.gz"

if python3 -c "import pandas" >/dev/null 2>&1; then
    PY=python3
else
    PY=/data4/gulei/anaconda3/bin/python
fi

# Tomato_filter_wild.xlsx is the cultivated keep list (SLL/SLC/F1).
# Drop samples absent from that list, plus samples with no phenotype.
# VCF IDs are SL###; phenotype IDs are TS-N (SL001 -> TS-1).
"$PY" "${SCRIPT_DIR}/scripts/select_keep_samples.py" \
    "${VCF_IN}" "${KEEP_SAMPLES}"

bcftools view --threads "${THREADS}" \
    -S "${KEEP_SAMPLES}" \
    -Oz -o "${VCF_KEEP}" \
    "${VCF_IN}"
bcftools index --tbi --force "${VCF_KEEP}"

"${COMMON_SCRIPT_DIR}/filter_vcf.sh" "${VCF_KEEP}" "${VCF_FILT}" "${THREADS}"

beagle.nf --snp-vcf "${VCF_FILT}" --nthreads 32 --memory "200 GB" -resume

if [[ ! -f "${VCF_IMPUTE}" ]]; then
    echo "[ERROR] Missing imputed VCF: ${VCF_IMPUTE}" >&2
    echo "[ERROR] Re-run beagle.nf on ${VCF_FILT} before continuing." >&2
    exit 1
fi

"${COMMON_SCRIPT_DIR}/ld_prune_plink2.sh" \
    "${VCF_IMPUTE}" "${P}" "${LD_WINDOW}" "${LD_STEP}" "${LD_R2}"

# SL### in the VCF corresponds to TS-N in Tomato539_pheno.xlsx (SL001 -> TS-1).
"$PY" "${SCRIPT_DIR}/scripts/rename_ld_vcf_and_pheno.py" "${P}"

plink2 \
    --vcf "${VCF_IMPUTE}" \
    --import-max-alleles 255 \
    --vcf-half-call missing \
    --pca 2 \
    --out "${P}.merge_snp.pca"

plink2 \
    --vcf "${VCF_PRUNED}" \
    --pca 2 \
    --vcf-half-call missing \
    --out "${P}.LD.pca"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    PLOT_PY=python3
else
    PLOT_PY=/data4/gulei/anaconda3/bin/python
fi
"$PLOT_PY" plot_pca_comparison.py

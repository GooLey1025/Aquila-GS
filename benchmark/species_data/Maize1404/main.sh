#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
P=Maize1404
THREADS="${THREADS:-$(nproc)}"
N_SNPS=10000
FILTER_TAG="max_missing_0.5.maf_0.05.biallelic.filter"

VCF_IN="${P}.vcf.gz"
VCF_FILT="${P}.${FILTER_TAG}.vcf.gz"
VCF_IMPUTE="results/${P}.${FILTER_TAG}.impute.biallelic.vcf.gz"
VCF_MIC="${P}.MIC.vcf.gz"
VCF_MIC_RENAME="${P}.MIC.rename.vcf.gz"
PHENO_IN="${P}_GSTP004.pheno"
PHENO_OUT="${P}.pheno"

if python3 -c "import pandas" >/dev/null 2>&1; then
    PY=python3
else
    PY=/data4/gulei/anaconda3/bin/python
fi

"${COMMON_SCRIPT_DIR}/filter_vcf.sh" "${VCF_IN}" "${VCF_FILT}" "${THREADS}"

beagle.nf --snp-vcf "${VCF_FILT}" -resume

if [[ ! -f "${VCF_IMPUTE}" ]]; then
    echo "[ERROR] Missing imputed VCF: ${VCF_IMPUTE}" >&2
    exit 1
fi

# One shared 10k panel scored on benchmark.pheno (max MIC across its traits).
"$PY" "${COMMON_SCRIPT_DIR}/mic_select_snps.py" \
    --vcf "${VCF_IMPUTE}" \
    --pheno benchmark.pheno \
    --vcf-out "${VCF_MIC}" \
    --scores-out "${P}.MIC.scores.tsv" \
    --cache-dir "${P}.MIC.cache" \
    --sample-map maize \
    --n-snps "${N_SNPS}" \
    --threads "${THREADS}"

# VCF IDs are CUBIC_MG_*; phenotype LINEs are MG_*. Keep the intersection.
"$PY" "${SCRIPT_DIR}/scripts/rename_ld_vcf_and_pheno.py" \
    "${VCF_MIC}" "${PHENO_IN}" "${VCF_MIC_RENAME}" "${PHENO_OUT}"

plink2 \
    --vcf "${VCF_IMPUTE}" \
    --pca 2 \
    --out "${P}.impute.pca"

plink2 \
    --vcf "${VCF_MIC_RENAME}" \
    --pca 2 \
    --out "${P}.MIC.rename.pca"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    PLOT_PY=python3
else
    PLOT_PY=/data4/gulei/anaconda3/bin/python
fi
"$PLOT_PY" plot_pca.py "${P}.impute.pca"
"$PLOT_PY" plot_pca.py "${P}.MIC.rename.pca"

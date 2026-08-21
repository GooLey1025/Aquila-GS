#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
P=Wheat850
THREADS="${THREADS:-$(nproc)}"
LD_WINDOW=1000
LD_STEP=50
LD_R2=0.005
FILTER_TAG="max_missing_0.5.maf_0.05.biallelic.filter"

VCF_IN="${P}.merge.vcf.gz"
VCF_KEEP="${P}.merge.keep.vcf.gz"
KEEP_SAMPLES="${P}.keep.samples.txt"
VCF_FILT="${P}.merge.keep.${FILTER_TAG}.vcf.gz"
VCF_IMPUTE="results/${P}.merge.keep.${FILTER_TAG}.impute.biallelic.vcf.gz"
VCF_PRUNED="${P}.LD.vcf.gz"

if python3 -c "import pandas" >/dev/null 2>&1; then
    PY=python3
else
    PY=/data4/gulei/anaconda3/bin/python
fi

# Keep only samples with at least one non-missing phenotype in Wheat_All_traits_Matrix.xlsx.
"$PY" "${SCRIPT_DIR}/scripts/select_keep_samples.py" \
    "${VCF_IN}" "${KEEP_SAMPLES}"

bcftools view --threads "${THREADS}" \
    -S "${KEEP_SAMPLES}" \
    -Oz -o "${VCF_KEEP}" \
    "${VCF_IN}"
bcftools index --threads "${THREADS}" --csi --force "${VCF_KEEP}"

"${COMMON_SCRIPT_DIR}/filter_vcf.sh" "${VCF_KEEP}" "${VCF_FILT}" "${THREADS}"

beagle.nf --snp-vcf "${VCF_FILT}" --nthreads 24 --memory "180 GB" -resume

if [[ ! -f "${VCF_IMPUTE}" ]]; then
    echo "[ERROR] Missing imputed VCF: ${VCF_IMPUTE}" >&2
    echo "[ERROR] Re-run beagle.nf on ${VCF_FILT} before continuing." >&2
    exit 1
fi

"${COMMON_SCRIPT_DIR}/ld_prune_plink2.sh" \
    "${VCF_IMPUTE}" "${P}" "${LD_WINDOW}" "${LD_STEP}" "${LD_R2}"

# Phenotype IDs follow Wheat_All_traits_Matrix.xlsx (All_Data IID).
"$PY" "${SCRIPT_DIR}/scripts/write_ld_pheno.py" "${P}"

plink2 \
    --vcf "${VCF_IMPUTE}" \
    --pca 2 \
    --out "${P}.merge.pca"

plink2 \
    --vcf "${VCF_PRUNED}" \
    --pca 2 \
    --out "${P}.LD.pca"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    PLOT_PY=python3
else
    PLOT_PY=/data4/gulei/anaconda3/bin/python
fi
"$PLOT_PY" plot_pca_comparison.py

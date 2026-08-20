#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
P=Soybean2795
THREADS="${THREADS:-$(nproc)}"
LD_WINDOW=1000
LD_STEP=50
LD_R2=0.003
FILTER_TAG="max_missing_0.5.maf_0.05.biallelic.filter"

VCF_IN="${P}.merge.vcf.gz"
KEEP_LIST="improved_cultivar_pheno.samples.txt"
PHENO_IN="GSTP014.pheno"
PHENO_OUT="${P}.improved_cultivar.pheno"
INFO_XLSX="GSTP014_info.xlsx"
VCF_KEEP="${P}.merge.improved_cultivar_pheno.vcf.gz"
VCF_FILT="${P}.merge.improved_cultivar_pheno.${FILTER_TAG}.vcf.gz"
VCF_IMPUTE="results/${P}.merge.improved_cultivar_pheno.${FILTER_TAG}.impute.biallelic.vcf.gz"
VCF_PRUNED="${P}.LD.vcf.gz"
VCF_PRUNED_RENAME="${P}.LD.rename.vcf.gz"

if python3 -c "import pandas" >/dev/null 2>&1; then
    PY=python3
else
    PY=/data4/gulei/anaconda3/bin/python
fi

# Improved cultivar ∩ 至少有一个非 NA 表型；写出 VCF 原样本名（s63_s63）供 bcftools -S 使用。
"$PY" "${SCRIPT_DIR}/scripts/select_group_pheno_samples.py" \
    "${VCF_IN}" "${INFO_XLSX}" "${PHENO_IN}" "${KEEP_LIST}" "${PHENO_OUT}" \
    --group "Improved cultivar"

bcftools view --threads "${THREADS}" \
    -S "${KEEP_LIST}" \
    -Oz -o "${VCF_KEEP}" \
    "${VCF_IN}"
bcftools index --threads "${THREADS}" -f "${VCF_KEEP}"

"${COMMON_SCRIPT_DIR}/filter_vcf.sh" "${VCF_KEEP}" "${VCF_FILT}" "${THREADS}"

beagle.nf --snp-vcf "${VCF_FILT}" -resume

if [[ ! -f "${VCF_IMPUTE}" ]]; then
    echo "[ERROR] Missing imputed VCF: ${VCF_IMPUTE}" >&2
    exit 1
fi

"${COMMON_SCRIPT_DIR}/ld_prune_plink2.sh" \
    "${VCF_IMPUTE}" "${P}" "${LD_WINDOW}" "${LD_STEP}" "${LD_R2}"

bcftools query -l "${VCF_PRUNED}" > header.txt
sed -i 's/_.*//' header.txt
bcftools reheader -N header.txt -o "${VCF_PRUNED_RENAME}" "${VCF_PRUNED}"
bcftools index --threads "${THREADS}" -f "${VCF_PRUNED_RENAME}"

plink2 \
    --vcf "${VCF_IMPUTE}" \
    --pca 2 \
    --out "${P}.merge.improved_cultivar_pheno.${FILTER_TAG}.impute.biallelic.pca"

plink2 \
    --vcf "${VCF_PRUNED_RENAME}" \
    --pca 2 \
    --out "${P}.improved_cultivar_pheno.${FILTER_TAG}.imputed_pruned.pca"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    PLOT_PY=python3
else
    PLOT_PY=/data4/gulei/anaconda3/bin/python
fi
"$PLOT_PY" plot_pca_comparison.py

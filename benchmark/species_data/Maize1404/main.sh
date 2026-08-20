#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
P=Maize1404
THREADS="${THREADS:-$(nproc)}"
LD_WINDOW=1000
LD_STEP=50
LD_R2=0.003
FILTER_TAG="max_missing_0.5.maf_0.05.biallelic.filter"

VCF_IN="${P}.vcf.gz"
VCF_FILT="${P}.${FILTER_TAG}.vcf.gz"
VCF_IMPUTE="results/${P}.${FILTER_TAG}.impute.biallelic.vcf.gz"
VCF_PRUNED="${P}.LD.vcf.gz"
VCF_PRUNED_RENAME="${P}.LD.rename.vcf.gz"

"${COMMON_SCRIPT_DIR}/filter_vcf.sh" "${VCF_IN}" "${VCF_FILT}" "${THREADS}"

beagle.nf --snp-vcf "${VCF_FILT}" -resume

if [[ ! -f "${VCF_IMPUTE}" ]]; then
    echo "[ERROR] Missing imputed VCF: ${VCF_IMPUTE}" >&2
    exit 1
fi

"${COMMON_SCRIPT_DIR}/ld_prune_plink2.sh" \
    "${VCF_IMPUTE}" "${P}" "${LD_WINDOW}" "${LD_STEP}" "${LD_R2}"

bcftools query -l "${VCF_PRUNED}" > sample_name.txt
sed -i 's/^0_CUBIC_//' sample_name.txt
bcftools reheader -N sample_name.txt "${VCF_PRUNED}" -o "${VCF_PRUNED_RENAME}"
bcftools index --threads "${THREADS}" -f "${VCF_PRUNED_RENAME}"

plink2 \
    --vcf "${VCF_IMPUTE}" \
    --pca 2 \
    --out "${P}.impute.pca"

plink2 \
    --vcf "${VCF_PRUNED_RENAME}" \
    --pca 2 \
    --out "${P}.LD.rename.pca"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    PLOT_PY=python3
else
    PLOT_PY=/data4/gulei/anaconda3/bin/python
fi
"$PLOT_PY" plot_pca.py "${P}.impute"
"$PLOT_PY" plot_pca.py "${P}.LD.rename"

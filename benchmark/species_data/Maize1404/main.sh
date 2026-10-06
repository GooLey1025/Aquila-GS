#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
P=Maize1404
THREADS="${THREADS:-$(nproc)}"
FILTER_TAG="max_missing_0.5.maf_0.05.biallelic.filter"

VCF_IN="${P}.vcf.gz"
VCF_FILT="${P}.${FILTER_TAG}.vcf.gz"
VCF_IMPUTE="results/${P}.${FILTER_TAG}.impute.biallelic.vcf.gz"
PHENO_IN="${P}_GSTP004.pheno"
PHENO_OUT="${P}.pheno"
BENCHMARK_PHENO="benchmark.blup.pheno"
SINGLE_PHENO_DIR="single_phenotypes"

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

plink2 \
    --vcf "${VCF_IMPUTE}" \
    --pca 2 \
    --out "${P}.impute.pca"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    PLOT_PY=python3
else
    PLOT_PY=/data4/gulei/anaconda3/bin/python
fi
"$PLOT_PY" plot_pca.py "${P}.impute.pca"

# Synonymous and missense SNPs, merged, then LD-pruned.
# Coordinates match B73 RefGen_v3. PLINK REF alleles are flipped onto that
# reference before annotation. Sample names drop the CUBIC_ prefix so they
# match benchmark.pheno.
"${COMMON_SCRIPT_DIR}/prepare_zea_mays_agpv3.sh"
"${COMMON_SCRIPT_DIR}/snpeff_coding_ld.sh" \
    --java /usr/lib/jvm/java-21-openjdk-amd64/bin/java \
    --snpeff-home /data4/gulei/snpEff \
    --vcf "${VCF_IMPUTE}" \
    --genome Zea_mays_AGPv3 \
    --fasta /data4/gulei/snpEff/genomes/Zea_mays_AGPv3/sequences.fa \
    --prefix "${SCRIPT_DIR}/Maize1404.coding" \
    --skip-ld \
    --threads "${THREADS}"

BOTH_VCF="${SCRIPT_DIR}/Maize1404.coding.both.vcf.gz"
BOTH_LD="${SCRIPT_DIR}/Maize1404.coding.both.ld"
FINAL_VCF="${SCRIPT_DIR}/Maize1404.coding.ld.vcf.gz"
if [[ ! -s "${BOTH_VCF}" ]]; then
    bcftools concat -a \
        "${SCRIPT_DIR}/Maize1404.coding.syn.vcf.gz" \
        "${SCRIPT_DIR}/Maize1404.coding.nonsyn.vcf.gz" \
        -Ou \
        | bcftools sort -Oz -o "${BOTH_VCF}"
    bcftools index -f "${BOTH_VCF}"
fi
if [[ ! -s "${BOTH_LD}.prune.in" ]]; then
    plink2 \
        --vcf "${BOTH_VCF}" \
        --indep-pairwise 1000 50 0.1 \
        --threads "${THREADS}" \
        --memory 30000 \
        --out "${BOTH_LD}"
fi
if [[ ! -s "${FINAL_VCF}" ]]; then
    bcftools query -l "${BOTH_VCF}" \
        | awk '{ name = $1; sub(/^CUBIC_/, "", name); print name }' \
        > "${SCRIPT_DIR}/Maize1404.coding.ld.samples"
    bcftools view -i "ID=@${BOTH_LD}.prune.in" -Oz -o "${FINAL_VCF}.tmp.vcf.gz" --threads "${THREADS}" "${BOTH_VCF}"
    bcftools reheader -N "${SCRIPT_DIR}/Maize1404.coding.ld.samples" -o "${FINAL_VCF}" "${FINAL_VCF}.tmp.vcf.gz"
    rm -f "${FINAL_VCF}.tmp.vcf.gz"
    bcftools index -f "${FINAL_VCF}"
fi
echo "[INFO] ${FINAL_VCF}: $(bcftools index -n "${FINAL_VCF}") variants"

# VCF IDs are CUBIC_MG_*; phenotype LINEs are MG_*. Keep their intersection.
"$PY" "${SCRIPT_DIR}/scripts/rename_ld_vcf_and_pheno.py" \
    "${FINAL_VCF}" "${PHENO_IN}" "${FINAL_VCF}.renamed.vcf.gz" "${PHENO_OUT}"
mv -f "${FINAL_VCF}.renamed.vcf.gz" "${FINAL_VCF}"
mv -f "${FINAL_VCF}.renamed.vcf.gz.csi" "${FINAL_VCF}.csi"

# Representative, nearly complete agronomic traits:
# flowering time (DTS), plant/ear height (PH/EH), and yield components
# kernel number and kernel weight per ear (KNPE/KWPE).
"$PY" "${COMMON_SCRIPT_DIR}/export_benchmark_phenotypes.py" \
    --input "${PHENO_OUT}" \
    --output "${BENCHMARK_PHENO}" \
    --single-dir "${SINGLE_PHENO_DIR}" \
    --traits DTS PH EH KNPE KWPE \
    --min-observed 100 \
    --min-observed-fraction 0.5

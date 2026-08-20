#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 INPUT_VCF PREFIX [LD_WINDOW] [LD_STEP] [LD_R2]" >&2
    exit 1
fi

INPUT_VCF="$1"
PREFIX="$2"
LD_WINDOW="${3:-1000}"
LD_STEP="${4:-50}"
LD_R2="${5:-0.003}"

plink2 \
    --vcf "${INPUT_VCF}" \
    --make-bed \
    --out "${PREFIX}.impute"

plink2 \
    --bfile "${PREFIX}.impute" \
    --snps-only just-acgt \
    --set-all-var-ids '@:#:$r:$a' \
    --make-pgen \
    --out "${PREFIX}.SNP" \
    --memory 220000

plink2 \
    --pfile "${PREFIX}.SNP" \
    --indep-pairwise "${LD_WINDOW}" "${LD_STEP}" "${LD_R2}" \
    --out "${PREFIX}.LD" \
    --memory 220000

plink2 \
    --pfile "${PREFIX}.SNP" \
    --extract "${PREFIX}.LD.prune.in" \
    --recode vcf bgz \
    --out "${PREFIX}.LD"

echo "${PREFIX}.LD.vcf.gz"

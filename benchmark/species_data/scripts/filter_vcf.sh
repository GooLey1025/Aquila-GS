#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 INPUT_VCF OUTPUT_VCF [THREADS]" >&2
    exit 1
fi

INPUT_VCF="$1"
OUTPUT_VCF="$2"
THREADS="${3:-$(nproc)}"

bcftools view --threads "${THREADS}" \
    -m2 -M2 -v snps \
    -i 'F_MISSING<=0.5 && MAF>=0.05' \
    -Oz -o "${OUTPUT_VCF}" \
    "${INPUT_VCF}"
bcftools index --threads "${THREADS}" -f "${OUTPUT_VCF}"

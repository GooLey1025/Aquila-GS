#!/usr/bin/env bash
# Extract the 120-Rice genotypes, then score them with the production
# rrBLUP and Aquila-SNP models.
# Phenotype BLUPs are fit on all 3605 lines and the 120 lines are extracted
# afterwards. Do not refit the mixed model on the 120 lines alone.
#
# Run from this directory inside the aquila environment:
#   bash main.sh
set -euo pipefail
cd "$(dirname "$0")"

MAP=300lines_map.txt
VCF_IN=Rice655_ref.snp.indel.sv.impute.biallelic.vcf.gz
PHENO=3605rice_Phenotype.tsv
RRBLUP_DIR=../results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.GYP_BLUP.rrblup
AQUILA_DIR=../results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.per-fold-hpo/fold_0
VCF_120=120rice.snp.indel.sv.vcf.gz
TRAITS=(GYP_BLUP)
RRBLUP_PREDICTIONS=predictions/rrblup.GYP_BLUP.10fold-mean.120rice.tsv
RRBLUP_FIGURES=figures/rrblup.GYP_BLUP.10fold-mean
AQUILA_PREDICTIONS=predictions/aquila-snp.fold0.120rice.tsv
AQUILA_FIGURES=figures/aquila-snp.fold0

mkdir -p predictions figures

RSCRIPT="${RSCRIPT:-Rscript}"
if ! "$RSCRIPT" -e 'quit(status=!requireNamespace("jsonlite", quietly=TRUE))' >/dev/null 2>&1; then
  if [[ -x "${HOME}/anaconda3/envs/aquila/bin/Rscript" ]]; then
    RSCRIPT="${HOME}/anaconda3/envs/aquila/bin/Rscript"
  fi
fi

AQUILA_CHECKPOINT="${AQUILA_DIR}/best_model.pt"
AQUILA_CONFIG="${AQUILA_DIR}/config.yaml"
for required_file in "$AQUILA_CHECKPOINT" "$AQUILA_CONFIG"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required Aquila production-model file not found: ${required_file}" >&2
    exit 1
  fi
done
echo "[Aquila] checkpoint: $(realpath "$AQUILA_CHECKPOINT")"
echo "[Aquila] config:     $(realpath "$AQUILA_CONFIG")"

awk -F '\t' 'NR > 1 && $1 == "120-Rice" { print $2 }' "$MAP" > samples_120.txt
n_samples=$(wc -l < samples_120.txt)
if [[ "$n_samples" -ne 120 ]]; then
  echo "Expected 120 sample IDs in ${MAP}, found ${n_samples}" >&2
  exit 1
fi

bcftools view -S samples_120.txt -Oz -o "$VCF_120" "$VCF_IN"
n_vcf=$(bcftools query -l "$VCF_120" | wc -l)
if [[ "$n_vcf" -ne 120 ]]; then
  echo "Expected 120 samples in ${VCF_120}, found ${n_vcf}" >&2
  exit 1
fi
echo "[VCF] ${VCF_120}: ${n_vcf} samples"

python compute_phenotype_blup.py \
  --phenotype "$PHENO" \
  --output 3605rice_Phenotype_BLUP.tsv \
  --samples samples_120.txt \
  --extract 120rice_Phenotype_BLUP.tsv

# python predict_rrblup_vcf.py \
#   --vcf "$VCF_120" \
#   --model-dir "$RRBLUP_DIR" \
#   --traits "${TRAITS[@]}" \
#   --average-folds \
#   --rscript "$RSCRIPT" \
#   -o "$RRBLUP_PREDICTIONS"

aquila_predict.py \
  --checkpoint "$AQUILA_CHECKPOINT" \
  --config "$AQUILA_CONFIG" \
  --vcf "$VCF_120" \
  -o "$AQUILA_PREDICTIONS"

# python correlation_analysis.py \
#   --preds "$RRBLUP_PREDICTIONS" \
#   --blup 120rice_Phenotype_BLUP.tsv \
#   --meta "$MAP" \
#   --model-name "rrBLUP (10-fold mean)" \
#   --output-dir "$RRBLUP_FIGURES"

python correlation_analysis.py \
  --preds "$AQUILA_PREDICTIONS" \
  --blup 120rice_Phenotype_BLUP.tsv \
  --meta "$MAP" \
  --model-name "Aquila-SNP (fold 0 best model)" \
  --output-dir "$AQUILA_FIGURES"

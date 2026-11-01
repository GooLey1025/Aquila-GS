# Population-level integrated gradients for the 10 Rice655 production
# Aquila-SNP seeds (42, plus 43-51), then the GWAS comparison PNGs.
# Seed 42 is the existing multi-trait one-sd production checkpoint.
# Seeds 43-51 are refit on the same data and locked hyperparameters for
# the same 16 epochs (scheduler length stays 30). All PNGs are copied into PNG_DIR.
export MODEL_ROOT=../../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.production.multi_trait.one-sd.10folds.hpo
export DATA_DIR=../../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.production.multi_trait.one-sd.10folds.data
export LOCKED_CONFIG=./production.multi_trait.one-sd.seed_refit.yaml
export VCF=../../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.vcf.gz
export PHENO=../../benchmark/Rice655.pheno
if [[ ! -e ./Rice655.full_marker_panel_gwas_results ]]; then
  rsync -rlthP 6000:/data3/home/gulei/projects/GraphPan/Multi_Source_Marker_Panel_generation/655rice.full_marker_panel/marker_selection/current_gwas/association_results ./Rice655.full_marker_panel_gwas_results
fi
export GWAS_DIR=./Rice655.full_marker_panel_gwas_results
if [[ ! -e ./655rice.canonical_markers.tsv ]]; then
  rsync -rlthP 6000:/data3/home/gulei/projects/GraphPan/Multi_Source_Marker_Panel_generation/655rice.full_marker_panel/marker_panel/655rice.canonical_markers.tsv .
fi
export MARKERS=./655rice.canonical_markers.tsv
export PREFIX=655rice.production.gwas_interpretability
export PNG_DIR=./ig_png_10seeds
export WORK_DIR=./ig_10seeds_work

# GYP_BLUP TGW_BLUP TNP_BLUP GL_BLUP PPNP_BLUP
SEEDS=(42 43 44 45 46 47 48 49 50 51)

mkdir -p "$PNG_DIR"

run_seed() {
  set -euo pipefail
  local SEED="$1"
  local SLOT="$2"
  local GPU=$((SLOT - 1))
  export CUDA_VISIBLE_DEVICES="$GPU"
  local TRAITS=(HD_BLUP GW_BLUP PH_BLUP)
  local MODEL_DIR
  if [[ "$SEED" == "42" ]]; then
    MODEL_DIR="$MODEL_ROOT"
  else
    MODEL_DIR="${MODEL_ROOT}/seeds/seed_${SEED}"
    if [[ ! -f "${MODEL_DIR}/best_model.pt" ]]; then
      echo "[train] seed ${SEED} gpu ${GPU}"
      aquila_train.py \
        --data-dir "$DATA_DIR" \
        --config "$LOCKED_CONFIG" \
        --seed "$SEED" \
        -o "$MODEL_DIR" \
        --precision bf16 \
        --live-metrics-log \
        --use-deterministic
    fi
  fi
  local SEED_WORK="${WORK_DIR}/seed_${SEED}"
  mkdir -p "$SEED_WORK"

  local TRAIT IG_DIR IMPORTANCE_DIR IMPORTANCE PLOT_PDF LOCUS_PDF
  for TRAIT in "${TRAITS[@]}"; do
    IG_DIR=${SEED_WORK}/${PREFIX}.seed_${SEED}.${TRAIT}.ig
    IMPORTANCE_DIR=${SEED_WORK}/${PREFIX}.seed_${SEED}.${TRAIT}.position_importance
    IMPORTANCE=${IMPORTANCE_DIR}/importance_ranking_${TRAIT}.tsv
    PLOT_PDF=${SEED_WORK}/${PREFIX}.seed_${SEED}.${TRAIT}.gwas_ig.pdf

    echo "[IG] seed ${SEED} gpu ${GPU} ${TRAIT}"
    aquila_ig.py \
      --model-dir "$MODEL_DIR" \
      --vcf "$VCF" \
      --encoding-type diploid_onehot \
      --variant-type snp \
      --id-prefix SNP- \
      --task "$TRAIT" \
      --streaming \
      -o "$IG_DIR"

    echo "[rank] seed ${SEED} gpu ${GPU} ${TRAIT}"
    aquila_ig_interpretation.py \
      -i "${IG_DIR}/ig_results.h5" \
      -o "$IMPORTANCE_DIR" \
      --tasks "$TRAIT" \
      --variant-type snp

    python3 gwas_ig_multi_plot_v3.py \
      --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
      --importance "$IMPORTANCE" \
      -o "$PLOT_PDF" \
      --also-png \
      --smooth 5 \
      --ig-top-k 500 \
      --qtn-annot Final_summary_347_QTNsites_geno_redefined.xlsx \
      --annot-tolerance-kb 100
    cp -f "${PLOT_PDF%.pdf}.png" "$PNG_DIR/"

    python3 importance_genotype_and_phenotype_distribution.py \
      --vcf "$VCF" \
      --importance "$IMPORTANCE" \
      --pheno "$PHENO" \
      --trait "$TRAIT" \
      --top-k 10 \
      --out-prefix "${SEED_WORK}/${TRAIT}.seed_${SEED}"

    python3 importance_genotype_and_phenotype_distribution_plot.py \
      --long-table "${SEED_WORK}/${TRAIT}.seed_${SEED}.long.tsv" \
      --outdir "${SEED_WORK}/${TRAIT}.seed_${SEED}.gt_ph.plots"
  done

  # Check locus: 3-16733441
  TRAIT=GW_BLUP
  IMPORTANCE=${SEED_WORK}/${PREFIX}.seed_${SEED}.${TRAIT}.position_importance/importance_ranking_${TRAIT}.tsv
  LOCUS_PDF=${SEED_WORK}/${PREFIX}.seed_${SEED}.${TRAIT}.chr3-16733441.gwas_ig.pdf
  python3 gwas_ig_multi_plot_v3.py \
    --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
    --importance "$IMPORTANCE" \
    -o "$LOCUS_PDF" \
    --also-png \
    --smooth 5 \
    --ig-top-k 500 \
    --qtn-annot Final_summary_347_QTNsites_geno_redefined.xlsx \
    --highlight 3:16733441
  cp -f "${LOCUS_PDF%.pdf}.png" "$PNG_DIR/"
}

export -f run_seed
parallel --line-buffer --tagstring 'seed {1}' -j 2 run_seed {1} {%} ::: "${SEEDS[@]}"

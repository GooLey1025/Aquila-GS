# Population-level integrated gradients for the Rice655 multi-trait Aquila-SNP
# 10-fold ensemble. Prefer fold_*/best_model.pt written by cv-refit. The
# attributed output is the mean of the inverse-transformed fold predictions.
export MODEL_DIR=../../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo
export DATA_DIR=../../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.data
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

TRAITS=(HD_BLUP GW_BLUP PH_BLUP) #GYP_BLUP TGW_BLUP TNP_BLUP GL_BLUP PPNP_BLUP

if [[ ! -f "${MODEL_DIR}/fold_0/best_model.pt" ]]; then
  echo "[train-folds] ${MODEL_DIR} has no fold checkpoints; backfilling locked folds"
  aquila_train_locked_folds.py \
    --data-dir "$DATA_DIR" \
    --config "$LOCKED_CONFIG" \
    -o "$MODEL_DIR" \
    --precision bf16 \
    --use-deterministic
fi

for TRAIT in "${TRAITS[@]}"; do
  IG_DIR=${PREFIX}.${TRAIT}.ensemble.ig
  IMPORTANCE_DIR=${PREFIX}.${TRAIT}.ensemble.position_importance
  IMPORTANCE=${IMPORTANCE_DIR}/importance_ranking_${TRAIT}.tsv

  echo "[IG] ${TRAIT} 10-fold ensemble"
  aquila_ig_ensemble.py \
    --model-dir "$MODEL_DIR" \
    --vcf "$VCF" \
    --encoding-type diploid_onehot \
    --variant-type snp \
    --id-prefix SNP- \
    --task "$TRAIT" \
    --streaming \
    -o "$IG_DIR"

  echo "[rank] ${TRAIT}"
  aquila_ig_interpretation.py \
    -i "${IG_DIR}/ig_results.h5" \
    -o "$IMPORTANCE_DIR" \
    --tasks "$TRAIT" \
    --variant-type snp

  python3 gwas_ig_multi_plot_v3.py \
    --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
    --importance "$IMPORTANCE" \
    -o "${PREFIX}.${TRAIT}.ensemble.gwas_ig.pdf" \
    --also-png \
    --smooth 5 \
    --ig-top-k 500 \
    --qtn-annot Final_summary_347_QTNsites_geno_redefined.xlsx \
    --annot-tolerance-kb 100

  python3 importance_genotype_and_phenotype_distribution.py \
    --vcf "$VCF" \
    --importance "$IMPORTANCE" \
    --pheno "$PHENO" \
    --trait "$TRAIT" \
    --top-k 10 \
    --out-prefix "${TRAIT}.ensemble"

  python3 importance_genotype_and_phenotype_distribution_plot.py \
    --long-table "${TRAIT}.ensemble.long.tsv" \
    --outdir "${TRAIT}.gt_ph.ensemble.plots"
done

python3 plot_topk_gwas_marker_venn_v3.py \
  --importance ${PREFIX}.HD_BLUP.ensemble.position_importance/importance_ranking_HD_BLUP.tsv \
  --gwas \
    ${GWAS_DIR}/HD_BeiJ15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/HD_BLUP.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/HD_LingS15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/HD_LingS16.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/HD_WenJ15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/HD_YangZ15.gemma_lmm.assoc.txt \
  --markers "$MARKERS" \
  --trait "Heading date related" \
  --top-k 500 \
  -o HD.ensemble.upset.pdf \
  --summary-tsv HD.ensemble.top500.source_percentage.tsv

python3 plot_topk_gwas_marker_venn_v3.py \
  --importance ${PREFIX}.GW_BLUP.ensemble.position_importance/importance_ranking_GW_BLUP.tsv \
  --gwas \
    ${GWAS_DIR}/GW_BeiJ15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_BLUP.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_LingS16.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_WenJ15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_YangZ15.gemma_lmm.assoc.txt \
  --markers "$MARKERS" \
  --trait "Grain width related" \
  --top-k 500 \
  -o GW.ensemble.upset.pdf \
  --summary-tsv GW.ensemble.top500.source_percentage.tsv

# Check locus: 3-16733441
# TRAIT=GW_BLUP
# IMPORTANCE=${PREFIX}.${TRAIT}.ensemble.position_importance/importance_ranking_${TRAIT}.tsv
# python3 gwas_ig_multi_plot_v3.py \
#     --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
#     --importance "$IMPORTANCE" \
#     -o "${PREFIX}.${TRAIT}.ensemble.chr3-16733441.gwas_ig.pdf" \
#     --also-png \
#     --smooth 5 \
#     --ig-top-k 500 \
#     --qtn-annot Final_summary_347_QTNsites_geno_redefined.xlsx \
#     --highlight 3:16733441

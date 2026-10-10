# Population-level integrated gradients for the Rice655 r2=0.045 production
# Aquila-SNP model, then compare those loci with the existing GWAS results.
export MODEL_DIR=../../production_GS_model_train/results/655rice.r2_0.045.aquila-snp
export VCF=../../production_GS_model_train/655rice.r2_0.045.panel.vcf.gz
export PHENO=../../benchmark/Rice655.pheno
rsync -rlthP 6000:/data3/home/gulei/projects/GraphPan/Multi_Source_Marker_Panel_generation/655rice.full_marker_panel/marker_selection/current_gwas/association_results ./Rice655.full_marker_panel_gwas_results
export GWAS_DIR=./Rice655.full_marker_panel_gwas_results/association_results
rsync -rlthP 6000:/data3/home/gulei/projects/GraphPan/Multi_Source_Marker_Panel_generation/655rice.full_marker_panel/marker_panel/655rice.canonical_markers.tsv .
export MARKERS=./655rice.canonical_markers.tsv
export PREFIX=655rice.r2_0.045.aquila-snp

TRAITS=(HD_BLUP GW_BLUP PH_BLUP) #GYP_BLUP TGW_BLUP TNP_BLUP GL_BLUP PPNP_BLUP

for TRAIT in "${TRAITS[@]}"; do
  IG_DIR=${PREFIX}.${TRAIT}.ig
  IMPORTANCE_DIR=${PREFIX}.${TRAIT}.position_importance
  IMPORTANCE=${IMPORTANCE_DIR}/importance_ranking_${TRAIT}.tsv

  echo "[IG] ${TRAIT}"
  aquila_ig.py \
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
    -o "${PREFIX}.${TRAIT}.gwas_ig.pdf" \
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
    --out-prefix "${TRAIT}.production"

  python3 importance_genotype_and_phenotype_distribution_plot.py \
    --long-table "${TRAIT}.production.long.tsv" \
    --outdir "${TRAIT}.gt_ph.production.plots"
done

python3 plot_topk_gwas_marker_venn_v3.py \
  --importance ${PREFIX}.HD_BLUP.position_importance/importance_ranking_HD_BLUP.tsv \
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
  -o HD.upset.pdf \
  --summary-tsv HD.top500.source_percentage.tsv

python3 plot_topk_gwas_marker_venn_v3.py \
  --importance ${PREFIX}.GW_BLUP.position_importance/importance_ranking_GW_BLUP.tsv \
  --gwas \
    ${GWAS_DIR}/GW_BeiJ15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_BLUP.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_LingS16.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_WenJ15.gemma_lmm.assoc.txt \
    ${GWAS_DIR}/GW_YangZ15.gemma_lmm.assoc.txt \
  --markers "$MARKERS" \
  --trait "Grain width related" \
  --top-k 500 \
  -o GW.upset.pdf \
  --summary-tsv GW.top500.source_percentage.tsv

# Check locus: 3-16733441
# TRAIT=GW_BLUP
# IMPORTANCE=${PREFIX}.${TRAIT}.position_importance/importance_ranking_${TRAIT}.tsv
# python3 gwas_ig_multi_plot_v3.py \
#     --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
#     --importance "$IMPORTANCE" \
#     -o "${PREFIX}.${TRAIT}.chr3-16733441.gwas_ig.pdf" \
#     --also-png \
#     --smooth 5 \
#     --ig-top-k 500 \
#     --qtn-annot Final_summary_347_QTNsites_geno_redefined.xlsx \
#     --highlight 3:16733441
yaml=705rice_conv_mha.aquila-snp.hpo.yaml

aquila_train_multi.py --config params/$yaml --n-folds 5 --n-seeds 10 -o ${yaml%.hpo.yaml}.best_fold_find

BEST_DS=$(python parse_best_fold.py ${yaml%.hpo.yaml}.best_fold_find/results_summary.tsv --print-only-path)

aquila_train_hpo.py --config params/$yaml -o ${yaml%.yaml} -dsf $BEST_DS > ${yaml%.yaml}.log 2>&1

BEST_TRIAL=$(awk -F': ' '/Best trial number/ {print $2}' ${yaml%.yaml}/optuna_summary.txt)

cp -rf ${yaml%.yaml}/trial_$BEST_TRIAL ${yaml%.yaml}.best_model
aquila_predict.py --model-dir ${yaml%.yaml}.best_model --vcf user.vcf.gz --output user.preds.tsv


PREFIX=705rice_conv_mha.aquila-snp.GYP_BLUP
VCF=705rice_0.03.full.all.impute.biallelic.vcf.gz
aquila_ig.py --model-dir $PREFIX.best_model \
    --vcf $VCF -o $PREFIX.ig

aquila_ig_interpretation.py -i $PREFIX.ig/ig_results.h5 -o $PREFIX.position_importance

python extract_attention.py --model-dir $PREFIX.best_model \
    --vcf 705rice_0.03.full.all.impute.biallelic.vcf.gz -o $PREFIX.attention --save-mean-only


TRAIT="GYP_BLUP"
python3 gwas_ig_plot.py --gwas 705rice_gwas_results/GYP_BLUP/gemma_lmm.assoc.txt --importance 705rice_conv_mha.aquila-snp.GYP_BLUP.position_importance/importance_ranking_GYP_BLUP.tsv -o GYP_BLUP.gwas.ig.png
python attention_gwas_importance_plot.py \
  --attention $PREFIX.attention/attention_attention_mean.h5 \
  --gwas 705rice_gwas_results/$TRAIT/gemma_lmm.assoc.txt \
  --importance $PREFIX.position_importance/importance_ranking_$TRAIT.tsv \
  --normalize none \
  -o plots/$TRAIT.gwas_attention_ig_comparison.png

##

PREFIX=705rice_conv_mha.aquila-snp
GWAS_DIR=705rice_gwas_results
IMP_DIR=$PREFIX.position_importance
OUT_DIR=plots
mkdir -p $OUT_DIR

# Automatically detect all trait directories
TRAITS=($(ls -d $GWAS_DIR/*/ 2>/dev/null | xargs -n1 basename))

# Number of parallel jobs
NJOBS=8

# Function to run one plot
run_one() {
    TRAIT=$1
    GWAS_FILE=$GWAS_DIR/$TRAIT/gemma_lmm.assoc.txt
    IMP_FILE=$IMP_DIR/importance_ranking_$TRAIT.tsv
    OUT_FILE=$OUT_DIR/${TRAIT}.gwas_attention_ig_comparison.png

    if [[ ! -f $GWAS_FILE ]]; then
        echo "[SKIP] $GWAS_FILE not found"
        return
    fi
    if [[ ! -f $IMP_FILE ]]; then
        echo "[SKIP] $IMP_FILE not found"
        return
    fi

    echo "[RUN] $TRAIT"
    python attention_gwas_importance_plot.py \
        --attention $PREFIX.attention/attention_attention_mean.h5 \
        --gwas $GWAS_FILE \
        --importance $IMP_FILE \
        --normalize none \
        -o $OUT_FILE
}

export -f run_one
export GWAS_DIR IMP_DIR PREFIX OUT_DIR

printf "%s\n" "${TRAITS[@]}" | xargs -n1 -P$NJOBS -I{} bash -c 'run_one "$@"' _ {}

echo "Done! All plots saved to $OUT_DIR/"




PREFIX=705rice_conv_mha.aquila-snp.GYP_BLUP
aquila_ig.py --model-dir $PREFIX.hpo/trial_30 --vcf $VCF -o $PREFIX.trial_30.ig
aquila_ig_interpretation.py -i $PREFIX.trial_30.ig/ig_results.h5 -o $PREFIX.trial_30.position_importance

TRAIT="GYP_BLUP"
python3 gwas_ig_plot.py --gwas 705rice_gwas_results/$TRAIT/gemma_lmm.assoc.txt \
    --importance $PREFIX.trial_30.position_importance/importance_ranking_$TRAIT.tsv -o $TRAIT.trial_30.gwas.ig.png


python extract_attention.py --model-dir $PREFIX.hpo/trial_30 \
    --vcf 705rice_0.03.full.all.impute.biallelic.vcf.gz -o $PREFIX.trial_30.attention --save-mean-only


python attention_gwas_importance_plot.py \
  --attention $PREFIX.trial_30.attention/attention_attention_mean.h5 \
  --gwas 705rice_gwas_results/$TRAIT/gemma_lmm.assoc.txt \
  --importance $PREFIX.trial_30.position_importance/importance_ranking_$TRAIT.tsv \
  --normalize none \
  -o plots/$TRAIT.trial_30.gwas_attention_ig_comparison.png
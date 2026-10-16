rsync -rlthP 6000:/data3/home/gulei/projects/GraphPan/Multi_Source_Marker_Panel_generation/655rice.wg_ld_r2_scan/runs/r2_0.035/panel_output/final_panel/655rice.final.imputed.panel.vcf.gz ./655rice.msmp_panel.imputed.snp_indel_sv.vcf.gz

bcftools annotate -x 'INFO,^FORMAT/GT' -Ou 655rice.msmp_panel.imputed.snp_indel_sv.vcf.gz \
| bcftools view \
| awk '
  /^##fileformat=/ || /^##contig=/ || /^##FORMAT=<ID=GT,/ { print; next }
  /^##ALT=<ID=/ && $0 !~ /;/ { print; next }
  /^##/ { next }
  { print }
' | bgzip -c > 655rice.msmp_panel.imputed.snp_indel_sv.clean.vcf.gz

conda activate aquila

# For Aquila-SNP
aquila_data_cv_production.py \
  --vcf 655rice.msmp_panel.imputed.snp_indel_sv.clean.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  -o 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
  --config 64hpo_budgets.bayesian.yaml \
  --fold 0 \
  --n-trials 500 \
  -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.fold0-hpo \
  --precision bf16 --live-metrics-log \
  --use-deterministic --overwrite

./aquila_export_locked_config.py \
  --summary results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.fold0-hpo/fold_0/summary.json \
  -o 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.locked.yaml

# Omit --fold to run one independent 500-trial Bayesian study for every
# prepared fold. Available GPUs are assigned different folds.
# aquila_train_cv_production.py \
#   --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
#   --config 64hpo_budgets.bayesian.yaml \
#   --n-trials 500 \
#   -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.all-fold-hpo \
#   --precision bf16 --use-deterministic --overwrite

# Legacy behavior: aggregate every fold for HPO, then refit on the full data.
# aquila_train_cv_production.py \
#   --mode cv-refit \
#   --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
#   --config 64hpo_budgets.bayesian.yaml \
#   -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.cv-refit \
#   --precision bf16 --use-deterministic --overwrite

for seed in $(seq 43 51); do
  aquila_train.py \
    --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
    --config 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.locked.yaml \
    --seed "$seed" \
    -o "results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.10folds/seeds/seed_${seed}" \
    --precision bf16 \
    --live-metrics-log \
    --use-deterministic
done

# rrBLUP uses fixed REML parameters. Fit one model on each of the 10
# train splits. 120_lines_test-set/main.sh averages those fold predictions.
python rrBLUP/rrblup_production.py \
  --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
  --config rrBLUP/configs/production.yaml \
  --traits GYP_BLUP \
  -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.GYP_BLUP.rrblup \
  --overwrite


# For Aquila-Vars
aquila_data_cv_production.py \
  --vcf 655rice.msmp_panel.imputed.snp_indel_sv.clean.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --encoding-type diploid_onehot \
  --folds 5 \
  --seed 42 \
  -o 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-vars.production.data \
  --overwrite
aquila_train_cv_production.py \
  --mode cv-refit \
  --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-vars.production.data \
  --config ../benchmark/aquila-vars/32hpo_budgets.aquila-vars.yaml \
  -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite

# For rrBLUP
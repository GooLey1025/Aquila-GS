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
  -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.10folds \
  --precision bf16 --live-metrics-log \
  --use-deterministic --overwrite

./aquila_export_locked_config.py \
  --summary results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.10folds/summary.json \
  -o 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.locked.yaml

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

# rrBLUP: fixed REML parameters, so fit the full reference set directly.
python rrBLUP/rrblup_production.py \
  --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.production.10folds.data \
  --config rrBLUP/configs/production.yaml \
  -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-snp.rrblup


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
  --data-dir 655rice.msmp_panel.imputed.snp_indel_sv.clean.aquila-vars.production.data \
  --config ../benchmark/aquila-vars/32hpo_budgets.aquila-vars.yaml \
  -o results/655rice.msmp_panel.imputed.snp_indel_sv.clean \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite

# For rrBLUP
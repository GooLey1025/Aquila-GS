rsync -rlthP 6000:/data3/home/gulei/projects/GraphPan/Multi_Source_Marker_Panel_generation/655rice.wg_ld_r2_scan/runs/r2_0.045/panel_output/final_panel/655rice.final.imputed.panel.vcf.gz .

bcftools annotate -x 'INFO,^FORMAT/GT' -Ou 655rice.final.imputed.panel.vcf.gz \
| bcftools view \
| awk '
  /^##fileformat=/ || /^##contig=/ || /^##FORMAT=<ID=GT,/ { print; next }
  /^##ALT=<ID=/ && $0 !~ /;/ { print; next }
  /^##/ { next }
  { print }
' | bgzip -c > 655rice.r2_0.045.panel.vcf.gz

conda activate aquila

# For Aquila-SNP
aquila_data_cv_production.py \
  --vcf 655rice.r2_0.045.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 5 \
  --seed 42 \
  -o 655rice.r2_0.045.aquila-snp.production.data \
  --overwrite
aquila_train_cv_production.py \
  --data-dir 655rice.r2_0.045.aquila-snp.production.data \
  --config ../benchmark/aquila-snp/32hpo_budgets.yaml \
  -o results/655rice.r2_0.045.aquila-snp \
  --precision bf16 --live-metrics-log \
  --use-deterministic --overwrite

# For Aquila-Vars
aquila_data_cv_production.py \
  --vcf 655rice.r2_0.045.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --encoding-type diploid_onehot \
  --folds 5 \
  --seed 42 \
  -o 655rice.r2_0.045.aquila-vars.production.data \
  --overwrite
aquila_train_cv_production.py \
  --data-dir 655rice.r2_0.045.aquila-vars.production.data \
  --config ../benchmark/aquila-vars/32hpo_budgets.aquila-vars.yaml \
  -o results/655rice.r2_0.045 \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite

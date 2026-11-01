# Other-cohort GS model train

Run from this directory after `conda activate aquila`. The fold map, data prep, and per-fold training match `README.md`. Each cohort below only changes the VCF, phenotype table, and Aquila-SNP config.

Generate one SampleID validation-fold map, then pass it to every `aquila_data_cv_production.py` call so multi-trait and single-trait runs keep the same fold for every shared SampleID. Single-trait missing phenotypes only drop those IDs from a fold. Per-fold training writes `fold_<k>/best_model.pt`.

## Rice1171

- VCF: `1171rice.final.imputed.panel.vcf.gz` (17,875 SNPs, 8,109 INDELs, 30,826 SVs)
- Phenotype: `1171rice.pheno`
- Aquila-SNP config: `aquila-snp.medium.hpo.yaml`
- Aquila-Vars config: `aquila-vars.hpo.yaml`

`1171rice.pheno` columns are raw traits, not the Rice655 BLUPs. The single-trait blocks are one example. Set `TRAIT` to any phenotype column and rerun the same commands.

### Shared 10-fold map

```bash
aquila_cv.py \
  --phenotype 1171rice.pheno \
  --outer-folds 10 \
  --seed 42 \
  --min-observed 10 \
  -o 1171rice.final.imputed.panel.10folds.nested_cv.json
```

### Per-fold Aquila-SNP: multi-trait

```bash
aquila_data_cv_production.py \
  --vcf 1171rice.final.imputed.panel.vcf.gz \
  --phenotype 1171rice.pheno \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping 1171rice.final.imputed.panel.10folds.nested_cv.json \
  -o 1171rice.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 1171rice.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.data \
  --config aquila-snp.medium.hpo.yaml \
  -o 1171rice.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

### Per-fold Aquila-SNP: one trait

```bash
TRAIT=Heading_date

aquila_data_cv_production.py \
  --vcf 1171rice.final.imputed.panel.vcf.gz \
  --phenotype 1171rice.pheno \
  --traits ${TRAIT} \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping 1171rice.final.imputed.panel.10folds.nested_cv.json \
  -o 1171rice.final.imputed.panel.aquila-snp.${TRAIT}.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 1171rice.final.imputed.panel.aquila-snp.${TRAIT}.per-fold.10folds.data \
  --config aquila-snp.medium.hpo.yaml \
  -o 1171rice.final.imputed.panel.aquila-snp.${TRAIT}.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

### Per-fold Aquila-Vars: multi-trait

Leave out `--variant-type` and `--id-prefix` so SNP, INDEL, and SV stay as separate branches.

```bash
aquila_data_cv_production.py \
  --vcf 1171rice.final.imputed.panel.vcf.gz \
  --phenotype 1171rice.pheno \
  --encoding-type diploid_onehot \
  --folds 10 \
  --seed 42 \
  --cv-mapping 1171rice.final.imputed.panel.10folds.nested_cv.json \
  -o 1171rice.final.imputed.panel.aquila-vars.multi_trait.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 1171rice.final.imputed.panel.aquila-vars.multi_trait.per-fold.10folds.data \
  --config aquila-vars.hpo.yaml \
  -o 1171rice.final.imputed.panel.aquila-vars.multi_trait.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

### Per-fold Aquila-Vars: one trait

```bash
TRAIT=Heading_date

aquila_data_cv_production.py \
  --vcf 1171rice.final.imputed.panel.vcf.gz \
  --phenotype 1171rice.pheno \
  --traits ${TRAIT} \
  --encoding-type diploid_onehot \
  --folds 10 \
  --seed 42 \
  --cv-mapping 1171rice.final.imputed.panel.10folds.nested_cv.json \
  -o 1171rice.final.imputed.panel.aquila-vars.${TRAIT}.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 1171rice.final.imputed.panel.aquila-vars.${TRAIT}.per-fold.10folds.data \
  --config aquila-vars.hpo.yaml \
  -o 1171rice.final.imputed.panel.aquila-vars.${TRAIT}.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Rice3305

- VCF: `Rice3305.vcf.gz` (13,282 SNP records)
- Phenotype: `Rice3305.pheno.tsv` (3,305 samples, filtered from `3606rice_Phenotype.tsv`)
- Aquila-SNP config: `aquila-snp.medium.hpo.yaml`

This panel has no INDEL or SV records, so only Aquila-SNP is trained. Phenotype columns are environment-specific. The single-trait blocks are one example. Set `TRAIT` to any column in `Rice3305.pheno.tsv` and rerun the same commands on the shared fold map.

### Shared 10-fold map

```bash
aquila_cv.py \
  --phenotype Rice3305.pheno.tsv \
  --outer-folds 10 \
  --seed 42 \
  --min-observed 10 \
  -o Rice3305.10folds.nested_cv.json
```

### Per-fold Aquila-SNP: multi-trait

```bash
aquila_data_cv_production.py \
  --vcf Rice3305.vcf.gz \
  --phenotype Rice3305.pheno.tsv \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping Rice3305.10folds.nested_cv.json \
  -o Rice3305.aquila-snp.multi_trait.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir Rice3305.aquila-snp.multi_trait.per-fold.10folds.data \
  --config aquila-snp.medium.hpo.yaml \
  -o Rice3305.aquila-snp.multi_trait.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

### Per-fold Aquila-SNP: one trait

```bash
TRAIT=Headingdate_2023

aquila_data_cv_production.py \
  --vcf Rice3305.vcf.gz \
  --phenotype Rice3305.pheno.tsv \
  --traits ${TRAIT} \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping Rice3305.10folds.nested_cv.json \
  -o Rice3305.aquila-snp.${TRAIT}.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir Rice3305.aquila-snp.${TRAIT}.per-fold.10folds.data \
  --config aquila-snp.medium.hpo.yaml \
  -o Rice3305.aquila-snp.${TRAIT}.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

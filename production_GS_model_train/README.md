# Production GS model train

Run from this directory after `conda activate aquila`.

Generate one SampleID validation-fold map on the phenotype table (VCF ∩ phenotype universe after data prep), then pass that map to every `aquila_data_cv_production.py` call so multi-trait and single-trait Aquila runs keep the same fold for every shared SampleID; single-trait missing phenotypes only drop those IDs from a fold and do not reshuffle the rest. Per-fold Aquila training writes `fold_<k>/best_model.pt`. rrBLUP reads the same `cv/fold_*/train_idx.npy` from the prepared `--data-dir`, so point it at the matching `*.data` directory.

## Shared 10-fold map

```bash
aquila_cv.py \
  --phenotype ../benchmark/Rice655.pheno \
  --outer-folds 10 \
  --seed 42 \
  --min-observed 10 \
  -o 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json
```

## Per-fold Aquila-SNP: multi-trait

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.data \
  --overwrite
```

Pass `--mode per-fold` so each prepared train/valid split runs its own HPO and writes `fold_<k>/best_model.pt`. The default `--mode cv-refit` instead shares one hyperparameter search across folds and then refits a full-data `best_model.pt`. We use the 10-fold ensemble here because averaging predictions across fold models is more robust than relying on a single full-data refit.

```bash
aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.data \
  --config aquila-snp.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-SNP: GYP_BLUP

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --traits GYP_BLUP \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.GYP_BLUP.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.GYP_BLUP.per-fold.10folds.data \
  --config aquila-snp.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.GYP_BLUP.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-SNP: PH_BLUP

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --traits PH_BLUP \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.PH_BLUP.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.PH_BLUP.per-fold.10folds.data \
  --config aquila-snp.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.PH_BLUP.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-SNP: HD_BLUP

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --traits HD_BLUP \
  --encoding-type diploid_onehot \
  --variant-type snp \
  --id-prefix SNP- \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.HD_BLUP.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.HD_BLUP.per-fold.10folds.data \
  --config aquila-snp.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.HD_BLUP.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-Vars: multi-trait

Aquila-Vars uses the same fold map and per-fold training as Aquila-SNP. Leave out `--variant-type` and `--id-prefix` so SNP, INDEL, and SV stay as separate branches.

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --encoding-type diploid_onehot \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.multi_trait.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.multi_trait.per-fold.10folds.data \
  --config aquila-vars.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.multi_trait.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-Vars: GYP_BLUP

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --traits GYP_BLUP \
  --encoding-type diploid_onehot \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.GYP_BLUP.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.GYP_BLUP.per-fold.10folds.data \
  --config aquila-vars.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.GYP_BLUP.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-Vars: HD_BLUP

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --traits HD_BLUP \
  --encoding-type diploid_onehot \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.HD_BLUP.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.HD_BLUP.per-fold.10folds.data \
  --config aquila-vars.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.HD_BLUP.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## Per-fold Aquila-Vars: PH_BLUP

```bash
aquila_data_cv_production.py \
  --vcf 655rice.production.more_sv_markers.final.imputed.panel.vcf.gz \
  --phenotype ../benchmark/Rice655.pheno \
  --traits PH_BLUP \
  --encoding-type diploid_onehot \
  --folds 10 \
  --seed 42 \
  --cv-mapping 655rice.production.more_sv_markers.final.imputed.panel.10folds.nested_cv.json \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.PH_BLUP.per-fold.10folds.data \
  --overwrite

aquila_train_cv_production.py \
  --mode per-fold \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.PH_BLUP.per-fold.10folds.data \
  --config aquila-vars.hpo.yaml \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-vars.PH_BLUP.per-fold.10folds.hpo \
  --precision bf16 \
  --live-metrics-log \
  --use-deterministic \
  --overwrite
```

## rrBLUP on the same splits

Use the same prepared `*.data` directory as the Aquila run you want to compare against.

```bash
python rrBLUP/rrblup_production.py \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.GYP_BLUP.per-fold.10folds.data \
  --config rrBLUP/configs/production.yaml \
  --traits GYP_BLUP \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.GYP_BLUP.per-fold.10folds.rrblup \
  --overwrite

python rrBLUP/rrblup_production.py \
  --data-dir 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.PH_BLUP.per-fold.10folds.data \
  --config rrBLUP/configs/production.yaml \
  --traits PH_BLUP \
  -o 655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.PH_BLUP.per-fold.10folds.rrblup \
  --overwrite
```

## Evaluation on the Rice300 hold-out panel

Each command below handles one trait at a time. `aquila_predict_ensemble.py`
accepts exactly one production model directory and averages that directory's
`fold_<k>/best_model.pt` predictions. The rrBLUP prediction and correlation
commands use the same single-trait organization. Run from
`300_lines_test-set/` in the `aquila` environment.

```bash
cd 300_lines_test-set

# 1. Aquila-SNP: predict GYP_BLUP with one model directory
aquila_predict_ensemble.py \
  --model-dir ../655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.GYP_BLUP.per-fold.10folds.hpo \
  --vcf Rice300.Rice655-ref.snp.indel.sv.impute.biallelic.vcf.gz \
  --encoding-type diploid_onehot \
  --id-prefix SNP- \
  -o predictions/aquila-snp.GYP_BLUP.10fold-mean.300rice.tsv

# 2. Evaluate the Aquila-SNP GYP_BLUP predictions
TRAIT=GYP_BLUP
python correlation_analysis.py \
  --preds predictions/aquila-snp.${TRAIT}.10fold-mean.300rice.tsv \
  --blup 3605rice_Phenotype_BLUP.tsv \
  --meta 300lines_map.txt \
  --trait ${TRAIT} \
  --model-name Aquila-SNP \
  --output-dir figures/aquila-snp.${TRAIT}.10fold-mean

# 3. rrBLUP: predict GYP_BLUP with its fold models
python predict_rrblup_vcf.py \
  --vcf Rice300.Rice655-ref.snp.indel.sv.impute.biallelic.vcf.gz \
  --model-dir ../655rice.production.more_sv_markers.final.imputed.panel.rrblup \
  --traits ${TRAIT} \
  --average-folds \
  --id-prefix SNP- \
  -o predictions/rrblup.${TRAIT}.10fold-mean.300rice.tsv

# 4. Evaluate the rrBLUP GYP_BLUP predictions
python correlation_analysis.py \
  --preds predictions/rrblup.${TRAIT}.10fold-mean.300rice.tsv \
  --blup 3605rice_Phenotype_BLUP.tsv \
  --meta 300lines_map.txt \
  --trait ${TRAIT} \
  --model-name rrBLUP \
  --output-dir figures/rrblup.${TRAIT}.10fold-mean
```

Replace `GYP_BLUP` with `HD_BLUP` or `PH_BLUP` to evaluate another trait. Add `--keep-fold-predictions predictions/aquila-snp.GYP_BLUP.folds` to the Aquila command when the individual fold tables are needed.

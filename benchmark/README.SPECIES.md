# Species nested-CV benchmark

This is the multi-species run of the nested cross-validation protocol.

## Protocol

Each cohort uses one fixed nested split, and every model uses that split.

1. Outer 10-fold, test:train = 1:9, with 4 inner folds inside each outer-training set. `aquila_cv.py` writes this split.
2. Inner folds are used only to compare candidate configurations. The selected configuration is refit on the complete outer-training partition and scored on the outer test.
3. Methods with one fixed configuration, currently rrBLUP and BayesCpi, skip inner-fold fitting and train directly on the complete outer-training partition.
4. Phenotype preprocessing is fit on the training side of each split. For an inner fold it is fit on `inner_train` and applied to `inner_valid`. For final evaluation it is fit again on `outer_train` and applied to `outer_train` and `outer_test`.

## Cohorts

Set one cohort, then run the commands below from `benchmark/`.

```sh
COHORT=Maize1404_pheno_10folds
PHENO_FILE=species_data/Maize1404/benchmark.pheno
VCF_FILE=species_data/Maize1404/Maize1404.coding.ld.vcf.gz

# COHORT=Soybean975_blup_pheno_10folds
# PHENO_FILE=species_data/Soybean975/benchmark.blup.pheno
# VCF_FILE=species_data/Soybean975/Soybean975.coding.ld.vcf.gz

# COHORT=wheat994_blup_pheno_10folds
# PHENO_FILE=species_data/wheat994/benchmark.blup.pheno
# VCF_FILE=species_data/wheat994/wheat994.coding.ld.vcf.gz

# COHORT=Rice1171_pheno_10folds
# PHENO_FILE=species_data/Rice1171/1171rice.pheno
# VCF_FILE=species_data/Rice1171/Rice1171.coding.ld.vcf.gz

# COHORT=Rice532
# PHENO_FILE=species_data/Rice532/532rice.pheno
# VCF_FILE=species_data/Rice532/Rice532.coding.ld.vcf.gz

COHORT=Rice655_Only_WG_LD
PHENO_FILE=Rice655.blup.pheno
VCF_FILE=species_data/Rice655.WG_0.035_LD/655rice.WG_0.035_LD.snp_indel_sv.vcf.gz

```

`$COHORT.cv.data` is built with `--id-prefix SNP-` for every cohort except Rice1171. Rice1171 omits that prefix and does not build an Aquila-Vars directory. The other cohorts prepare both `$COHORT.cv.data` and `$COHORT.vars.cv.data`.

## Prepare the shared splits

Run from `benchmark/` in the `aquila` environment.

```sh
export PATH="$CONDA_PREFIX/bin:$PATH"
conda activate aquila

aquila_cv.py --phenotype $PHENO_FILE \
  -o $COHORT.nested_cv.json \
  --outer-folds 10 --inner-folds 4 --seed 42 --min-observed 20

if [[ "$COHORT" == *Rice1171* ]]; then
  SNP_PREFIX=(--id-prefix "SNP-")
else
  SNP_PREFIX=()
fi
aquila_data_cv.py \
  --vcf $VCF_FILE \
  --phenotype $PHENO_FILE \
  --encoding-type diploid_onehot \
  --variant-type snp \
  "${SNP_PREFIX[@]}" \
  --fold-mapping $COHORT.nested_cv.json \
  -o $COHORT.cv.data \
  --save-raw-genotype \
  --overwrite
if [[ "$COHORT" != *Rice1171* ]]; then
  aquila_data_cv.py \
    --vcf $VCF_FILE \
    --phenotype $PHENO_FILE \
    --encoding-type diploid_onehot \
    --fold-mapping $COHORT.nested_cv.json \
    -o $COHORT.vars.cv.data \
    --save-raw-genotype \
    --overwrite
fi
```

## Comparison models

Each block below is run from `benchmark/` after the data step. Outputs land in that model's `results/$COHORT`.

```sh
cd croparnet
/usr/bin/time -v -o $COHORT.time.txt python src_benchmark/adapter.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT \
  --jobs-per-gpu 4 --overwrite

cd ../cropformer
/usr/bin/time -v -o $COHORT.time.txt python src_benchmark/adapter.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT \
  --jobs-per-gpu 4 --overwrite

cd ../xgboost
/usr/bin/time -v -o $COHORT.time.txt python xgboost_train_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/xgboost_nested_cv.yaml \
  -o results/$COHORT \
  --n-jobs 4 --overwrite

cd ../bayescpi
/usr/bin/time -v -o $COHORT.time.txt python bayescpi_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT --overwrite

cd ../rrBLUP
/usr/bin/time -v -o $COHORT.time.txt python rrblup_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT --overwrite

cd ../Lasso
python lasso_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT --overwrite

cd ../ElasticNet
python elasticnet_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT --overwrite

cd ../CLCNet
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
/usr/bin/time -v -o "$COHORT.time.txt" \
python CLCNet_train_cv.py \
  --data-dir "../$COHORT.cv.data" \
  --config configs/CLCNet_nested_cv.yaml \
  --jobs-per-gpu 1 \
  -o "results/$COHORT" \
  --overwrite

cd ../MENET
/usr/bin/time -v -o $COHORT.time.txt python MENET_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/MeNet_nested_cv.yaml \
  --jobs-per-gpu 2 \
  -o results/$COHORT --overwrite

cd ../DEM
/usr/bin/time -v -o $COHORT.DEM-SNP.time.txt python DEM_train_benchmark.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/DEM-SNP_nested_cv.yaml \
  --output-dir results/DEM-SNP/$COHORT \
  --jobs-per-gpu 2 --overwrite

conda activate aquila_dnawhisperer
cd ../Whisperer_of_DNA
/usr/bin/time -v -o $COHORT.time.txt python Whisperer_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/Whisperer_nested_cv.yaml \
  --output-dir results/$COHORT \
  --jobs-per-gpu 2 --overwrite

conda activate aquila
cd ../BNNs
/usr/bin/time -v -o $COHORT.time.txt python BNNs_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/BNNs_nested_cv.yaml \
  --output-dir results/$COHORT \
  --jobs-per-gpu 2 --overwrite
```

## Aquila

Aquila-SNP multi-trait:

```sh
cd ../aquila-snp
aquila_train_cv.py --data-dir ../$COHORT.cv.data \
  --config 32hpo_budgets.yaml \
  -o results/$COHORT --live-metrics-log --overwrite
```

Aquila-Vars, when `$COHORT.vars.cv.data` exists:

```sh
cd ../aquila-vars
aquila_train_cv.py --data-dir ../$COHORT.vars.cv.data \
  --config aquila-vars.hpo.yaml \
  -o results/$COHORT --live-metrics-log --use-deterministic --overwrite
```

## (Discarded)

Single-trait Aquila-SNP repeats the nested-CV preparation and training once per phenotype column:

```sh
cd ../aquila-snp
IFS=$'\t' read -r -a PHENO_COLUMNS < "../$PHENO_FILE"
for TRAIT in "${PHENO_COLUMNS[@]:1}"; do
  aquila_data_cv.py \
    --vcf "../$VCF_FILE" \
    --phenotype "../$PHENO_FILE" \
    --traits "$TRAIT" \
    --encoding-type diploid_onehot \
    --variant-type snp \
    "${SNP_PREFIX[@]}" \
    --fold-mapping "../$COHORT.nested_cv.json" \
    -o "../$COHORT.single_task.cv.data/$TRAIT" \
    --overwrite
  aquila_train_cv.py \
    --data-dir "../$COHORT.single_task.cv.data/$TRAIT" \
    --config 32hpo_budgets.yaml \
    -o "results/$COHORT.single_task/$TRAIT" \
    --live-metrics-log \
    --overwrite
done
```

`SNP_PREFIX` is `--id-prefix SNP-` by default, and empty for Rice1171.

## Summary

From `benchmark/`:

```sh
python summary_and_plot_benchmark_model.py --benchmark-dir . "$COHORT"
```

The summary reads `results/$COHORT` under each model directory.

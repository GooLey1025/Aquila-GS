# Benchmark

All benchmarks were conducted using the versions of the compared methods that were available at the time (2026-07) of our experiments.

## Prerequisites

All integrated benchmark adapters are run from the existing Aquila environment. Install the additional Python and R dependencies with:

```sh
cd ~/projects/Aquila-GS/benchmark
conda activate aquila
conda env update -n aquila -f environment.yml
```

The environment file adds the dependencies not supplied by Aquila itself, including XGBoost, the Whisperer of DNA runtime packages, R 4.3, `jsonlite`, `glmnet`, `rrBLUP`, and CLCNet's optional LightGBM selector. Integrated models are tested in this updated `aquila` environment; their upstream Conda environments and pinned PyTorch/CUDA stacks are not used. Consequently, for benchmarking purposes, it is not necessary to clone all original repositories, as the required scripts have been integrated or adapted into this project.

The CRAN `hibayes` package used by BayesCpi is not available from the configured Conda channels. Install it once into the R library inside the activated Aquila environment:

```sh
# You need to download r-library in the aquila env. CONDA_PREFIX is the path to the conda root directory. E.g. /data4/gulei/anaconda3
# A proxy may be required to download this package.
conda activate aquila
$CONDA_PREFIX/bin/Rscript -e 'install.packages("hibayes", repos="https://cloud.r-project.org")'
```

## Data Prepare

To ensure a fair comparison, every method uses the same 10-fold holdout. This is not nested CV: there is no inner cross-validation loop.

1. Split all samples into 10 folds. For fold k, the held-out fold is test, and the other nine folds together are outer-train.
2. Split that outer-train once into train and valid. The default `--valid-fraction 0.2` puts 20% of outer-train in valid and 80% in train (4:1). Pass another value between 0 and 1 to change the ratio.

The mapping file is named `holdout_cv.json`. Hyperparameter models run Bayesian search on the validation set, keep the early-stopping checkpoint, and score the outer test once. They do not refit on the full outer-train. The default budget is 200 trials. Whisperer of DNA and BNNs use 50 trials. rrBLUP and BayesCpi have no searched hyperparameters: they train only on the train split and do not use the validation samples. All models are evaluated on the same held-out outer-test samples.

For Aquila, missing phenotypic observations are handled natively through a masked multi-task learning strategy. Specifically, missing trait values are excluded from the loss calculation while the remaining observed traits continue to contribute to model optimization, allowing the model to exploit correlations among multiple traits without requiring phenotype imputation. For other methods that do not support missing phenotypes, missing observations were handled according to their model assumptions. Single-trait models were trained using only individuals with available phenotypic records for the target trait. Importantly, these models still followed the same predefined cross-validation partitions as Aquila, ensuring that differences in performance reflect model behavior rather than differences in data splitting. This evaluation framework assesses genomic prediction performance under realistic incomplete phenotype conditions encountered in practical breeding programs.

### Generate fold mapping:

```sh
# GSTP008.pheno downloaded from CropGS-hub (https://iagr.genomics.cn/CropGS/#/Datasets)
wget https://iagr.genomics.cn/static/gstool/data/GSTP008/population/GSTP008.pheno
# However, the 705rice have a total of 50 low-depth duplicated samples, filtering required.

aquila_cv.py --phenotype Rice655.pheno -o 655rice_holdout_cv_10folds.json --outer-folds 10 --valid-fraction 0.2 --seed 42
```

Each record in this JSON is one outer fold: `test` is the held-out fold, `outer_train` is the other nine folds, and `train` / `valid` are the single split of that outer-train. It was used throughout the pipeline, including [GWAS lead-variant selection](https://github.com/GooLey1025/Multi_Source_Marker_Panel_generation), to avoid information leakage. GWAS discovery uses `outer_train` and excludes `test`. `aquila_data_cv_benchmark.py` uses the written train, validation, and test IDs. An older nested-CV JSON can still be read: its outer tests are kept, and the inner folds are replaced by one train/validation split.

When `--save-raw-genotype` is enabled, sample-subset VCF files are written under each prepared directory's `raw_genotype/`. Every outer fold contains `test.vcf.gz` and a full outer-train `train.vcf.gz`. The protocol training genotypes are `raw_genotype/outer_fold_k/inner_fold_0/train.vcf.gz` (80%) and `valid.vcf.gz` (20%). Do not train on the full outer-train VCF. These files preserve the source variants and genotype fields, but include only the samples assigned to that split.

Phenotypes are preprocessed once during data preparation and the resulting fold-specific standardized targets are reused by all downstream benchmarks. This provides a consistent phenotype input and prevents each method from introducing differences through independent preprocessing.

To avoid information leakage, preprocessing parameters are estimated once from the 80% HPO train split:

- Trait skewness is calculated using the observed phenotypes in that train split. A `log1p` transformation is applied when the absolute skewness exceeds the configured threshold, after which the trait is standardized using the training-set mean and standard deviation. The same fitted transformation is applied to the validation split and the outer test. There is no second fit on the full outer-train.
- Missing phenotype values are excluded when fitting preprocessing parameters and remain masked in the prepared targets.

Each outer fold stores `Y_train_processed.pt`, `Y_valid_processed.pt`, `Y_test_processed.pt`, and `preprocessing.json` at `cv/outer_fold_k/`. Together with the outer-fold mapping and fold-specific VCF files, these artifacts form the common data inputs used for all benchmark models.

Single-trait benchmark models that do not natively support missing phenotypes, such as MENET, discard samples with an unobserved target separately within each training, validation, and test partition. The `-999` missing-value sentinel is never passed to their loss functions or evaluation metrics. The remaining samples retain the same outer-fold assignments and the train-split phenotype transformation used by Aquila.

For regression benchmarks, both Aquila-GS and the integrated comparison models report Pearson r, R², MSE, RMSE, and MAE on the available test observations. Fold outputs include metrics on both the standardized phenotype scale and the inverse-transformed original scale. The comparison summary uses within-trait Pearson: for each phenotype, Pearson correlation is calculated across its held-out samples and then averaged across outer folds. This is distinct from Aquila's optional within-accession Pearson, which correlates traits within a sample and is undefined for independent single-trait models such as DEM-SNP and DEM-Vars.

`summary_and_plot_benchmark_model.py` reads this per-trait Pearson contract from all benchmark models, including Aquila-SNP, Aquila-Vars, DEM-SNP, and DEM-Vars. It supports both complete cohort result directories and the `fold_0` to `fold_9` result shards produced for fold-specific GWAS data. Its long-form output labels the metric as `within_trait_pearson`.

## Benchmark methodology

See the [Benchmark Fairness Checklist](BENCHMARK_FAIRNESS.md) for the partitioning, leakage prevention, preprocessing, model-selection, evaluation, and reporting requirements.

### SNP input and feature-selection policy

All models in the main benchmark use the same fold-specific SNP panels whenever their architectures permit, so differences in performance are not caused by independently selected input markers. Model-specific feature selection was disabled when it was an optional preprocessing step rather than an essential part of the method.

Three exceptions require explicit treatment. BNNs retains its internal Lasso-based feature-selection module because it is a defining component of the proposed architecture; consequently, its effective SNP subset can differ from those used by other models, although its nested-CV performance was not competitive in this benchmark. Cropformer originally applies mutual-information-coefficient (MIC) filtering with a default maximum of 10,000 SNPs; this optional filtering was disabled so that Cropformer receives the same SNP panel as the other models. DEM originally uses random-forest SNP selection. Removing this stage would expand each sample to approximately `number_of_SNPs × 10` genotype-class features, producing an impractically large model input and causing out-of-memory errors on the RTX 4090 benchmark server. DEM was therefore excluded from the main horizontal comparison and evaluated separately against Aquila-SNP using the same marker set selected by DEM, allowing the architectural capabilities of DEM and Aquila-SNP to be compared under matched markers.

## VarsGT reference panel

The Rice655 SNP, INDEL, and SV resources consumed by VarsGT are rebuilt with [`breeding_application/VarsGT/main.make_ref.sh`](../breeding_application/VarsGT/main.make_ref.sh). The script installs a new genotyping panel into an existing VarsGT index directory and leaves the Giraffe graph files unchanged. Inputs, environment variables, installed filenames, and the validation checks are described in [Rebuild the Rice655 VarsGT reference panel](../breeding_application/VarsGT/docs/make_ref.md).

## Reproduce of 655rice dataset

### [Aquila](https://github.com/GooLey1025/Aquila-GS)

GWAS lead variants are selected independently from the training samples of each outer fold, resulting in a different marker panel for every fold. Run the following data preparation commands from the `benchmark` root directory.

#### Prepare Aquila-SNP data

```sh
for FOLD in {0..9}; do
  aquila_data_cv_benchmark.py \
    --vcf "./species_data/Rice655-current_10folds_GWAS/outer_fold_${FOLD}/final_panel/marker_panel.outer_fold_${FOLD}.final.imputed.panel.vcf.gz" \
    --phenotype Rice655.pheno \
    --encoding-type diploid_onehot \
    --variant-type snp \
    --id-prefix "SNP-" \
    --fold-mapping 655rice_holdout_cv_10folds.json \
    --fold-specific-gwas-fold "$FOLD" \
    -o "Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --save-raw-genotype \
    --overwrite
done
```

#### Prepare Aquila-Vars data

Aquila-Vars uses separate SNP, INDEL, and SV branches, so its prepared data are generated without forcing SNP-only encoding or applying an ID-prefix filter:

```sh
for FOLD in {0..9}; do
  aquila_data_cv_benchmark.py \
    --vcf "./species_data/Rice655-current_10folds_GWAS/outer_fold_${FOLD}/final_panel/marker_panel.outer_fold_${FOLD}.final.imputed.panel.vcf.gz" \
    --phenotype Rice655.pheno \
    --encoding-type diploid_onehot \
    --fold-mapping 655rice_holdout_cv_10folds.json \
    --fold-specific-gwas-fold "$FOLD" \
    -o "Rice655.current_fold_GWAS.vars.cv.data/fold_${FOLD}" \
    --save-raw-genotype \
    --overwrite
done
```

Each output directory records its bound outer fold in `metadata.json`. Training without `--folds` automatically selects that fold; requesting any other fold is rejected. For datasets that use one shared genotype panel across all folds, omit `--fold-specific-gwas-fold`.

#### Train Aquila-SNP

```sh
cd aquila-snp

for FOLD in {0..9}; do
  aquila_train_cv_benchmark.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config  200hpo_bayesian.aquila-snp.yaml \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --live-metrics-log \
    --use-deterministic \
    --overwrite
done

cd ..
```

#### Train Aquila-Vars

```sh
cd aquila-vars

for FOLD in {0..9}; do
  aquila_train_cv_benchmark.py \
    --data-dir "../Rice655.current_fold_GWAS.vars.cv.data/fold_${FOLD}" \
    --config 200hpo_bayesian.aquila-vars.yaml \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --live-metrics-log \
    --use-deterministic \
    --overwrite
done

cd ..
```

### [CropARNet](https://github.com/Zhoushuchang-lab/CropARNet)

Some scripts from CropARNet were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `b9996564d0f021d2d24781935abb04a166c0342e`.

```sh
cd croparnet

for FOLD in {0..9}; do
  python src_benchmark/adapter.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

All benchmark runners in this section read `fold_specific_gwas` from `metadata.json`. When `--outer-folds` is omitted for fold-specific data, they automatically train only the bound outer fold; requesting another fold or multiple folds is rejected. Legacy prepared-data directories without this metadata retain the original all-fold behavior.

### [Cropformer](https://github.com/jiekesen/Cropformer.git)

Some scripts from Cropformer were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `e0a77cf699b034d956b249f41b2e1f357f486f06`.

The upstream pipeline uses MIC-based SNP filtering and limits the selected set to 10,000 SNPs by default. This filtering was disabled in the integrated benchmark so that Cropformer uses the same fold-specific SNP panel as the other models.

```sh
cd cropformer
for FOLD in {0..9}; do
  python src_benchmark/adapter.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

### XGBoost

```sh
cd xgboost
for FOLD in {0..9}; do
  python xgboost_train_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/xgboost_holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --n-jobs 32 \
    --overwrite
done
```

### BayesCpi

BayesCpi uses the single fixed configuration in `bayescpi/configs/holdout_cv.yaml`. It trains only on the 80% HPO train split and does not use the validation samples. `niter`, `nburn`, and `thin` control MCMC sampling accuracy, convergence, and runtime; they are fixed before evaluation rather than selected using outer-test performance. Convergence should be checked through MCMC diagnostics when changing these values. The historical script name is retained for command-line compatibility.

```sh
cd bayescpi
for FOLD in {0..9}; do
  python bayescpi_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --overwrite
done
```

### rrBLUP

rrBLUP uses the single fixed `REML` configuration in `rrBLUP/configs/holdout_cv.yaml`. It trains only on the 80% HPO train split and does not use the validation samples. Variance components and marker effects are estimated from that training partition by `mixed.solve`; they are fitted model parameters rather than searched hyperparameters. The historical script name is retained for command-line compatibility.

```sh
cd rrBLUP
for FOLD in {0..9}; do
  python rrblup_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --overwrite
done
```

### Lasso

```sh
cd Lasso
for FOLD in {0..9}; do
  python lasso_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --overwrite
done
```

### ElasticNet

```sh
cd ElasticNet
for FOLD in {0..9}; do
  python elasticnet_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/holdout_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --overwrite
done
```

### [CLCNet](https://github.com/SuppurNewer/CLCNet)

Some scripts from CLCNet were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `01d7792c97dc05f8a54afbfb2f62427607f60aad`.

```sh
cd CLCNet
for FOLD in {0..9}; do
  python CLCNet_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/CLCNet_holdout_cv.yaml \
    --outer-folds "$FOLD" \
    --jobs-per-gpu 2 \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --overwrite
done
```

### [MeNet](https://github.com/ganlab/MENET)

Some scripts from MENET were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `06456e4542ab26716c4db8dd8f17517aa5155ff4`.

```sh
cd MENET
for FOLD in {0..9}; do
  python MENET_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/MeNet_holdout_cv.yaml \
    --outer-folds "$FOLD" \
    --jobs-per-gpu 2 \
    -o "results/Rice655_10folds/fold_${FOLD}" \
    --overwrite
done
```

The benchmark adaptation preserves MENET's two-stage architecture. On holdout data, both stages use Bayesian search (200 trials), early stopping, and the winning checkpoints for the single outer-test score. See the [detailed MENET benchmark adaptation](docs/MENET_benchmark_adaptation.md).

### [DNAwhisper](https://github.com/Marxin1992/Whisperer_of_DNA)

Some scripts from Whisperer of DNA were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `7a657cc00d44263c4b1085d3991ecc0e935c14e5`.

```sh
cd Whisperer_of_DNA
conda activate aquila_dnawhisperer
for FOLD in {0..9}; do
  python Whisperer_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/Whisperer_holdout_cv.yaml \
    --outer-folds "$FOLD" \
    --output-dir "results/Rice655_10folds/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

### [Bayesian Neural Networks](https://github.com/GSBreeder/BNNs)

Some scripts from BNNs were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `6a8a82fd68625443a1c6dbcef19b428d031fa18f`.

BNNs retains its internal Lasso-based feature-selection module because it is treated as part of the model's architecture rather than external preprocessing. Lasso is fit on the 80% train split only. BNNs uses 50 Bayesian trials, keeps the early-stopping checkpoint, and scores the outer test once. Its effective SNP subset can therefore differ from the shared benchmark panel.

```sh
cd BNNs
for FOLD in {0..9}; do
  python BNNs_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/BNNs_holdout_cv.yaml \
    --outer-folds "$FOLD" \
    --output-dir "results/Rice655_10folds/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

All benchmark runners in this section enforce the same prepared-data fold binding. For directories with enabled `fold_specific_gwas` metadata, omitting `--outer-folds` selects only the bound fold, while explicitly requesting another fold or multiple folds is rejected. Prepared-data directories without this metadata retain the original all-fold default.

### Specific case: DEM vs Aquila

#### [DEM](https://github.com/cma2015/DEM/)

Some scripts from DEM were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `86de718f950d0ecc5554ff1916e2d59f51a33ce8`.

DEM's random-forest (RF) marker selection is retained because removing it produces approximately `number_of_SNPs × 10` genotype-class features per sample and exceeded the available memory on the RTX 4090 benchmark server. DEM is not included in the main horizontal comparison. Instead, `GYP_BLUP`, `HD_BLUP`, and `PH_BLUP` are evaluated as independent single-trait tasks. DEM-SNP and Aquila-SNP share the 1,000 SNP markers selected by the DEM-SNP RF, while DEM-Vars and Aquila-Vars share up to 1,000 markers independently selected for each SNP, INDEL, and SV branch by the DEM-Vars RF.

Marker selection is fit on the 80% HPO train split only. The winning early-stopping checkpoint is applied to the outer test; the random forest is not refit on validation samples or on the full outer-train. Aquila uses the same selected marker identities but retains its own genotype encodings. The complete two-stage workflow is provided in `DEM/run_dem_rf_matched_comparison.sh`.

```sh
cd DEM
python DEM_train_benchmark.py \
  --data-dir ../Maize1404.snp.cv.data \
  --config configs/DEM-SNP_holdout_cv.yaml \
  --output-dir results/DEM-SNP/Maize1404

python DEM_train_benchmark.py \
  --data-dir ../Maize1404.vars.cv.data \
  --config configs/DEM-Vars_holdout_cv.yaml \
  --output-dir results/DEM-Vars/Maize1404
```

For the matched-marker comparison on Rice655:

```sh
cd DEM
bash run_dem_rf_matched_comparison.sh
```

DEM-SNP and DEM-Vars train independent single-output models for each selected trait and outer fold. Missing phenotypes are removed only within the assigned train, validation, or test split, so predefined fold membership is unchanged, and training uses ordinary MSE. DEM-SNP uses the original ten-channel DEM SNP encoding. DEM-Vars uses ordered SNP, INDEL, and SV branches, with ten channels for SNPs and Aquila-Vars four-class channels for INDELs and SVs. Random-forest marker selection is fitted separately per branch using retained training samples only. The original batch-dependent Transformer behavior is preserved. See the [detailed DEM benchmark adaptation](docs/DEM_benchmark_adaptation.md).

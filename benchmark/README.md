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

To ensure a fair comparison, we first generated a fixed nested cross-validation scheme and applied exactly the same sample partitions to all methods. The outer folds were used for final model evaluation, while the inner folds were used for hyperparameter optimization and model selection. All models were therefore evaluated on identical training, validation, and testing sets.

For Aquila, missing phenotypic observations are handled natively through a masked multi-task learning strategy. Specifically, missing trait values are excluded from the loss calculation while the remaining observed traits continue to contribute to model optimization, allowing the model to exploit correlations among multiple traits without requiring phenotype imputation. For other methods that do not support missing phenotypes, missing observations were handled according to their model assumptions. Single-trait models were trained using only individuals with available phenotypic records for the target trait. Importantly, these models still followed the same predefined cross-validation partitions as Aquila, ensuring that differences in performance reflect model behavior rather than differences in data splitting. This evaluation framework assesses genomic prediction performance under realistic incomplete phenotype conditions encountered in practical breeding programs.

### Generate fold mapping:

```sh
# GSTP008.pheno downloaded from CropGS-hub (https://iagr.genomics.cn/CropGS/#/Datasets)
wget https://iagr.genomics.cn/static/gstool/data/GSTP008/population/GSTP008.pheno
# However, the 705rice have a total of 50 low-depth duplicated samples, filtering required.

aquila_cv.py --phenotype Rice655.pheno -o 655rice_nested_cv.json --outer-folds 5 --inner-folds 4 --seed 42
```

The JSON mapping fixes both outer and inner folds and was used throughout the pipeline, including [GWAS lead-variant selection](https://github.com/GooLey1025/Multi_Source_Marker_Panel_generation), to avoid information leakage. Specifically, GWAS discovery and lead-variant selection were performed using only the training samples within each outer fold, while test samples were completely excluded from this process.

When `--save-raw-genotype` is enabled, sample-subset VCF files are written under each prepared directory's `raw_genotype/`. Every outer fold contains `train.vcf.gz` and `test.vcf.gz`; every nested inner fold contains `train.vcf.gz` and `valid.vcf.gz`. These files preserve the source variants and genotype fields, but include only the samples assigned to that split. These raw fold-specific VCFs allow benchmark models to apply their own genotype encodings while using exactly the same samples as Aquila.

Phenotypes are preprocessed once during data preparation and the resulting fold-specific standardized targets are reused by all downstream benchmarks. This provides a consistent phenotype input and prevents each method from introducing differences through independent preprocessing.

To avoid information leakage, preprocessing parameters are always estimated from the corresponding training samples only:

- For each inner fold, trait skewness is calculated using the observed phenotypes in `inner_train`. A `log1p` transformation is applied when the absolute skewness exceeds the configured threshold, after which the trait is standardized using the training-set mean and standard deviation. The same fitted transformation is then applied to `inner_valid`.
- For final evaluation in each outer fold, preprocessing is fitted again using the complete `outer_train` partition. The fitted transformation is applied unchanged to both `outer_train` and `outer_test`.
- Missing phenotype values are excluded when fitting preprocessing parameters and remain masked in the prepared targets.

Each inner-fold directory stores `Y_train_processed.pt`, `Y_valid_processed.pt`, and `preprocessing.json`. The `final` directory of each outer fold stores `Y_train_processed.pt`, `Y_test_processed.pt`, and its own `preprocessing.json`. Together with the predefined nested-CV mapping and fold-specific VCF files, these artifacts form the common data inputs used for all benchmark models.

Single-trait benchmark models that do not natively support missing phenotypes, such as MENET, discard samples with an unobserved target separately within each training, validation, and test partition. The `-999` missing-value sentinel is never passed to their loss functions or evaluation metrics. The remaining samples retain the same predefined nested-CV assignments and fold-local phenotype transformations used by Aquila.

For regression benchmarks, both Aquila-GS and the integrated comparison models report Pearson r, R², MSE, RMSE, and MAE on the available test observations. Fold outputs include metrics on both the standardized phenotype scale and the inverse-transformed original scale. The comparison summary uses within-trait Pearson: for each phenotype, Pearson correlation is calculated across its held-out samples and then averaged across outer folds. This is distinct from Aquila's optional within-accession Pearson, which correlates traits within a sample and is undefined for independent single-trait models such as DEM-SNP and DEM-Vars.

`summary_and_plot_benchmark_model.py` reads this per-trait Pearson contract from all benchmark models, including Aquila-SNP, Aquila-Vars, DEM-SNP, and DEM-Vars. It supports both complete cohort result directories and the `fold_0` to `fold_4` result shards produced for fold-specific GWAS data. Its long-form output labels the metric as `within_trait_pearson`.

## Benchmark methodology

See the [Benchmark Fairness Checklist](BENCHMARK_FAIRNESS.md) for the partitioning, leakage prevention, preprocessing, model-selection, evaluation, and reporting requirements.

### SNP input and feature-selection policy

All models in the main benchmark use the same fold-specific SNP panels whenever their architectures permit, so differences in performance are not caused by independently selected input markers. Model-specific feature selection was disabled when it was an optional preprocessing step rather than an essential part of the method.

Three exceptions require explicit treatment. BNNs retains its internal Lasso-based feature-selection module because it is a defining component of the proposed architecture; consequently, its effective SNP subset can differ from those used by other models, although its nested-CV performance was not competitive in this benchmark. Cropformer originally applies mutual-information-coefficient (MIC) filtering with a default maximum of 10,000 SNPs; this optional filtering was disabled so that Cropformer receives the same SNP panel as the other models. DEM originally uses random-forest SNP selection. Removing this stage would expand each sample to approximately `number_of_SNPs × 10` genotype-class features, producing an impractically large model input and causing out-of-memory errors on the RTX 4090 benchmark server. DEM was therefore excluded from the main horizontal comparison and evaluated separately against Aquila-SNP using the same marker set selected by DEM, allowing the architectural capabilities of DEM and Aquila-SNP to be compared under matched markers.

## Reproduce of 655rice dataset

### [Aquila](https://github.com/GooLey1025/Aquila-GS)

GWAS lead variants are selected independently from the training samples of each outer fold, resulting in a different marker panel for every fold. Run the following data preparation commands from the `benchmark` root directory.

#### Prepare Aquila-SNP data

```sh
for FOLD in 0 1 2 3 4; do
  aquila_data_cv.py \
    --vcf "species_data/Rice655-current_fold_GWAS/outer_fold_${FOLD}/final_panel/marker_panel.outer_fold_${FOLD}.final.imputed.panel.vcf.gz" \
    --phenotype Rice655.pheno \
    --encoding-type diploid_onehot \
    --variant-type snp \
    --id-prefix "SNP-" \
    --fold-mapping 655rice_nested_cv.json \
    --fold-specific-gwas-fold "$FOLD" \
    -o "Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --save-raw-genotype \
    --overwrite
done
```

#### Prepare Aquila-Vars data

Aquila-Vars uses separate SNP, INDEL, and SV branches, so its prepared data are generated without forcing SNP-only encoding or applying an ID-prefix filter:

```sh
for FOLD in 0 1 2 3 4; do
  aquila_data_cv.py \
    --vcf "species_data/Rice655-current_fold_GWAS/outer_fold_${FOLD}/final_panel/marker_panel.outer_fold_${FOLD}.final.imputed.panel.vcf.gz" \
    --phenotype Rice655.pheno \
    --encoding-type diploid_onehot \
    --fold-mapping 655rice_nested_cv.json \
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

for FOLD in 0 1 2 3 4; do
  aquila_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config  32hpo_budgets.yaml \
    -o "results/Rice655/fold_${FOLD}" \
    --live-metrics-log \
    --use-deterministic \
    --overwrite
done

cd ..
```

#### Train Aquila-Vars

```sh
cd aquila-vars

for FOLD in 0 1 2 3 4; do
  aquila_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.vars.cv.data/fold_${FOLD}" \
    --config 32hpo_budgets.aquila-vars.yaml \
    -o "results/Rice655/fold_${FOLD}" \
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

for FOLD in 0 1 2 3 4; do
  python src_benchmark/adapter.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
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
for FOLD in 0 1 2 3 4; do
  python src_benchmark/adapter.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

### XGBoost

```sh
cd xgboost
for FOLD in 0 1 2 3 4; do
  python xgboost_train_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/xgboost_nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
    --n-jobs 32 \
    --overwrite
done
```

### BayesCpi

```sh
cd bayescpi
for FOLD in 0 1 2 3 4; do
  python bayescpi_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
    --overwrite
done
```

### rrBLUP

```sh
cd rrBLUP
for FOLD in 0 1 2 3 4; do
  python rrblup_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
    --overwrite
done
```

### Lasso

```sh
cd Lasso
for FOLD in 0 1 2 3 4; do
  python lasso_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
    --overwrite
done
```

### ElasticNet

```sh
cd ElasticNet
for FOLD in 0 1 2 3 4; do
  python elasticnet_nested_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/nested_cv.yaml \
    --outer-folds "$FOLD" \
    -o "results/Rice655/fold_${FOLD}" \
    --overwrite
done
```

### [CLCNet](https://github.com/SuppurNewer/CLCNet)

Some scripts from CLCNet were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `01d7792c97dc05f8a54afbfb2f62427607f60aad`.

```sh
cd CLCNet
for FOLD in 0 1 2 3 4; do
  python CLCNet_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/CLCNet_nested_cv.yaml \
    --outer-folds "$FOLD" \
    --jobs-per-gpu 2 \
    -o "results/Rice655/fold_${FOLD}" \
    --overwrite
done
```

### [MeNet](https://github.com/ganlab/MENET)

Some scripts from MENET were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `06456e4542ab26716c4db8dd8f17517aa5155ff4`.

```sh
cd MENET
for FOLD in 0 1 2 3 4; do
  python MENET_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/MeNet_nested_cv.yaml \
    --outer-folds "$FOLD" \
    --jobs-per-gpu 2 \
    -o "results/Rice655/fold_${FOLD}" \
    --overwrite
done
```

The benchmark adaptation preserves MENET's two-stage architecture while making model selection and evaluation compatible with the shared leakage-safe nested CV protocol. See the [detailed MENET benchmark adaptation](docs/MENET_benchmark_adaptation.md).

### [DNAwhisper](https://github.com/Marxin1992/Whisperer_of_DNA)

Some scripts from Whisperer of DNA were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `7a657cc00d44263c4b1085d3991ecc0e935c14e5`.

```sh
cd Whisperer_of_DNA
conda activate aquila_dnawhisperer
for FOLD in 0 1 2 3 4; do
  python Whisperer_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/Whisperer_nested_cv.yaml \
    --outer-folds "$FOLD" \
    --output-dir "results/Rice655/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

### [Bayesian Neural Networks](https://github.com/GSBreeder/BNNs)

Some scripts from BNNs were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `6a8a82fd68625443a1c6dbcef19b428d031fa18f`.

BNNs retains its internal Lasso-based feature-selection module because it is treated as part of the model's architecture rather than external preprocessing. Its effective SNP subset can therefore differ from the shared benchmark panel. Despite retaining this feature-selection mechanism, BNNs did not achieve competitive nested-CV performance in this benchmark.

```sh
cd BNNs
for FOLD in 0 1 2 3 4; do
  python BNNs_train_cv.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/BNNs_nested_cv.yaml \
    --outer-folds "$FOLD" \
    --output-dir "results/Rice655/fold_${FOLD}" \
    --jobs-per-gpu 2 \
    --overwrite
done
```

All benchmark runners in this section enforce the same prepared-data fold binding. For directories with enabled `fold_specific_gwas` metadata, omitting `--outer-folds` selects only the bound fold, while explicitly requesting another fold or multiple folds is rejected. Prepared-data directories without this metadata retain the original all-fold default.


### Specific case: DEM vs Aquila

#### [DEM](https://github.com/cma2015/DEM/)

Some scripts from DEM were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `86de718f950d0ecc5554ff1916e2d59f51a33ce8`.

DEM's random-forest (RF) marker selection is retained because removing it produces approximately `number_of_SNPs × 10` genotype-class features per sample and exceeded the available memory on the RTX 4090 benchmark server. DEM is not included in the main horizontal comparison. Instead, `GYP_BLUP`, `HD_BLUP`, and `PH_BLUP` are evaluated as independent single-trait tasks. DEM-SNP and Aquila-SNP share the 1,000 SNP markers selected by the DEM-SNP RF, while DEM-Vars and Aquila-Vars share up to 1,000 markers independently selected for each SNP, INDEL, and SV branch by the DEM-Vars RF.

Marker selection remains nested-CV local: each inner-fold RF is fitted only on that inner-training split, and the final RF is refitted using the complete outer-training split before outer-test evaluation. Aquila uses the same selected marker identities but retains its own genotype encodings. The complete two-stage workflow is provided in `DEM/run_dem_rf_matched_comparison.sh`.

```sh
cd DEM
python DEM_train_benchmark.py \
  --data-dir ../Maize1404.snp.cv.data \
  --config configs/DEM-SNP_nested_cv.yaml \
  --output-dir results/DEM-SNP/Maize1404

python DEM_train_benchmark.py \
  --data-dir ../Maize1404.vars.cv.data \
  --config configs/DEM-Vars_nested_cv.yaml \
  --output-dir results/DEM-Vars/Maize1404
```

For the matched-marker comparison on Rice655:

```sh
cd DEM
bash run_dem_rf_matched_comparison.sh
```

DEM-SNP and DEM-Vars train independent single-output models for each selected trait and outer fold. Missing phenotypes are removed only within the assigned train, validation, or test split, so predefined fold membership is unchanged, and training uses ordinary MSE. DEM-SNP uses the original ten-channel DEM SNP encoding. DEM-Vars uses ordered SNP, INDEL, and SV branches, with ten channels for SNPs and Aquila-Vars four-class channels for INDELs and SVs. Random-forest marker selection is fitted separately per branch using retained training samples only. The original batch-dependent Transformer behavior is preserved. See the [detailed DEM benchmark adaptation](docs/DEM_benchmark_adaptation.md).
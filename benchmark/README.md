# Benchmark

All benchmarks were conducted using the versions of the compared methods that were available at the time (2026-07) of our experiments.

## Data Prepare

To ensure a fair comparison, we first generated a fixed nested cross-validation scheme and applied exactly the same sample partitions to all methods. The outer folds were used for final model evaluation, while the inner folds were used for hyperparameter optimization and model selection. All models were therefore evaluated on identical training, validation, and testing sets.

For Aquila, missing phenotypic observations are handled natively through a masked multi-task learning strategy. Specifically, missing trait values are excluded from the loss calculation while the remaining observed traits continue to contribute to model optimization, allowing the model to exploit correlations among multiple traits without requiring phenotype imputation. For other methods that do not support missing phenotypes, missing observations were handled according to their model assumptions. Single-trait models were trained using only individuals with available phenotypic records for the target trait. Importantly, these models still followed the same predefined cross-validation partitions as Aquila, ensuring that differences in performance reflect model behavior rather than differences in data splitting. This evaluation framework assesses genomic prediction performance under realistic incomplete phenotype conditions encountered in practical breeding programs.

### Generate fold mapping:

```sh
# GSTP008.pheno downloaded from CropGS-hub (https://iagr.genomics.cn/CropGS/#/Datasets)
wget https://iagr.genomics.cn/static/gstool/data/GSTP008/population/GSTP008.pheno
aquila_cv.py --phenotype GSTP008.pheno -o 705rice_nested_cv.json --outer-folds 5 --inner-folds 4 --seed 42
```

The JSON mapping fixes both outer and inner folds and was used throughout the pipeline, including [GWAS lead-variant selection](to_be_add), to avoid information leakage. Specifically, GWAS discovery and lead-variant selection were performed using only the training samples within each outer fold, while test samples were completely excluded from this process.

### Generate 5-fold training and testing sets:

When having done the GWAS lead variant selection, we can use the following command to generate the training and testing sets:

```sh
aquila_data_cv.py --vcf ../case/705rice_0.03.full.all.impute.biallelic.vcf.gz --phenotype GSTP008.pheno --encoding-type diploid_onehot --variant-type snp --id-prefix "SNP-" --fold-mapping 705rice_nested_cv.json -o test --save-raw-genotype --overwrite
aquila_data_cv.py --vcf ../case/705rice_0.03.full.all.impute.biallelic.vcf.gz --phenotype GSTP008.pheno --encoding-type 10classed_onehot --variant-type snp --id-prefix "SNP-" --fold-mapping 705rice_nested_cv.json -o test_10classes_onehot --save-raw-genotype --overwrite
```

When `--save-raw-genotype` is enabled, sample-subset VCF files are written under `test/raw_genotype/`. Every outer fold contains `train.vcf.gz` and `test.vcf.gz`; every nested inner fold contains `train.vcf.gz` and `valid.vcf.gz`. These files preserve the source variants and genotype fields, but include only the samples assigned to that split. These raw fold-specific VCFs allow benchmark models to apply their own genotype encodings while using exactly the same samples as Aquila.

Phenotypes are preprocessed once during data preparation and the resulting fold-specific standardized targets are reused by all downstream benchmarks. This provides a consistent phenotype input and prevents each method from introducing differences through independent preprocessing.

To avoid information leakage, preprocessing parameters are always estimated from the corresponding training samples only:

- For each inner fold, trait skewness is calculated using the observed phenotypes in `inner_train`. A `log1p` transformation is applied when the absolute skewness exceeds the configured threshold, after which the trait is standardized using the training-set mean and standard deviation. The same fitted transformation is then applied to `inner_valid`.
- For final evaluation in each outer fold, preprocessing is fitted again using the complete `outer_train` partition. The fitted transformation is applied unchanged to both `outer_train` and `outer_test`.
- Missing phenotype values are excluded when fitting preprocessing parameters and remain masked in the prepared targets.

Each inner-fold directory stores `Y_train_processed.pt`, `Y_valid_processed.pt`, and `preprocessing.json`. The `final` directory of each outer fold stores `Y_train_processed.pt`, `Y_test_processed.pt`, and its own `preprocessing.json`. Together with the predefined nested-CV mapping and fold-specific VCF files, these artifacts form the common data inputs used for all benchmark models.

Single-trait benchmark models that do not natively support missing phenotypes, such as MENET, discard samples with an unobserved target separately within each training, validation, and test partition. The `-999` missing-value sentinel is never passed to their loss functions or evaluation metrics. The remaining samples retain the same predefined nested-CV assignments and fold-local phenotype transformations used by Aquila.

For regression benchmarks, both Aquila-GS and the integrated comparison models report Pearson r, R², MSE, RMSE, and MAE on the available test observations. Fold outputs include metrics on both the standardized phenotype scale and the inverse-transformed original scale.

## Benchmark Fairness Checklist

| Check target                        | Purpose                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fixed nested-CV mapping             | Use the same 5 outer and 4 inner sample partitions for every model so performance differences are not caused by different random splits.                                                                                                                                                                                                                                                    |
| Outer-test isolation                | Keep outer-test genotypes and phenotypes out of preprocessing, feature selection, HPO, early stopping, and epoch selection.                                                                                                                                                                                                                                                                 |
| Training-only learned preprocessing | Fit phenotype transforms, genotype scaling or imputation, GWAS, marker selection, encoders, and reference representations only on the corresponding training partition.                                                                                                                                                                                                                     |
| Shared phenotype inputs             | Reuse fold-local processed phenotypes fitted during common data preparation instead of independently normalizing the full cohort for each model.                                                                                                                                                                                                                                            |
| Missing-phenotype handling          | Exclude unavailable targets from losses and metrics without moving samples between the predefined folds; never pass the`-999` sentinel into training or evaluation.                                                                                                                                                                                                                       |
| Identical HPO budget                | Evaluate the specified number of candidate configurations on every inner fold and select by mean inner-validation Pearson correlation.                                                                                                                                                                                                                                                      |
| Final refit from scratch            | Retrain the selected configuration on complete outer-training data for an inner-CV-derived duration before evaluating the outer test fold once.                                                                                                                                                                                                                                             |
| Complete training batches           | Set`drop_last=True` for neural-network training loaders, so a final batch smaller than `batch_size` is not used for an optimizer step. This keeps training batch size consistent and avoids unstable batch-dependent normalization statistics. Validation and test loaders retain all samples with `drop_last=False`. Full-batch methods without mini-batch loaders are not affected. |
| Consistent metrics                  | Report Pearson r, R², MSE, RMSE, and MAE on both processed and inverse-transformed original phenotype scales using observed test targets only.                                                                                                                                                                                                                                             |
| Deterministic execution             | Derive and record seeds for folds, candidates, data shuffling, pair or triplet sampling, and learned preprocessing where supported.                                                                                                                                                                                                                                                         |
| Auditable outputs                   | Save selected parameters, inner-fold results, epochs, preprocessing metadata, sample IDs, predictions, metrics, runtime, and final checkpoints for each outer fold.                                                                                                                                                                                                                         |

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
# A proxy may be required to download this package.
Rscript -e 'install.packages("hibayes", repos="https://cloud.r-project.org")'
```

Environment: NVIDIA-GPU-4090 x3, Driver Version: 580.105.08, CUDA Version: 12.2

## Reproduce of 705rice dataset

### [Aquila](https://github.com/GooLey1025/Aquila-GS)

Generate the shared prepared-data directory once, as described above. Every model command below requires this directory explicitly and consumes its fixed outer and inner folds.

Run Aquila with the same prepared data:

```sh
cd aquila

# For Aquila-SNP
aquila_train_cv.py --data-dir ../test --config params/705rice_conv_mha.aquila-snp.hpo.yaml -o test_train_0722_night  --live-metrics-log --overwrite --folds 1

# For Aquila-Vars
aquila_train_cv.py --data-dir ../test --config params/705rice_conv_mha.aquila-vars.hpo.yaml -o test_train_0722_night  --live-metrics-log --overwrite --folds 1
```

### [CropARNet](https://github.com/Zhoushuchang-lab/CropARNet)

Some scripts from CropARNet were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `b9996564d0f021d2d24781935abb04a166c0342e`.

```sh
cd croparnet
python src_benchmark/adapter.py \
  --data-dir /path/to/prepared-data \
  --config configs/nested_cv.yaml \
  -o results/croparnet \
  --jobs-per-gpu 2
```

### [Cropformer](https://github.com/jiekesen/Cropformer.git)

Some scripts from Cropformer were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `e0a77cf699b034d956b249f41b2e1f357f486f06`.

```sh
cd cropformer
python src_benchmark/adapter.py \
  --data-dir /path/to/prepared-data \
  --config configs/nested_cv.yaml \
  -o results/cropformer \
  --jobs-per-gpu 2
```

### XGBoost

```sh
cd xgboost
python xgboost_train_nested_cv.py \
  --data-dir /path/to/prepared-data \
  --config configs/xgboost_nested_cv.yaml \
  -o results/xgboost \
  --n-jobs 32
```

### BayesCpi

```sh
cd bayescpi
python bayescpi_nested_cv.py \
  --data-dir /path/to/prepared-data \
  --config configs/nested_cv.yaml \
  -o results/bayescpi
```

### rrBLUP

```sh
cd rrBLUP
python rrblup_nested_cv.py \
  --data-dir /path/to/prepared-data \
  --config configs/nested_cv.yaml \
  -o results/rrblup
```

### Lasso

```sh
cd Lasso
python lasso_nested_cv.py \
  --data-dir /path/to/prepared-data \
  --config configs/nested_cv.yaml \
  -o results/lasso
```

### ElasticNet

```sh
cd ElasticNet
python elasticnet_nested_cv.py \
  --data-dir /path/to/prepared-data \
  --config configs/nested_cv.yaml \
  -o results/elasticnet
```

### [CLCNet](https://github.com/SuppurNewer/CLCNet)

Some scripts from CLCNet were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `01d7792c97dc05f8a54afbfb2f62427607f60aad`.

```sh
cd clcnet
python CLCNet_train_cv.py \
  --data-dir ../test_v2 \
  --config configs/CLCNet_nested_cv.yaml \
  --jobs-per-gpu 2 \
  -o results/clcnet_gyp
```

### [MeNet](https://github.com/ganlab/MENET)

Some scripts from MENET were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `06456e4542ab26716c4db8dd8f17517aa5155ff4`.

```sh
cd menet
python MENET_train_cv.py \
  --data-dir ../test_v2 \
  --config configs/MeNet_nested_cv.yaml \
  --jobs-per-gpu 2 \
  -o results/menet

```

The benchmark adaptation preserves MENET's two-stage architecture while making model selection and evaluation compatible with the shared leakage-safe nested CV protocol. See the [detailed MENET benchmark adaptation](docs/MENET_benchmark_adaptation.md).

### [DEM](https://github.com/cma2015/DEM/)

Some scripts from DEM were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `86de718f950d0ecc5554ff1916e2d59f51a33ce8`.

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

DEM-SNP and DEM-Vars train independent single-output models for each selected
trait and outer fold. Missing phenotypes are removed only within the assigned
train, validation, or test split, so predefined fold membership is unchanged,
and training uses ordinary MSE. DEM-SNP uses the original ten-channel DEM SNP
encoding. DEM-Vars uses ordered SNP, INDEL, and SV branches, with ten channels
for SNPs and Aquila-Vars four-class channels for INDELs and SVs. Optional
random-forest marker selection is fitted separately per branch on retained
training samples only. The original batch-dependent Transformer behavior is
preserved. See the
[detailed DEM benchmark adaptation](docs/DEM_benchmark_adaptation.md).

### [DNAwhisper](https://github.com/Marxin1992/Whisperer_of_DNA)

Some scripts from Whisperer of DNA were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `7a657cc00d44263c4b1085d3991ecc0e935c14e5`.

#### Train

```sh
cd ~/projects/Aquila-GS/benchmark/Whisperer_of_DNA
python Whisperer_train_cv.py \
  --data-dir ../test \
  --config configs/Whisperer_nested_cv.yaml \
  --output-dir whisperer_nested_cv_output \
  --jobs-per-gpu 2
```

### [Bayesian Neural Networks](https://github.com/GSBreeder/BNNs)

Some scripts from BNNs were copied or adapted into our project repository. The upstream source code version referenced for this benchmark is commit `6a8a82fd68625443a1c6dbcef19b428d031fa18f`.

```sh
cd ~/projects/Aquila-GS/benchmark/BNNs
python BNNs_train_cv.py \
  --data-dir ../test_v2 \
  --config configs/BNNs_nested_cv.yaml \
  --output-dir results/bnn_gyp \
  --jobs-per-gpu 2
```

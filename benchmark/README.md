# Benchmark

All benchmarks were conducted using the versions of the compared methods that were available at the time (2026-07) of our experiments. For transparency and reproducibility, we report the exact commit hashes used in our experiments.

Note that for benchmarking purposes, it is not necessary to clone all original repositories, as the required scripts have been integrated or adapted into this project.

Environment: NVIDIA-GPU-4090 x3


## Data Prepare

To ensure a fair comparison, we first generated a fixed nested cross-validation scheme and applied exactly the same sample partitions to all methods. The outer folds were used for final model evaluation, while the inner folds were used for hyperparameter optimization and model selection. All models were therefore evaluated on identical training, validation, and testing sets.

For Aquila, missing phenotypic observations are handled natively through a masked multi-task learning strategy. Specifically, missing trait values are excluded from the loss calculation while the remaining observed traits continue to contribute to model optimization, allowing the model to exploit correlations among multiple traits without requiring phenotype imputation. For other methods that do not support missing phenotypes, missing observations were handled according to their model assumptions. Single-trait models were trained using only individuals with available phenotypic records for the target trait. Importantly, these models still followed the same predefined cross-validation partitions as Aquila, ensuring that differences in performance reflect model behavior rather than differences in data splitting. Prediction accuracy was calculated using only individuals with available phenotypic records in the test sets. For each trait, test samples with missing phenotype values were excluded from the accuracy calculation, and the same evaluation criteria were applied consistently to all models. This evaluation framework assesses genomic prediction performance under realistic incomplete phenotype conditions encountered in practical breeding programs.

### Generate fold mapping:
```sh
# GSTP008.pheno downloaded from CropGS-hub (https://iagr.genomics.cn/CropGS/#/Datasets)
wget https://iagr.genomics.cn/static/gstool/data/GSTP008/population/GSTP008.pheno
aquila_cv.py --phenotype GSTP008.pheno -o 705rice_nested_cv.json --outer-folds 5 --inner-folds 4 --seed 42
```
The JSON mapping fixes both outer and inner folds and was used throughout the
pipeline, including [GWAS lead-variant selection](to_be_add), to avoid
information leakage. Specifically, GWAS discovery and lead-variant selection
were performed using only the training samples within each outer fold, while test
samples were completely excluded from this process.

### Generate 5-fold training and testing sets:
When having done the GWAS lead variant selection, we can use the following command to generate the training and testing sets:

```sh
aquila_data_cv.py --vcf ../case/705rice_0.03.full.all.impute.biallelic.vcf.gz --phenotype GSTP008.pheno --encoding-type diploid_onehot --variant-type snp --fold-mapping 705rice_nested_cv.json -o test_v2 --save-raw-genotype --overwrite
```

When `--save-raw-genotype` is enabled, sample-subset VCF files are written
under `test/raw_genotype/`. Every outer fold contains `train.vcf.gz` and
`test.vcf.gz`; every nested inner fold contains `train.vcf.gz` and
`valid.vcf.gz`. These files preserve the source variants and genotype fields,
but include only the samples assigned to that split. These raw fold-specific
VCFs allow benchmark models to apply their own genotype encodings while using
exactly the same samples as Aquila.

Phenotypes are preprocessed once during data preparation and the resulting
fold-specific standardized targets are reused by all downstream benchmarks.
This provides a consistent phenotype input and prevents each method from
introducing differences through independent preprocessing.

To avoid information leakage, preprocessing parameters are always estimated
from the corresponding training samples only:

- For each inner fold, trait skewness is calculated using the observed
  phenotypes in `inner_train`. A `log1p` transformation is applied when the
  absolute skewness exceeds the configured threshold, after which the trait is
  standardized using the training-set mean and standard deviation. The same
  fitted transformation is then applied to `inner_valid`.
- For final evaluation in each outer fold, preprocessing is fitted again using
  the complete `outer_train` partition. The fitted transformation is applied
  unchanged to both `outer_train` and `outer_test`.
- Missing phenotype values are excluded when fitting preprocessing parameters
  and remain masked in the prepared targets.

Each inner-fold directory stores `Y_train_processed.pt`,
`Y_valid_processed.pt`, and `preprocessing.json`. The `final` directory of
each outer fold stores `Y_train_processed.pt`, `Y_test_processed.pt`, and its
own `preprocessing.json`. Together with the predefined nested-CV mapping and
fold-specific VCF files, these artifacts form the common data inputs used for
all benchmark models.

Single-trait benchmark models that do not natively support missing
phenotypes, such as MENET, discard samples with an unobserved target separately
within each training, validation, and test partition. The `-999` missing-value
sentinel is never passed to their loss functions or evaluation metrics. The
remaining samples retain the same predefined nested-CV assignments and
fold-local phenotype transformations used by Aquila.

For regression benchmarks, both Aquila-GS and the integrated comparison
models report Pearson r, R², MSE, RMSE, and MAE on the available test
observations. Fold outputs include metrics on both the standardized phenotype
scale and the inverse-transformed original scale.


## Prerequisites

### Install required R packages for [XGBoost](#xgboost), [BayesA](#bayesa), [rrBLUP](#rrblup), [Lasso](#lasso),[ ElasticNet](#elasticnet).
Our R version: 4.3.3
```r
install.packages(c(
  "xgboost",           # XGBoost
  "BGLR",              # BayesA
  "rrBLUP",            # rrBLUP
  "glmnet",            # Lasso and ElasticNet
  "vcfR",              # VCF file reading
  "yaml",              # Configuration files
  "data.table",        # Fast data loading
  "rBayesianOptimization",  # HPO (optional)
  "foreach",           # Parallel processing
  "doParallel",        # Parallel backend
  "optparse",           # Command-line arguments
  "hibayes"
))
```


## Reproduce

### [Aquila](https://github.com/GooLey1025/Aquila-GS)
First, run Aquila training to generate fair data for benchmarking:
```sh
conda activate aquila
cd aquila
aquila_train.py --config params/705rice_conv_mha.aquila-snp.hpo.yaml \
  --vcf beagle_impute/705rice_0.005.full.snp.impute.biallelic.vcf.gz \
  --pheno 705Rice.pheno.imputed.tsv -o aquila_benchmark --save-postprocess-data
```
This will create:
- `aquila/aquila_benchmark/data_postprocess/geno_train.vcf`
- `aquila/aquila_benchmark/data_postprocess/geno_valid.vcf`
- `aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv`
- `aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv`

#### Aquila-SNP
Run hpo search for Aquila-SNP
```sh
yaml=705rice_conv_mha.aquila-snp.hpo.yaml
rm -rf ${yaml%.yaml}
nohup aquila_train_hpo.py --config params/$yaml -o ${yaml%.yaml} -dsf aquila_benchmark/data_split.tsv > ${yaml%.yaml}.log 2>&1 &
```
#### For Aquila-Vars
Run hpo search for Aquila-Vars
```sh
yaml=705rice_conv_mha.aquila-vars.hpo.yaml
rm -rf ${yaml%.yaml}
nohup aquila_train_hpo.py --config params/$yaml -o ${yaml%.yaml} -dsf aquila_benchmark/data_split.tsv > ${yaml%.yaml}.log 2>&1 &

```
### [CropARNet](https://github.com/Zhoushuchang-lab/CropARNet)

Some scripts from CropARNet were copied or adapted into our project repository.
Therefore, you do not need to clone the original repository to run our code.
`git clone` information below is provided solely for reporting the exact commit hashes used in our experiments.

>```sh
>git clone https://github.com/Zhoushuchang-lab/CropARNet.git
>cd CropARNet
>git rev-parse HEAD
># Commit used in our experiments:
># d53f381de0b453d6ce626e70f0a8b1c2d0c7efde
>
># (Optional) To exactly reproduce our setup:
>git checkout d53f381de0b453d6ce626e70f0a8b1c2d0c7efde
>```

```sh
conda activate aquila
model=croparnet
cd $model
python train_benchmark.py \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir ${model}_output \
  --config ${model}_config.yaml \
  --enable-hpo \
  --seed 42
```

### [Cropformer](https://github.com/jiekesen/Cropformer.git)
>```sh
>git clone https://github.com/jiekesen/Cropformer.git
>cd Cropformer
>git rev-parse HEAD
># Commit used in our experiments:
># e0a77cf699b034d956b249f41b2e1f357f486f06
>```
```sh
conda activate aquila
model=cropformer
cd cropformer
python model_benchmark.py \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir ${model}_output \
  --config ${model}_config.yaml \
  --enable-hpo \
  --seed 42
```

### XGBoost
```sh
model=xgboost
cd $model
Rscript ${model}_train.r \
  --config ${model}_config.yaml \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-di ${model}_output --n-cores 32 \
  --enable-hpo

```
### BayesCpi
```sh
model=bayescpi
cd $model
Rscript ${model}_train.r \
  --config ${model}_config.yaml \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir ${model}_output --n-cores 32
```

### rrBLUP
```sh
model=rrBLUP
cd $model
Rscript ${model}_train.r \
  --config ${model}_config.yaml \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir ${model}_output --n-cores 32
```
### Lasso
```sh
model=Lasso
cd $model
Rscript ${model}_train.r \
  --config ${model}_config.yaml \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir ${model}_output --n-cores 32
```

### ElasticNet
```sh
model=ElasticNet
cd $model
Rscript ${model}_train.r \
  --config ${model}_config.yaml \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir ${model}_output --n-cores 32
```

### [DNNGP](https://github.com/AIBreeding/DNNGP) (Excluded)
Although DNNGP is publicly available, it relies on precompiled binary files and does not release its source code. The provided binaries are incompatible with modern CUDA and GPU environments (e.g., RTX 4090), preventing fair GPU-based evaluation. While we attempted to run DNNGP on CPU, the performance was not comparable; therefore, it was excluded from the final benchmark.
```sh
git clone https://github.com/AIBreeding/DNNGP.git
cd DNNGP
git rev-parse HEAD
# Commit used in our experiments:
# 3bbac096969fb2b46958a672d342297cb4457116

# (Optional) To reproduce the exact version:
git checkout 3bbac096969fb2b46958a672d342297cb4457116
```

### [CLCNet](https://github.com/SuppurNewer/CLCNet)
```sh
conda create -n CLCNet python=3.10.13
conda activate CLCNet

git clone https://github.com/SuppurNewer/CLCNet.git
cd CLCNet

# Dependency conflicts, manually replace.
sed -i 's/pandas==1\.5\.3/pandas>=2.2,<3.0/' requirements.txt
sed -i 's/\r$//' requirements.txt
sed -i 's/^tqdm==4\.65\.0$/tqdm>=4.66,<5.0/' requirements.txt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

```
python ChromosomeAwareProcessor.py \
  --gstp_name example \
  --data_dir example \
  --traits Trait1 Trait2

### [MeNet](https://github.com/ganlab/MENET)
```sh
conda create -n MeNet python=3.9.17
conda activate MeNet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/ganlab/MENET.git
cd MENET
pip install -r requirements.txt
python src_benchmark/MENET_train_cv.py \
  --data-dir ../test_v2 \
  --config src_benchmark/configs/MeNet_nested_cv.yaml \
  --traits GYP_BLUP \
  --device cuda:0 \
  -o results/menet_gyp
```

### [DEM](https://github.com/cma2015/DEM/)
#### Install
```sh
conda create -n dem python=3.11
conda activate dem

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
git clone https://github.com/cma2015/DEM.git
cd DEM
pip3 install -e .
```

#### Train
```sh
conda activate dem

cd ~/projects/Aquila-GS
python benchmark/DEM/DEM_train_benchmark.py \
  --data-dir benchmark/test_v2 \
  --config benchmark/DEM/configs/DEM_nested_cv.yaml \
  --output-dir benchmark/DEM/dem_nested_cv_output \
  --device cuda:0
```

The benchmark adapter uses all regression traits as one multi-output target.
Because the original DEM loss has no phenotype mask, only individuals observed
for every regression trait are retained separately within each prepared split.
The predefined 5 outer and 4 inner folds are unchanged. Test metrics are
calculated only after hyperparameter selection and final outer-training
retraining.

Raw fold-specific VCFs under `benchmark/test_v2/raw_genotype/` are encoded as
0/1/2 alternate-allele dosage. For every inner fold and final outer fold, a
multi-output random forest is fitted on training samples only and selects 1,000
SNPs. The selected marker schema is then applied unchanged to validation or
test samples.

The HPO budget matches MENET's candidate-count convention: five binary DEM
hyperparameters form a deterministic `2^5 = 32` grid. Each candidate is
evaluated on all four inner folds, ranked by mean validation `avg_pearson`, and
the selected candidate is retrained on the complete outer-training fold for
the median selected epoch.

Each `fold_<n>/` directory contains the final checkpoint, full HPO trace,
selected variants, metrics on normalized and original phenotype scales,
predictions, preprocessing metadata, and sample audit. The root `summary.json`
reports per-fold results and the mean and standard deviation across the five
outer test folds; `outer_fold_summary.primary.outer_fold_mean` is the primary
five-fold mean test Pearson value.

For a reduced integration check:

```sh
python benchmark/DEM/DEM_train_benchmark.py \
  --data-dir benchmark/test_v2 \
  --output-dir /tmp/dem_smoke \
  --outer-folds 0 \
  --max-inner-folds 1 \
  --max-candidates 1 \
  --device cpu \
  --overwrite
```

The complete prepared directory must include `Y_raw.pt`, `Y_mask.pt`, fold
indices, fold-local processed target tensors, preprocessing JSON files, and the
raw VCF hierarchy described in the Data Prepare section.

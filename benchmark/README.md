# Benchmarking

All benchmarks were conducted using the versions of the compared methods that were available at the time (2025-12) of our experiments.For transparency and reproducibility, we report the exact commit hashes used in our experiments.

Note that for benchmarking purposes, it is not necessary to clone all original repositories, as the required scripts have been integrated or adapted into this project.


## Data Prepare

To ensure a fair comparison among models with different requirements for handling missing values, phenotype imputation was performed prior to model benchmarking. Missing values were imputed using PIXANT with default parameter settings, where the initial imputation strategy was set to random (`initialImputeType = "random"`).
For phenotypes with a very small number of missing observations (fewer than five), a minimal random masking strategy was applied to maintain numerical stability during imputation. Specifically, a small subset of observed values was temporarily masked and included as missing inputs for the imputation procedure, and the original values were subsequently restored after imputation. This approach prevented instability caused by extremely sparse missingness while preserving the original data distribution and integrity.

```sh
wget https://iagr.genomics.cn/static/gstool/data/GSTP008/population/GSTP008.pheno
Rscript pixant_impute/phenotype_pixant_impute.R GSTP008.pheno 705Rice.pheno.imputed.tsv logs_dir
```

`705Rice.pheno.imputed.tsv` was used as the phenotype file for subsequent benchmarking.


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
  "optparse"           # Command-line arguments
))
```


## Reproduce

### [Aquila](https://github.com/GooLey1025/Aquila-GS)
First, run Aquila training to generate fair data for benchmarking:
```sh
conda activate aquila
cd aquila
aquila_train.py --config 705rice_benchmark.yaml --vcf 705Rice_Inbred_ssSNP_0.5LD.ID.reheader.vcf \
  --pheno 705Rice.pheno.imputed.tsv -o aquila_benchmark --save-postprocess-data
```
This will create:
- `aquila/aquila_benchmark/data_postprocess/geno_train.vcf`
- `aquila/aquila_benchmark/data_postprocess/geno_valid.vcf`
- `aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv`
- `aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv`

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
python train_benchmark.py \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir croparnet_output_hpo \
  --config croparnet_config.yaml \
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
cd Cropformer
python model_benchmark.py \
  --train-vcf ../aquila/aquila_benchmark/data_postprocess/geno_train.vcf \
  --valid-vcf ../aquila/aquila_benchmark/data_postprocess/geno_valid.vcf \
  --train-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_train_normalized.tsv \
  --valid-pheno ../aquila/aquila_benchmark/data_postprocess/pheno_valid_normalized.tsv \
  --output-dir cropformer_output_hpo \
  --config cropformer_config.yaml \
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
  --output-dir ${model}_output_hpo --n-cores 32 \
  --enable-hpo

```
### BayesA
```sh
model=bayesa
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
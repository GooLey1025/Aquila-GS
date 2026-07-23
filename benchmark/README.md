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
aquila_cv.py --phenotype GSTP008.pheno -o 705rice_fold_mapping.txt --folds 5 --seed 42
```
The predefined fold mapping `705rice_fold_mapping.txt` was used throughout the pipeline, including [GWAS lead-variant selection](to_be_add), to avoid information leakage. Specifically, GWAS discovery and lead-variant selection were performed using only the training samples within each fold, while test samples were completely excluded from this process.

### Generate 5-fold training and testing sets:
When having done the GWAS lead variant selection, we can use the following command to generate the training and testing sets:

```sh
aquila_data_cv.py --vcf ../case/705rice_0.03.full.all.impute.biallelic.vcf.gz --phenotype GSTP008.pheno --encoding-type diploid_onehot --variant-type snp --fold-mapping 705rice_fold_mapping.txt -o test  --overwrite
```

The preparation stage also caches fold-local phenotype targets. Each inner
fold contains `Y_train_processed.pt`, `Y_valid_processed.pt`, and
`preprocessing.json`; each outer fold contains corresponding `final`
training/test targets. `aquila_train_cv.py` reads these files directly, so
phenotype preprocessing is not repeated for every HPO trial.

```txt
每个 inner fold
    ├── 只使用 inner_train 的有效表型计算 skewness
    ├── 如果 abs(skewness) > 2
    │      └── 对该 trait 做 log1p
    ├── 计算 mean/std
    ├── 对该 trait 做 Z-score
    ├── 保存 Y_train_processed.pt
    ├── 使用同一组参数处理 inner_valid
    └── 保存 Y_valid_processed.pt

完整 outer_train
    ├── 重新计算 skewness、log 参数、mean、std
    ├── 保存 preprocessing.json
    ├── 处理 outer_train
    └── 使用相同参数处理 outer_test完整 outer_train
    ├── 重新计算 skewness、log 参数、mean、std
    ├── 保存 preprocessing.json
    ├── 处理 outer_train
    └── 使用相同参数处理 outer_test
```


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
## Train a production genomic prediction model

Nested CV is used for retrospective, unbiased model evaluation. Once the modeling strategy has been established, a production genomic prediction model can be trained for prospective breeding applications. It uses K-fold cross-validation across the complete reference population to select hyperparameters and the training epoch, then retrains the final model on all available samples. This maximizes the reference information available for predicting new breeding materials or selection candidates.

### 1. Prepare the full reference dataset

```sh
aquila_data_cv_production.py \
  --genotype input.vcf.gz \
  --phenotype phenotype.tsv \
  --encoding diploid_onehot \
  --folds 5 \
  -o dataset.production.data
```

### 2. Train the production model

```sh
aquila_train_cv_production.py \
  --data-dir dataset.production.data \
  --config config.yaml \
  -o results/production
```

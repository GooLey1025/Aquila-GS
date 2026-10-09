COHORT=Maize1404_blup_pheno_10folds
PHENO_FILE=species_data/Maize1404/benchmark.blup.pheno
VCF_FILE=species_data/Maize1404/Maize1404.coding.ld.vcf.gz

COHORT=Soybean975_blup_pheno_10folds
PHENO_FILE=species_data/Soybean975/benchmark.blup.pheno
VCF_FILE=species_data/Soybean975/Soybean975.coding.ld.vcf.gz

COHORT=wheat994_blup_pheno_10folds
PHENO_FILE=species_data/wheat994/benchmark.blup.pheno
VCF_FILE=species_data/wheat994/wheat994.coding.ld.vcf.gz

export PATH="$CONDA_PREFIX/bin:$PATH"

conda activate aquila
aquila_cv.py --phenotype $PHENO_FILE -o $COHORT.nested_cv.json --outer-folds 10 --inner-folds 4 --seed 42 --min-observed 20

aquila_data_cv.py --vcf $VCF_FILE --phenotype $PHENO_FILE --encoding-type diploid_onehot --variant-type snp --fold-mapping $COHORT.nested_cv.json -o $COHORT.cv.data --save-raw-genotype --overwrite

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

# /usr/bin/time -v -o $COHORT.DEM-Vars.time.txt python DEM_train_benchmark.py \
#   --data-dir ../$COHORT.vars.cv.data \
#   --config configs/DEM-Vars_nested_cv.yaml \
#   --output-dir results/DEM-Vars/$COHORT \
#   --jobs-per-gpu 2

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

cd ../aquila-snp
aquila_train_cv.py --data-dir ../$COHORT.cv.data --config 32hpo_budgets.yaml \
  -o results/$COHORT --live-metrics-log --overwrite

# Aquila-SNP single-task benchmark:
# prepare and train one independent model for every phenotype trait while
# reusing the same nested-CV split as the multi-task benchmark above.
IFS=$'\t' read -r -a PHENO_COLUMNS < "../$PHENO_FILE"
for TRAIT in "${PHENO_COLUMNS[@]:1}"; do
  SINGLE_TASK_DATA="../$COHORT.single_task.cv.data/$TRAIT"
  SINGLE_TASK_OUTPUT="results/$COHORT.single_task/$TRAIT"

  echo "Preparing Aquila-SNP single-task data: $COHORT / $TRAIT"
  aquila_data_cv.py \
    --vcf "../$VCF_FILE" \
    --phenotype "../$PHENO_FILE" \
    --traits "$TRAIT" \
    --encoding-type diploid_onehot \
    --variant-type snp \
    --fold-mapping "../$COHORT.nested_cv.json" \
    -o "$SINGLE_TASK_DATA" \
    --overwrite

  echo "Training Aquila-SNP single-task model: $COHORT / $TRAIT"
  aquila_train_cv.py \
    --data-dir "$SINGLE_TASK_DATA" \
    --config 32hpo_budgets.yaml \
    -o "$SINGLE_TASK_OUTPUT" \
    --live-metrics-log \
    --overwrite
done

cd ..
python summary_and_plot_benchmark_model.py --benchmark-dir . Maize1404

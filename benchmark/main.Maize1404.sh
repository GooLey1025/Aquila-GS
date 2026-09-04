COHORT=Maize1404
PHENO_FILE=species_data/Maize1404/benchmark.pheno
VCF_FILE=species_data/Maize1404/Maize1404.LD.rename.vcf.gz

COHORT=Soybean2795
PHENO_FILE=species_data/$COHORT/benchmark.pheno
VCF_FILE=species_data/$COHORT/Soybean2795.LD.rename.vcf.gz

COHORT=wheat994
PHENO_FILE=species_data/wheat994/benchmark.pheno
VCF_FILE=species_data/wheat994/wheat994.LD.vcf.gz

COHORT=Tomato706
PHENO_FILE=species_data/$COHORT/benchmark.pheno
VCF_FILE=species_data/$COHORT/Tomato706.LD.vcf.gz

export PATH="$CONDA_PREFIX/bin:$PATH"

conda activate aquila
aquila_cv.py --phenotype $PHENO_FILE -o $COHORT.nested_cv.json --outer-folds 5 --inner-folds 4 --seed 42 --min-observed 20

aquila_data_cv.py --vcf $VCF_FILE --phenotype $PHENO_FILE --encoding-type diploid_onehot --variant-type snp --fold-mapping $COHORT.nested_cv.json -o $COHORT.cv.data --save-raw-genotype --overwrite

cd croparnet
/usr/bin/time -v -o $COHORT.time.txt python src_benchmark/adapter.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT \
  --jobs-per-gpu 4

cd ../cropformer
/usr/bin/time -v -o $COHORT.time.txt python src_benchmark/adapter.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT \
  --jobs-per-gpu 4

cd ../xgboost
/usr/bin/time -v -o $COHORT.time.txt python xgboost_train_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/xgboost_nested_cv.yaml \
  -o results/$COHORT \
  --n-jobs 4

cd ../bayescpi
/usr/bin/time -v -o $COHORT.time.txt python bayescpi_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT

cd ../rrBLUP
/usr/bin/time -v -o $COHORT.time.txt python rrblup_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/$COHORT


cd ../Lasso
python lasso_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/lasso

cd ../ElasticNet
python elasticnet_nested_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/nested_cv.yaml \
  -o results/elasticnet

cd ../CLCNet
/usr/bin/time -v -o $COHORT.time.txt python CLCNet_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/CLCNet_nested_cv.yaml \
  --jobs-per-gpu 2 \
  -o results/$COHORT

cd ../MENET
/usr/bin/time -v -o $COHORT.time.txt python MENET_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/MeNet_nested_cv.yaml \
  --jobs-per-gpu 2 \
  -o results/$COHORT

cd ../DEM
/usr/bin/time -v -o $COHORT.DEM-SNP.time.txt python DEM_train_benchmark.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/DEM-SNP_nested_cv.yaml \
  --output-dir results/DEM-SNP/$COHORT \
  --jobs-per-gpu 2

# /usr/bin/time -v -o $COHORT.DEM-Vars.time.txt python DEM_train_benchmark.py \
#   --data-dir ../$COHORT.vars.cv.data \
#   --config configs/DEM-Vars_nested_cv.yaml \
#   --output-dir results/DEM-Vars/$COHORT \
#   --jobs-per-gpu 2

cd ../Whisperer_of_DNA
/usr/bin/time -v -o $COHORT.time.txt python Whisperer_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/Whisperer_nested_cv.yaml \
  --output-dir results/$COHORT \
  --jobs-per-gpu 2

cd ../BNNs
/usr/bin/time -v -o $COHORT.time.txt python BNNs_train_cv.py \
  --data-dir ../$COHORT.cv.data \
  --config configs/BNNs_nested_cv.yaml \
  --output-dir results/$COHORT \
  --jobs-per-gpu 2

cd ../aquila-snp
aquila_train_cv.py --data-dir ../$COHORT.cv.data --config conv_mha.aquila-snp.hpo.yaml \
  -o results/$COHORT --live-metrics-log

cd ..
python summary_and_plot_benchmark_model.py --benchmark-dir . Maize1404


## Dependency
- plink2 (https://www.cog-genomics.org/plink/2.0/)
- CUDA v11.2 and CUDNN v8.1 (required for TensorFlow v2.6.0)
## Environment
```sh
conda create -n DNNGP3 python=3.9.16
conda activate DNNGP3.9
conda install -c conda-forge --yes --file requirements.txt
conda install -c nvidia cuda-nvcc
pip install framework-reproducibility==0.4.0

## CUDA, CUDNN, TensorFlow Version Match
CUDA_PATH=/usr/local/cuda-11.8_cudnn8.6 # path to your CUDA
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
python3 cuda_cudnn_tensorflow_match.py
```
## Data preprocess
```sh
geno_path=/home/gulei/projects/Aquila-GS/snp_train/705rice_data/705Rice_Inbred_ssSNP_0.5LD.ID.reheader.vcf
pheno_path=/home/gulei/projects/Aquila-GS/snp_train/705rice_data/705Rice.phenos.completed.tsv

PCA=10
plink2 --threads 32 --vcf $geno_path --pca $PCA --out pca$PCA --allow-extra-chr
python3 tsv2pkl_CLI.py pca$PCA.eigenvec pca$PCA.pkl
python3 phenotype_preprocess.py -i $pheno_path -o phenotypes -r pca$PCA.eigenvec

python3 Scripts/dnngp_runner.py --batch_size 32 --lr 0.001 --epoch 100 --dropout1 0.5 --dropout2 0.3 --patience 20 --seed 42 --cv 10 --part 1 --earlystopping 10 --snp pca$PCA.pkl --pheno phenotypes/BGNP_YangZ15.tsv --output .


OUTDIR="train_out"
LOGDIR="logs"
find phenotypes -maxdepth 1 -type f -name "*.tsv" | sort | \
parallel -j 4 --bar '
  f={}
  base=$(basename "$f" .tsv)
  outdir="'"$OUTDIR"'/${base}.out"
  logfile="'"$LOGDIR"'/${base}.log"
  mkdir -p "$outdir"
  python3 Scripts/dnngp_runner.py \
    --batch_size 32 --lr 0.001 --epoch 300 \
    --dropout1 0.5 --dropout2 0.3 --patience 20 \
    --seed 42 --cv 10 --part 1 --earlystopping 10 \
    --snp "'"pca${PCA}.pkl"'" --pheno "$f" --output "$outdir" \
    > "$logfile" 2>&1
'
```

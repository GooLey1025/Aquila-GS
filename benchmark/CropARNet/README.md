### Requirements
- **Input Files**:
  - `SNP.tsv`: Tab-separated SNP data file
    - First row: headers
    - First column: sample IDs
    - Data values: 0, 1, 2 (genotypes) or -1 (missing values)
  
  - `pheno.tsv`: Tab-separated phenotype data file
    - First row: headers (phenotype names)
    - First column: sample IDs (must match SNP file)
    - Subsequent columns: phenotype values
- **Environment**: The environment is ready once `pip install -e .` completes successfully in the project root directory(`Aquila-GS/`).

### Data preprocess
#### For our project:
```sh
geno_path=/home/gulei/projects/Aquila-GS/snp_train/705rice_data/705Rice_Inbred_ssSNP_0.5LD.ID.reheader.vcf
pheno_path=/home/gulei/projects/Aquila-GS/snp_train/705rice_data/705Rice.phenos.completed.tsv
python vcf_to_CropARNet_snp.py \
  $geno_path \
  CropARNet_SNP.tsv

python align_pheno_to_geno.py CropARNet_SNP.tsv $pheno_path pheno.aligned.tsv
```

### Note: Source Code Modifications for code compatibility

This benchmark uses a slightly modified version (`train_modify.py`) of the original CropARNet training script(`train.py`).

Two minor changes were required to ensure compatibility with recent PyTorch versions:
- EarlyStopping interface fix
The original code passes a col argument to EarlyStopping.__call__(), but the method did not accept it.
We added col=None to the function signature to avoid runtime errors.
- Updated PyTorch AMP API
PyTorch has deprecated torch.cuda.amp.
We updated the code to use: torch.amp.GradScaler("cuda", ...),torch.amp.autocast("cuda", ...)

These changes do not affect model behavior or results, but are required for successful execution with modern PyTorch versions.

### Train
```sh
python3 train_modify.py --snp_path CropARNet_SNP.tsv --pheno pheno.aligned.tsv --result train_out --start_col 1 --end_col 18 --config config.json  > train.log
python3 extract_best_metrics.py train.log best_metrics.tsv
```


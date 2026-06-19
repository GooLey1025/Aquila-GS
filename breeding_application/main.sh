aquila_predict.py --model-dir 705rice_conv_mha.aquila-snp.best_model --vcf 120_inbred_line.snp.indel.sv.impute.biallelic.vcf.gz -o 120_inbred_line.predictions.tsv

python3 plot_inbred_line_growth.py
python3 plot_inbred_line_growth_subpop.py 
python3 Inbred_120_heatmap.py
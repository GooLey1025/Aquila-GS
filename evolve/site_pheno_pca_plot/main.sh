# python3 plot_snp_pca_violin_panel.py   --snps "SNP-3-11326677-1,SNP-2-6397412-1,SNP-11-17985707-1, SNP-10-17076177-1,SNP-11-16984309-1, SNP-2-5410205-1, SNP-9-18122850-1,SNP-3-36150781-1,SNP-6-6752888-1,SNP-10-19058375-1,SNP-7-29628481-1"   --vcf 705rice_0.03.full.all.impute.biallelic.vcf.gz   --pca 705rice.pca.eigenvec   --pheno GSTP008.pheno.tsv   --outdir snp_pca_violin_panels

# python3 plot_qtn_gain_gwas_panel.py \
#   --snp-id "SNP-3-11326677-1,SNP-2-6397412-1,SNP-11-17985707-1, SNP-10-17076177-1,SNP-11-16984309-1, SNP-2-5410205-1, SNP-9-18122850-1,SNP-3-36150781-1,SNP-6-6752888-1,SNP-10-19058375-1,SNP-7-29628481-1" \
#   --gain-file ../705rice_maximize-minimize_seed_mean_QTN_gain_HHZ-base.all_merged.tsv \
#   --gwas-lings 705rice_gwas_results/GYP_LingS15/gemma_lmm.assoc.txt \
#   --gwas-yangz 705rice_gwas_results/GYP_YangZ15/gemma_lmm.assoc.txt \
#   --gwas-delta 705rice_graph.0.5_0.05.SNP_INDEL_SV_GYP_DIFF/results/GYP_delta_YangZ15-LingS15/gemma_lmm.assoc.txt \
#   --out-prefix QTN_gain_GWAS_panel

python3 snp_violin_panel.py --snps snp.tsv --vcf 705rice_0.03.full.all.impute.biallelic.vcf.gz --pheno GSTP008.pheno.tsv --outdir three_violin_out
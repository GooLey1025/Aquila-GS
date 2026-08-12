beagle.nf --snp-vcf Soybean2795.merge.vcf.gz

genome-wide_LD_prune.nf --vcf results/Soybean2795.merge.impute.biallelic.vcf.gz --ld_windows 1000 --ld_step 50 --ld_r2 0.003 -resume
mv genome_wide_LD_prune/Soybean2795.merge.impute.biallelic_pruned.vcf.gz .

bcftools query -l Soybean2795.merge.impute.biallelic_pruned.vcf.gz > header.txt
sed -i 's/_.*//' header.txt
bcftools reheader -N header.txt -o Soybean2795.merge.impute.biallelic_pruned.rename.vcf.gz Soybean2795.merge.impute.biallelic_pruned.vcf.gz


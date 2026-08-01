beagle.nf --snp-vcf Maize1404.vcf.gz
plink2   --vcf results/Maize1404.impute.biallelic.vcf.gz   --make-bed   --out Maize1404.impute
plink   --bfile Maize1404.impute   --indep-pairwise 1000 50 0.003   --out Maize1404.LD
plink   --bfile Maize1404.impute   --extract Maize1404.LD.prune.in   --make-bed   --out Maize1404.LD
plink   --bfile Maize1404.LD   --recode vcf bgz --out Maize1404.LD
bcftools query -l Maize1404.LD.vcf.gz > sample_name.txt
sed -i 's/^0_CUBIC_//' sample_name.txt 
bcftools reheader -N sample_name.txt Maize1404.LD.vcf.gz -o Maize1404.LD.renmae.vcf.gz


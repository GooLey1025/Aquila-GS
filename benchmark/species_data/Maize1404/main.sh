beagle.nf --snp-vcf Maize1404.vcf.gz
plink2   --vcf results/Maize1404.impute.biallelic.vcf.gz   --make-bed   --out Maize1404.impute
plink   --bfile Maize1404.impute   --indep-pairwise 1000 50 0.003   --out Maize1404.LD
plink   --bfile Maize1404.impute   --extract Maize1404.LD.prune.in   --make-bed   --out Maize1404.LD
plink   --bfile Maize1404.LD   --recode vcf bgz --out Maize1404.LD



beagle.nf --snp-vcf  Wheat850.merge.vcf.gz --nthreads 32 -resume

P=Wheat850
plink2   --vcf results/Wheat850.merge.impute.biallelic.vcf.gz --make-bed   --out $P.impute

plink2 \
    --bfile ${P}.impute \
    --snps-only just-acgt \
    --set-all-var-ids '@:#:$r:$a' \
    --make-pgen \
    --out ${P}.SNP \
    --memory 220000

plink2   --pfile $P.SNP   --indep-pairwise 5000 100 0.0001   --out $P.LD --memory 220000
plink2   --pfile $P.SNP   --extract $P.LD.prune.in   --recode vcf bgz   --out $P.LD

beagle.nf --snp-vcf Tomato706.merge_snp.vcf.gz --nthreads 32 --memory "320 GB" -resume

P=Tomato706

plink2 \
    --bfile ${P}.impute \
    --snps-only just-acgt \
    --set-all-var-ids '@:#:$r:$a' \
    --make-pgen \
    --out ${P}.SNP \
    --memory 220000

plink2   --pfile $P.SNP   --indep-pairwise 1000 50 0.0001   --out $P.LD --memory 220000
plink2   --pfile $P.SNP   --extract $P.LD.prune.in   --recode vcf bgz   --out $P.LD

vcfwave Nip_42_rice.vcf.gz -t 24 > Nip_42_rice.vcfwave.vcf

mkdir -p vcf_split
echo {1..12} | tr ' ' '\n' | parallel -j 12 "bcftools view -r P8.Nipponbare.TEJ.Chr{} Nip_42_rice.vcf.gz -o vcf_split/Nip_42_rice.chr{}.vcf"
echo {1..12} | tr ' ' '\n' | parallel -j 12 "vcfwave vcf_split/Nip_42_rice.chr{}.vcf > vcf_split/Nip_42_rice.chr{}.vcfwave.vcf || echo '{}\t$(date +%Y-%m-%d\ %H:%M:%S)'>> vcfwave.error.log"
echo {1..12} | tr ' ' '\n' | parallel -j 12 "bcftools norm -m -both vcf_split/Nip_42_rice.chr{}.vcfwave.vcf -o vcf_split/Nip_42_rice.chr{}.vcfwave.biallelic.vcf; python3 annotate_vcfwave.py vcf_split/Nip_42_rice.chr{}.vcfwave.biallelic.vcf > vcf_split/Nip_42_rice.chr{}.vcfwave.biallelic.annotated.vcf"

# bcftools norm -m -both Nip_42_rice.vcfwave.vcf -o Nip_42_rice.vcfwave.biallelic.vcf
# python3 annotate_vcf.py Nip_42_rice.vcfwave.biallelic.vcf > Nip_42_rice.vcfwave.biallelic.annotated.vcf  

python3 extract_variants.py
python3 calc_density.py
python3 visualize_chromosome_density_v2.py
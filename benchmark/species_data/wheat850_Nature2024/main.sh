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

# Phenotype IDs follow Wheat_All_traits_Matrix.xlsx (All_Data IID).
# Keep only samples that have at least one trait and are present in the LD VCF.
if python3 -c "import pandas" >/dev/null 2>&1; then
    PY=python3
else
    PY=/data4/gulei/anaconda3/bin/python
fi
"$PY" - "$P" <<'PY'
import subprocess
import sys
from pathlib import Path

import pandas as pd

prefix = sys.argv[1]
vcf_in = Path(f"{prefix}.LD.vcf.gz")
pheno_xlsx = Path("Wheat_All_traits_Matrix.xlsx")
pheno_out = Path(f"{prefix}.pheno")

if not Path(str(vcf_in) + ".tbi").exists() and not Path(str(vcf_in) + ".csi").exists():
    subprocess.check_call(["bcftools", "index", "--csi", "--force", str(vcf_in)])

vcf_ids = set(
    subprocess.check_output(["bcftools", "query", "-l", str(vcf_in)], text=True).splitlines()
)

pheno = pd.read_excel(pheno_xlsx, sheet_name="All_Data")
pheno = pheno.rename(columns={pheno.columns[0]: "LINE"})
pheno["LINE"] = pheno["LINE"].astype(str)
trait_cols = [c for c in pheno.columns if c != "LINE"]
n_xlsx = len(pheno)
pheno = pheno.loc[pheno[trait_cols].notna().any(axis=1)].copy()
n_with_trait = len(pheno)
pheno = pheno.loc[pheno["LINE"].isin(vcf_ids)].copy()
pheno.to_csv(pheno_out, sep="\t", index=False, na_rep="NA")

print(f"[INFO] xlsx samples: {n_xlsx}")
print(f"[INFO] with >=1 trait: {n_with_trait}")
print(f"[INFO] also in {vcf_in}: {len(pheno)}")
print(f"[INFO] wrote {pheno_out}")
PY

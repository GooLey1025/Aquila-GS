beagle.nf --snp-vcf Tomato706.merge_snp.vcf.gz --nthreads 32 --memory "320 GB" -resume

P=Tomato706
plink2   --vcf results/Tomato706.merge_snp.impute.biallelic.vcf.gz --make-bed   --out $P.impute
plink2 \
    --bfile ${P}.impute \
    --snps-only just-acgt \
    --set-all-var-ids '@:#:$r:$a' \
    --make-pgen \
    --out ${P}.SNP \
    --memory 220000

plink2   --pfile $P.SNP   --indep-pairwise 1000 50 0.0001   --out $P.LD --memory 220000
plink2   --pfile $P.SNP   --extract $P.LD.prune.in   --recode vcf bgz   --out $P.LD

# SL### in the VCF corresponds to TS-N in Tomato539_pheno.xlsx (SL001 -> TS-1).
# Rename genotype samples to TS-* and keep only samples that exist in both files.
if python3 -c "import pandas" >/dev/null 2>&1; then
    PY=python3
else
    PY=/data4/gulei/anaconda3/bin/python
fi
"$PY" - "$P" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

prefix = sys.argv[1]
vcf_in = Path(f"{prefix}.LD.vcf.gz")
vcf_out = Path(f"{prefix}.LD.ts.vcf.gz")
map_path = Path("SL_to_TS.sample_map")
pheno_xlsx = Path("Tomato539_pheno.xlsx")
pheno_out = Path(f"{prefix}.pheno")

vcf_samples = subprocess.check_output(
    ["bcftools", "query", "-l", str(vcf_in)], text=True
).splitlines()
pat = re.compile(r"^SL(\d+)$")
rows = []
for sid in vcf_samples:
    match = pat.match(sid)
    if match is None:
        raise SystemExit(f"Unexpected VCF sample ID: {sid}")
    rows.append((sid, f"TS-{int(match.group(1))}"))
map_path.write_text("".join(f"{old}\t{new}\n" for old, new in rows))
vcf_ts_ids = {new for _, new in rows}

subprocess.check_call(
    ["bcftools", "reheader", "-s", str(map_path), "-o", str(vcf_out), str(vcf_in)]
)
subprocess.check_call(["bcftools", "index", "--tbi", "--force", str(vcf_out)])
vcf_out.replace(vcf_in)
tbi = Path(str(vcf_out) + ".tbi")
tbi.replace(Path(str(vcf_in) + ".tbi"))

pheno = pd.read_excel(pheno_xlsx, sheet_name="Final_539lines_AveTraits")
pheno = pheno.rename(columns={"Accession": "LINE"}).drop(columns=["No"])
trait_cols = [c for c in pheno.columns if c != "LINE"]
pheno = pheno.loc[pheno[trait_cols].notna().any(axis=1)].copy()
pheno["LINE"] = pheno["LINE"].astype(str)
n_before = len(pheno)
pheno = pheno.loc[pheno["LINE"].isin(vcf_ts_ids)].copy()
pheno.to_csv(pheno_out, sep="\t", index=False, na_rep="NA")

print(f"[INFO] renamed {len(rows)} VCF samples SL* -> TS-*")
print(f"[INFO] pheno kept {len(pheno)} / {n_before} samples present in {vcf_in}")
print(f"[INFO] wrote {pheno_out} and {map_path}")
PY

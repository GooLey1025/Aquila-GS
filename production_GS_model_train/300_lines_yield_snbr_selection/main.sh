#!/usr/bin/env bash
# Rice300 yield and neck-blast selection, plus leaf-blast importance labels.
#
# Run from any directory, in the aquila environment:
#   bash main.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
MODEL_DIR=../Rice3305.aquila-snp.multi_trait.per-fold.10folds.hpo
VCF=Rice300.Rice3305_ref.snp.format.id.impute.biallelic.vcf.gz
META=../300_lines_test-set/300lines_map.txt
PRED=predictions/aquila-snp.10fold-mean.300rice.tsv
BLUP=Yield_SNBR_BLUP_values.tsv
IMPORTANCE=../../case/gwas_interpretability/Rice300.leafblast.gwas_interpretability.ScaleOfLeafBlastResistance_BLUP.ensemble.position_importance/importance_ranking_ScaleOfLeafBlastResistance_BLUP.tsv
RICENAVI=../../evolve/ricenavi_319site_evaluation.csv

mkdir -p predictions/folds

echo "[predict] 10-fold mean"
aquila_predict_ensemble.py \
  --model-dir "$MODEL_DIR" \
  --vcf "$VCF" \
  --encoding-type diploid_onehot \
  --id-prefix SNP- \
  --device cuda:0 \
  --keep-fold-predictions predictions/folds \
  -o "$PRED"

echo "[table] predicted yield and neck-blast BLUPs"
"$PYTHON" - "$PRED" "$META" "$BLUP" << 'PY'
import sys
import pandas as pd

pred_path, meta_path, out_path = sys.argv[1:]
pred = pd.read_csv(pred_path, sep="\t")
# Latest Rice3305 multi-trait model is trained on the BLUP columns themselves.
yield_col = "YieldPerPlant_BLUP_Pred"
snbr_col = "ScaleOfNeckBlastResistance_BLUP_Pred"
missing = [c for c in (yield_col, snbr_col) if c not in pred.columns]
if missing:
    raise SystemExit(f"Missing predicted BLUP columns: {missing}")

meta = pd.read_csv(meta_path, sep="\t")
meta["Sample_ID"] = meta["Sample_ID"].astype(str)
out = pred[["Sample_ID"]].copy()
out["YieldPerPlant_BLUP"] = pred[yield_col]
out["ScaleOfNeckBlastResistance_BLUP"] = pred[snbr_col]
out = out.merge(meta[["Sample_ID", "Subspecies", "Cultivar_Name"]], on="Sample_ID", how="left")
out = out.rename(columns={"Subspecies": "Subpopulation"})
if out["Subpopulation"].isna().any():
    missing_ids = out.loc[out["Subpopulation"].isna(), "Sample_ID"].tolist()
    raise SystemExit(f"Missing subspecies for: {missing_ids[:5]}")
out.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
print(f"Wrote {len(out)} rows to {out_path}")
PY

echo "[plot] yield and neck-blast selection"
"$PYTHON" plot_yield_snbr_blup.py

echo "[plot] leaf-blast importance with QTN markers"
"$PYTHON" plot_neck_blast_importance_riceqtn.py \
  --importance "$IMPORTANCE" \
  --ricenavi "$RICENAVI" \
  -o LeafBlastResistance_BLUP_RiceNavi_gene_label

echo "[plot] leaf-blast importance with QTN names only"
"$PYTHON" plot_neck_blast_importance_riceqtn.py \
  --importance "$IMPORTANCE" \
  --ricenavi "$RICENAVI" \
  --no-marker \
  --ymax 0.25 \
  -o LeafBlastResistance_BLUP_RiceNavi_gene_label.names_only

echo "[table] blast QTN genotypes for 300 lines"
"$PYTHON" write_blast_qtn_supp.py

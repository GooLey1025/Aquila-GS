#!/usr/bin/env bash
# Average the 10 Aquila-SNP seed IG rankings, then draw the ensemble
# GWAS comparison figures.
#
# Each seed is divided by its own mean importance before the seeds are
# averaged. Raw IG magnitudes differ by more than an order of magnitude
# across initializations even when the regression outputs stay on the same
# scale, so an unnormalized mean would follow only the largest seeds.
set -euo pipefail

export WORK_DIR=./ig_10seeds_work
export GWAS_DIR=./Rice655.full_marker_panel_gwas_results/association_results
export PREFIX=655rice.r2_0.045.aquila-snp
export OUT_DIR=./ig_ensemble
export PNG_DIR=./ig_png_ensemble
SEEDS=(42 43 44 45 46 47 48 49 50 51)
TRAITS=(HD_BLUP GW_BLUP PH_BLUP)
QTN_ANNOT=Final_summary_347_QTNsites_geno_redefined.xlsx
MARKERS=./655rice.canonical_markers.tsv

mkdir -p "$OUT_DIR" "$PNG_DIR"

python3 - "$WORK_DIR" "$OUT_DIR" "$PREFIX" "${SEEDS[@]}" -- "${TRAITS[@]}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

argv = sys.argv[1:]
separator = argv.index("--")
work_dir, out_dir, prefix, *seed_args = argv[:separator]
traits = argv[separator + 1 :]
seeds = [int(seed) for seed in seed_args]
out_root = Path(out_dir)
out_root.mkdir(parents=True, exist_ok=True)

for trait in traits:
    frames = []
    for seed in seeds:
        path = (
            Path(work_dir)
            / f"seed_{seed}"
            / f"{prefix}.seed_{seed}.{trait}.position_importance"
            / f"importance_ranking_{trait}.tsv"
        )
        if not path.is_file():
            raise SystemExit(f"Missing importance ranking: {path}")
        frame = pd.read_csv(path, sep="\t")
        required = {"locus_id", "importance", "locus_index"}
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
        scores = frame["importance"].to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise SystemExit(f"{path} contains non-finite importance")
        scale = float(scores.mean())
        if scale <= 0:
            raise SystemExit(f"{path} has a non-positive mean importance")
        frames.append(
            pd.DataFrame(
                {
                    "locus_id": frame["locus_id"].astype(str),
                    "locus_index": frame["locus_index"].astype(int),
                    "normalized": scores / scale,
                    "seed": seed,
                }
            )
        )
        print(f"[scale] {trait} seed {seed}: mean={scale:.6g} loci={len(frame)}")

    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby("locus_id")["seed"].nunique()
    if int(counts.min()) != len(seeds) or int(counts.max()) != len(seeds):
        raise SystemExit(f"{trait} loci are not shared by every seed")
    index_counts = combined.groupby("locus_id")["locus_index"].nunique()
    if int(index_counts.max()) != 1:
        raise SystemExit(f"{trait} locus_index disagrees across seeds")

    summary = (
        combined.groupby(["locus_id", "locus_index"], sort=False)
        .agg(
            importance_mean=("normalized", "mean"),
            importance_std=("normalized", lambda values: float(np.std(values, ddof=1))),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        ["importance_mean", "locus_id"], ascending=[False, True]
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.insert(2, "importance", summary["importance_mean"])
    trait_dir = out_root / trait
    trait_dir.mkdir(parents=True, exist_ok=True)
    output = trait_dir / f"importance_ranking_{trait}.tsv"
    summary.to_csv(output, sep="\t", index=False)
    print(
        f"[mean] {trait}: {len(summary)} loci, "
        f"top={summary.iloc[0]['locus_id']} "
        f"mean={summary.iloc[0]['importance_mean']:.4f} "
        f"-> {output}"
    )
PY

for TRAIT in "${TRAITS[@]}"; do
  IMPORTANCE="${OUT_DIR}/${TRAIT}/importance_ranking_${TRAIT}.tsv"
  PLOT_PDF="${OUT_DIR}/${PREFIX}.ensemble.${TRAIT}.gwas_ig.pdf"
  echo "[plot] ${TRAIT}"
  python3 gwas_ig_multi_plot_v3.py \
    --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
    --importance "$IMPORTANCE" \
    -o "$PLOT_PDF" \
    --also-png \
    --smooth 5 \
    --ig-top-k 500 \
    --qtn-annot "$QTN_ANNOT" \
    --annot-tolerance-kb 500
  cp -f "${PLOT_PDF%.pdf}.png" "$PNG_DIR/"

  PLOT_PDF="${OUT_DIR}/${PREFIX}.ensemble.${TRAIT}.gwas_ig.no_qtn.pdf"
  echo "[plot] ${TRAIT} without QTN labels"
  python3 gwas_ig_multi_plot_v3.py \
    --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
    --importance "$IMPORTANCE" \
    -o "$PLOT_PDF" \
    --also-png \
    --smooth 5 \
    --ig-top-k 500
  cp -f "${PLOT_PDF%.pdf}.png" "$PNG_DIR/"
done

TRAIT=GW_BLUP
IMPORTANCE="${OUT_DIR}/${TRAIT}/importance_ranking_${TRAIT}.tsv"
LOCUS_PDF="${OUT_DIR}/${PREFIX}.ensemble.${TRAIT}.chr3-16733441.gwas_ig.pdf"
echo "[plot] ${TRAIT} chr3:16733441"
python3 gwas_ig_multi_plot_v3.py \
  --gwas "${GWAS_DIR}/${TRAIT}.gemma_lmm.assoc.txt" \
  --importance "$IMPORTANCE" \
  -o "$LOCUS_PDF" \
  --also-png \
  --smooth 5 \
  --ig-top-k 500 \
  --qtn-annot "$QTN_ANNOT" \
  --highlight 3:16733441
cp -f "${LOCUS_PDF%.pdf}.png" "$PNG_DIR/"

echo "Ensemble rankings: ${OUT_DIR}"
echo "Ensemble PNGs: ${PNG_DIR}"

echo "[upset] HD_BLUP 10-seed ensemble"
python3 plot_topk_gwas_marker_venn_v3.py \
  --importance "${OUT_DIR}/HD_BLUP/importance_ranking_HD_BLUP.tsv" \
  --gwas \
    "${GWAS_DIR}/HD_BeiJ15.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/HD_BLUP.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/HD_LingS15.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/HD_LingS16.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/HD_WenJ15.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/HD_YangZ15.gemma_lmm.assoc.txt" \
  --markers "$MARKERS" \
  --trait "Heading date related" \
  --top-k 500 \
  -o "${OUT_DIR}/${PREFIX}.ensemble.HD_BLUP.top500.upset.pdf" \
  --summary-tsv "${OUT_DIR}/${PREFIX}.ensemble.HD_BLUP.top500.source_percentage.tsv"

echo "[upset] GW_BLUP 10-seed ensemble"
python3 plot_topk_gwas_marker_venn_v3.py \
  --importance "${OUT_DIR}/GW_BLUP/importance_ranking_GW_BLUP.tsv" \
  --gwas \
    "${GWAS_DIR}/GW_BeiJ15.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/GW_BLUP.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/GW_LingS16.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/GW_WenJ15.gemma_lmm.assoc.txt" \
    "${GWAS_DIR}/GW_YangZ15.gemma_lmm.assoc.txt" \
  --markers "$MARKERS" \
  --trait "Grain width related" \
  --top-k 500 \
  -o "${OUT_DIR}/${PREFIX}.ensemble.GW_BLUP.top500.upset.pdf" \
  --summary-tsv "${OUT_DIR}/${PREFIX}.ensemble.GW_BLUP.top500.source_percentage.tsv"

echo "Ensemble UpSet plots and summaries: ${OUT_DIR}"

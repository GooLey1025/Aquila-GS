#!/usr/bin/env bash
# Plot the temporal distribution of predicted GYP_BLUP for the 300 rice lines.
#
# Inputs:
#   ../300_lines_test-set/300lines_map.txt
#   ../300_lines_test-set/predictions/aquila-snp.10fold-mean.300rice.tsv
#   ../SuppData_X_Rice300_ForFigure7ABC.xlsx
#     Figure column. Fig. 7C is the balanced 120-line heatmap panel.
#
# Breeding periods are inclusive release-year bins:
#   1976–2000, 2001–2010, 2011–2020, 2021–2024
#
# Run from any directory:
#   bash main.sh
set -euo pipefail
cd "$(dirname "$0")"

SOURCE_DIR=../300_lines_test-set
META="${SOURCE_DIR}/300lines_map.txt"
PRED="${SOURCE_DIR}/predictions/aquila-snp.10fold-mean.300rice.tsv"
PANEL=../SuppData_X_Rice300_ForFigure7ABC.xlsx
OUTPUT=GYP_BLUP_stage_by_subpop_pseudo_group_violin.pdf

for input in "$META" "$PRED" "$PANEL"; do
  if [[ ! -f "$input" ]]; then
    echo "Required input not found: $input" >&2
    exit 1
  fi
done

python3 plot_inbred_line_growth_subpop.py \
  --meta "$META" \
  --pred "$PRED" \
  --trait GYP_BLUP \
  --output "$OUTPUT"

python3 Inbred_120_heatmap.py \
  --meta "$META" \
  --pred "$PRED" \
  --panel "$PANEL" \
  --figure "Fig. 7C" \
  --output GYP_4env_heatmap_by_subpop_stage_sorted_horizontal_bwstage.pdf

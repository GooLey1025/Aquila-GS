#!/usr/bin/env bash
set -euo pipefail

TRAITS=(GYP_BLUP HD_BLUP PH_BLUP)
FOLDS=(0 1 2 3 4)
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEM_ROOT="$BENCHMARK_ROOT/DEM"

cd "$DEM_ROOT"

for FOLD in "${FOLDS[@]}"; do
  python DEM_train_benchmark.py \
    --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
    --config configs/DEM-SNP_nested_cv.yaml \
    --traits "${TRAITS[@]}" \
    --outer-folds "$FOLD" \
    -o "results/Rice655-rf-matched/DEM-SNP/fold_${FOLD}" \
    --overwrite

  python DEM_train_benchmark.py \
    --data-dir "../Rice655.current_fold_GWAS.vars.cv.data/fold_${FOLD}" \
    --config configs/DEM-Vars_nested_cv.yaml \
    --traits "${TRAITS[@]}" \
    --outer-folds "$FOLD" \
    -o "results/Rice655-rf-matched/DEM-Vars/fold_${FOLD}" \
    --overwrite

  for TRAIT in "${TRAITS[@]}"; do
    python export_dem_rf_aquila_data.py \
      --data-dir "../Rice655.current_fold_GWAS.cv.data/fold_${FOLD}" \
      --dem-output-dir "results/Rice655-rf-matched/DEM-SNP/fold_${FOLD}" \
      --trait "$TRAIT" \
      --outer-fold "$FOLD" \
      -o "../Rice655.DEM-RF.Aquila-SNP.data/${TRAIT}/fold_${FOLD}" \
      --overwrite

    python export_dem_rf_aquila_data.py \
      --data-dir "../Rice655.current_fold_GWAS.vars.cv.data/fold_${FOLD}" \
      --dem-output-dir "results/Rice655-rf-matched/DEM-Vars/fold_${FOLD}" \
      --trait "$TRAIT" \
      --outer-fold "$FOLD" \
      -o "../Rice655.DEM-RF.Aquila-Vars.data/${TRAIT}/fold_${FOLD}" \
      --overwrite
  done
done

cd "$BENCHMARK_ROOT/aquila-snp"
for TRAIT in "${TRAITS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    aquila_train_cv.py \
      --data-dir "../Rice655.DEM-RF.Aquila-SNP.data/${TRAIT}/fold_${FOLD}" \
      --config "configs/dem_rf_matched.${TRAIT}.yaml" \
      -o "results/Rice655-rf-matched/${TRAIT}/fold_${FOLD}" \
      --overwrite
  done
done

cd "$BENCHMARK_ROOT/aquila-vars"
for TRAIT in "${TRAITS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    aquila_train_cv.py \
      --data-dir "../Rice655.DEM-RF.Aquila-Vars.data/${TRAIT}/fold_${FOLD}" \
      --config "configs/dem_rf_matched.${TRAIT}.yaml" \
      -o "results/Rice655-rf-matched/${TRAIT}/fold_${FOLD}" \
      --overwrite
  done
done

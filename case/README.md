## Exmaple: Train a final model
```sh
yaml=705rice_conv_mha.aquila-snp.hpo.yaml

aquila_train_multi.py --config params/$yaml --n-folds 5 --n-seeds 10 -o ${yaml%.hpo.yaml}.best_fold_find

BEST_DS=$(python parse_best_fold.py ${yaml%.hpo.yaml}.best_fold_find/results_summary.tsv --print-only-path)

aquila_train_hpo.py --config params/$yaml -o ${yaml%.yaml} -dsf $BEST_DS > ${yaml%.yaml}.log 2>&1

BEST_TRIAL=$(awk -F': ' '/Best trial number/ {print $2}' ${yaml%.yaml}/optuna_summary.txt)

cp -rf ${yaml%.yaml}/trial_$BEST_TRIAL ${yaml%.hpo.yaml}.best_model
```

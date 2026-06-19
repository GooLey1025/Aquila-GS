
python3 aquila_evolve.py --model-dir 3485rice_conv_mha.aquila-snp.best_model \
  --vcf Teqing__SAMN04505840.snp.impute.biallelic.vcf.gz \
  --pheno ScaleOfNeckBlastResistance_Jinzhai_2023 --strategy combinatorial \
  --iterations 200 --mode minimize --output-vcf Teqing__SAMN04505840.snp.impute.biallelic_evolve.vcf.gz \
  --output-dir Teqing__SAMN04505840.snp.impute.biallelic_evolve

python3 sliding_divergence.py \
  --vcf1 Teqing__SAMN04505840.snp.impute.biallelic.vcf.gz \
  --vcf2 Huanghuazhan.snp.impute.biallelic.vcf.gz \
  --out_prefix Teqing_vs_Huanghuazhan --top_pct 0.3

python3 sliding_divergence.py \
  --vcf1 Teqing__SAMN04505840.snp.impute.biallelic.vcf.gz \
  --vcf2 Teqing__SAMN04505840.snp.impute.biallelic_evolve.vcf.gz \
  --out_prefix Teqing_vs_TeqingEvolve --top_pct 0.3

python3 plot_region.py
python3 plot_snp_density_diff.py
python3 plot_combos_generations.py

python plot_pred_vs_true_facets.py \
  -i tq_hhz.1171rice.phenos.csv \
  -o tq_hhz.1171rice.pred_vs_true.pdf

# python3 aquila_evolve.py --model-dir 3485rice_conv_mha.aquila-snp.best_model \
#   --vcf Teqing__SAMN04505840.snp.impute.biallelic.vcf.gz \
#   --pheno ScaleOfNeckBlastResistance_Jinzhai_2023 --strategy combinatorial \
#   --iterations 200 --mode minimize --output-vcf Teqing__SAMN04505840.snp.impute.biallelic_evolve.vcf.gz \
#   --output-dir Teqing__SAMN04505840.snp.impute.biallelic_evolve

## 1171rice - multi-phenotype SI with per-trait direction
python3 aquila_evolve.py --model-dir 1171rice_conv_mha.aquila-snp.best_model \
  --vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --direction-file teqing_to_hhz.evolve_direction.tsv \
  --output-dir 1171rice_SI_evolve --strategy combinatorial --iterations 20000 \
  --save-all-rounds

python plot_evolution_trait_heatmap.py \
  --pred 1171rice_SI_evolve/round_predictions.tsv \
  --direction teqing_to_hhz.evolve_direction.tsv \
  --out 1171rice_teqing_to_hhz_trait_evolution_heatmap

python3 compute_relative_snp_density.py \
  --vcf-baseline Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --vcf-sample1 1171rice_SI_evolve/Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz \
  --vcf-sample2 Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --fai Nip.chrnum.sorted.fa.fai \
  --label-sample1 TeqingEvolve \
  --label-sample2 HHZhan \
  --window-bp 1000000 \
  -o Teqing__SAMN04505840.1171rice_snp_density_per_mb.tsv

python3 plot_snp_density_diff_v2.py --density-tsv	Teqing__SAMN04505840.1171rice_snp_density_per_mb.tsv --karyotype	karyotype.txt \
  --ricenavi ricenavi_319site_evaluation.csv \
  --output-prefix	Teqing__SAMN04505840.1171rice_snp_divergence_landscape

# python3 calc_round_ibs.py -r 1171rice_SI_evolve/round_vcf --target-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz -o Teqing.evolve.1171rice.round_IBS_convergence.tsv

# python plot_ibs_convergence.py \
#     -i Teqing.evolve.1171rice.round_IBS_convergence.tsv \
#     -o Teqing_to_HHZ_IBS

# bash run_round_pca.sh 1171rice_SI_evolve/round_vcf 1171rice_SI_evolve_merged_rounds

## Case1
DIRECTION_FILE=teqing_to_hhz.evolve_direction_2.tsv
aquila_evolve.py --model-dir 1171rice_conv_mha.aquila-snp.best_model \
  --vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz  \
  --direction-file $DIRECTION_FILE  \
  --seed 42 \
  --sites-to-evolve ricenavi_gatk3_gatk4_snp_indel_id.list  \
  --output-dir 1171rice_SI_screening_evolve --strategy screening --save-all-rounds --homozygous
python plot_evolution_trait_heatmap.py \
  --pred 1171rice_SI_screening_evolve/round_predictions.tsv \
  --direction $DIRECTION_FILE --round-step 1\
  --out 1171rice_SI_screening_teqing_to_hhz_trait_evolution_heatmap
python3 plot_screening_gain_QTN.py  \
  --screening-file 1171rice_SI_screening_evolve/screening_si_per_round.tsv \
  --ricenavi-file ricenavi_319site_evaluation.csv  \
  --teqing-vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --evolved-vcf 1171rice_SI_screening_evolve/Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz   --out-prefix QTN_gain_contribution   --top-n 20

bash run_n_aquila_evolve.sh 100 $DIRECTION_FILE Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz 1171rice_SI_screening_evolve 1171rice_conv_mha.aquila-snp.best_model True
python plot_all_seed_round_predictions.py \
  --evolve-dir 1171rice_SI_screening_evolve \
  --out 1171rice_SI_all_traits_round_predictions.png
python plot_trait_change_summary.py   --evolve-dir 1171rice_SI_screening_evolve   --direction teqing_to_hhz.evolve_direction_2.tsv   --out 1171rice_SI_trait_change_summary
python plot_qtn_gain_seed_mean_panel.py \
  --evolve-dir 1171rice_SI_screening_evolve \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --teqing-vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --out-prefix 1171rice_seed_mean_QTN_gain

# Case2 Failed.
DIRECTION_FILE=hhz_as_base.evolve_direction.tsv
aquila_evolve.py --model-dir 1171rice_conv_mha.aquila-snp.best_model \
  --vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz  \
  --direction-file $DIRECTION_FILE  \
  --seed 42 \
  --sites-to-evolve ricenavi_gatk3_gatk4_snp_indel_id.list  \
  --output-dir 1171rice_HHZ_SI_screening_evolve --strategy screening --save-all-rounds --homozygous
python plot_evolution_trait_heatmap.py \
  --pred 1171rice_HHZ_SI_screening_evolve/round_predictions.tsv \
  --direction $DIRECTION_FILE --round-step 1\
  --out 1171rice_HHZ_SI_screening_teqing_to_hhz_trait_evolution_heatmap
python3 plot_screening_gain_QTN.py  \
  --screening-file 1171rice_HHZ_SI_screening_evolve/screening_si_per_round.tsv \
  --ricenavi-file ricenavi_319site_evaluation.csv  \
  --teqing-vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --evolved-vcf 1171rice_SI_screening_evolve/Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz   --out-prefix QTN_gain_contribution   --top-n 20

bash run_n_aquila_evolve.sh 100 $DIRECTION_FILE Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz 1171rice_HHZ_SI_screening_evolve 1171rice_conv_mha.aquila-snp.best_model
python plot_all_seed_round_predictions.py \
  --evolve-dir 1171rice_HHZ_SI_screening_evolve \
  --out 1171rice_HHZ_SI_all_traits_round_predictions
python plot_trait_change_summary.py   --evolve-dir 1171rice_HHZ_SI_screening_evolve   --direction $DIRECTION_FILE   --out 1171rice_HHZ_SI_trait_change_summary
python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir 1171rice_HHZ_SI_screening_evolve \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name Huanghuazhan.1171rice.snp.impute.biallelic_evolve.vcf.gz \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --out-prefix 1171rice_seed_mean_QTN_gain_HHZ-base

# Case3
DIRECTION_FILE=hhz_base_705rice.directions.tsv
LABEL=maximize-minimize
aquila_evolve.py --model-dir 705rice_conv_mha.aquila-snp.best_model \
  --vcf huanghuazhan.705rice.snp.vcf.gz  \
  --direction-file $DIRECTION_FILE  \
  --seed 42 \
  --sites-to-evolve ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list  \
  --output-dir 705rice_HHZ_${LABEL}_SI_screening_evolve --strategy screening --save-all-rounds --homozygous
bash run_n_aquila_evolve.sh 100 $DIRECTION_FILE huanghuazhan.705rice.snp.vcf.gz 705rice_HHZ_${LABEL}_SI_screening_evolve 705rice_conv_mha.aquila-snp.best_model True
python plot_all_seed_round_predictions.py \
  --evolve-dir 705rice_HHZ_${LABEL}_SI_screening_evolve \
  --out 705rice_HHZ_${LABEL}_SI_all_traits_round_predictions
python plot_trait_change_summary.py   --task-mapping 705rice_conv_mha.aquila-snp.best_model/task_mapping.tsv --evolve-dir 705rice_HHZ_${LABEL}_SI_screening_evolve   --direction $DIRECTION_FILE   --out 705rice_HHZ_${LABEL}_SI_trait_change_summary
python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir 705rice_HHZ_${LABEL}_SI_screening_evolve \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name huanghuazhan.705rice.snp_evolve.vcf.gz \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf huanghuazhan.705rice.snp.vcf.gz \
  --out-prefix 705rice_${LABEL}_seed_mean_QTN_gain_HHZ-base

# Case4; for genome wide
DIRECTION_FILE=hhz_base_705rice.directions.tsv

aquila_evolve.py --model-dir 705rice_conv_mha.aquila-snp.best_model \
  --vcf Huanghuazhan.705rice.snp.impute.biallelic.vcf.gz  \
  --direction-file $DIRECTION_FILE  \
  --seed 42 \
  --sites-to-evolve ricenavi_gatk3_gatk4_snp_indel_id.list  \
  --output-dir 705rice_HHZ_SI_screening_evolve --strategy screening --save-all-rounds --homozygous
bash run_n_aquila_evolve.sh 10 $DIRECTION_FILE huanghuazhan.705rice.snp.vcf.gz 705rice_HHZ_GW_SI_screening_evolve 705rice_conv_mha.aquila-snp.best_model False
python plot_all_seed_round_predictions.py \
  --evolve-dir 705rice_HHZ_GW_SI_screening_evolve \
  --out 705rice_HHZ_GW_SI_all_traits_round_predictions
python plot_trait_change_summary.py   --task-mapping 705rice_conv_mha.aquila-snp.best_model/task_mapping.tsv --evolve-dir 705rice_HHZ_GW_SI_screening_evolve   --direction $DIRECTION_FILE   --out 705rice_HHZ_GW_SI_trait_change_summary
python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir 705rice_HHZ_GW_SI_screening_evolve \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name huanghuazhan.705rice.snp_evolve.vcf.gz \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf huanghuazhan.705rice.snp.vcf.gz \
  --out-prefix 705rice_seed_mean_QTN_gain_HHZ-base-GW --all-sites

python3 calc_round_ibs.py -r 1171rice_SI_screening_evolve/round_vcf --target-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz -o Teqing.screening.evolve.1171rice.round_IBS_convergence.tsv
python plot_ibs_convergence.py \
    -i Teqing.screening.evolve.1171rice.round_IBS_convergence.tsv \
    -o Teqing_screening_to_HHZ_IBS

python plot_genomic_allele_shift.py \
  --teqing-vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --evolved-vcf 1171rice_SI_screening_evolve/Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz \
  --karyotype karyotype.txt \
  --qtn ricenavi_319site_evaluation.csv \
  -o Teqing_Evolved_HHZ_genomic_shift

python3 plot_si_gain_vs_gwas_diff.py \
  --si-gain 705rice_seed_mean_QTN_gain_HHZ-base-GW.all_sites_gain.tsv \
  --gwas-ling 705rice_gwas_results/GYP_LingS15/gemma_lmm.assoc.txt \
  --gwas-yang 705rice_gwas_results/GYP_YangZ15/gemma_lmm.assoc.txt \
  -o SI_gain_vs_GYP_GWAS_diff.pdf

aquila_evolve.py --model-dir 1171rice_conv_mha.aquila-snp.best_model \
  --vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz --top-k 8 \
  --direction-file teqing_to_hhz.evolve_direction.tsv --sites-to-evolve ricenavi_gatk3_gatk4_snp_id.list \
  --output-dir 1171rice_QTN_SI_evolve --strategy combinatorial --iterations 200 \
  --save-all-rounds --save-site-gain

python plot_evolution_trait_heatmap.py \
  --pred 1171rice_QTN_SI_evolve/round_predictions.tsv \
  --direction teqing_to_hhz.evolve_direction_2.tsv --round-step 1\
  --out 1171rice_QTN_teqing_to_hhz_trait_evolution_heatmap
python3 plot_screening_gain_QTN.py  \
  --screening-file 1171rice_QTN_SI_evolve/screening_si_per_round.tsv \
  --ricenavi-file ricenavi_319site_evaluation.csv  \
  --teqing-vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --evolved-vcf 1171rice_SI_screening_evolve/Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz   --out-prefix QTN_gain_contribution   --top-n 20


python plot_genomic_allele_shift.py \
  --teqing-vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --evolved-vcf 1171rice_QTN_SI_evolve/round_vcf/round500.vcf.gz \
  --karyotype karyotype.txt \
  --qtn ricenavi_319site_evaluation.csv \
  -o Teqing_QTN_Evolved_HHZ_genomic_shift

aquila_evolve_multi.py \
    --model-dir 1171rice_conv_mha.aquila-snp.best_model \
    --vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz \
    --direction-file teqing_to_hhz.evolve_direction.tsv \
    --sites-to-evolve ricenavi_gatk3_gatk4_snp_id.list \
    --output-dir 1171rice_QTN_multi_evolve_200 \
    --strategy combinatorial \
    --iterations 200 \
    --n-seeds 100 \
    --qtn-list ricenavi_gatk3_gatk4_snp_id.list

python stat_evolved_sites.py  \
  --parent-vcf  Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz  \
  --evolved-vcf 1171rice_QTN_multi_evolve_200/merged_evolved.vcf.gz  \
  --hhz-vcf Huanghuazhan.1171rice.snp.impute.biallelic.vcf.gz \
  --site-list ricenavi_gatk3_gatk4_snp_id.list   --evaluation-csv ricenavi_319site_evaluation.csv   --out-prefix Teqing_200seed_evolved_sites

python plot_evolved_bar_with_gene.py \
  --input Teqing_200seed_evolved_sites.with_gene.hhz_compare.tsv \
  --out-prefix Teqing_200seed_evolved_sites \
  --metric CEF \
  --top-n 200

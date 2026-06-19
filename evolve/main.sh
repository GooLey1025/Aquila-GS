## Case1: Teqing-to-HHZ
DIRECTION_FILE=teqing_to_hhz.evolve_direction_2.tsv
aquila_evolve.py --model-dir 1171rice_conv_mha.aquila-snp.best_model \
  --vcf Teqing__SAMN04505840.1171rice.snp.impute.biallelic.vcf.gz  \
  --direction-file $DIRECTION_FILE  \
  --seed 42 \
  --sites-to-evolve ricenavi_gatk3_gatk4_snp_indel_id.list  \
  --output-dir 1171rice_SI_screening_evolve --strategy screening --save-all-rounds --homozygous

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

## Case2: HHZ-base
DIRECTION_FILE=hhz_base_705rice.directions.tsv
LABEL=maximize-minimize
aquila_evolve.py --model-dir 705rice_conv_mha.aquila-snp.best_model \
  --vcf huanghuazhan.705rice.snp.vcf.gz  \
  --direction-file $DIRECTION_FILE  \
  --seed 42 \
  --sites-to-evolve ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list  \
  --output-dir 705rice_HHZ_${LABEL}_SI_screening_evolve --strategy screening --save-all-rounds --homozygous
bash run_n_aquila_evolve.sh 100 $DIRECTION_FILE huanghuazhan.705rice.snp.vcf.gz 705rice_HHZ_${LABEL}_SI_screening_evolve 705rice_conv_mha.aquila-snp.best_model True

python plot_trait_change_summary.py   --task-mapping 705rice_conv_mha.aquila-snp.best_model/task_mapping.tsv --evolve-dir 705rice_HHZ_${LABEL}_SI_screening_evolve   --direction $DIRECTION_FILE   --out 705rice_HHZ_${LABEL}_SI_trait_change_summary
python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir 705rice_HHZ_${LABEL}_SI_screening_evolve \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name huanghuazhan.705rice.snp_evolve.vcf.gz \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf huanghuazhan.705rice.snp.vcf.gz \
  --out-prefix 705rice_${LABEL}_seed_mean_QTN_gain_HHZ-base

# Other

python plot_evolved_bar_with_gene.py \
  --input Teqing_200seed_evolved_sites.with_gene.hhz_compare.tsv \
  --out-prefix Teqing_200seed_evolved_sites \
  --metric CEF \
  --top-n 200

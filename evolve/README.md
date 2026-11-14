# QTN Evolution

Run all commands below from the `evolve/` directory. This workflow evolves the RiceNavi QTNs themselves. It does not build a proxy map. Each resolved QTN is screened on the current genome, and the homozygous flip is kept only when the 10-fold ensemble selection index increases.

## Data preparation
```sh
PANEL_VCF=4lines.Rice655_ref.snp.indel.sv.impute.biallelic.vcf.gz
QTN_LIST=ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list
MODEL_DIR=../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo
```

#### Case 1: Yield changes from South China to North China.
```bash
SAMPLE=D2295 # D2785
PARENT_VCF=${SAMPLE}.parent.vcf.gz
DIRECTION_FILE=gyp_south_to_north.directions.tsv
EVOLVE_DIR=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_evolve
TRAIT_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_trait_change_summary
GAIN_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_si_gain
NULL_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_recommendation_null
```

#### Case 2: Yield changes from North China to South China.
```bash
SAMPLE=D341 # D309
PARENT_VCF=${SAMPLE}.parent.vcf.gz
DIRECTION_FILE=gyp_north_to_south.directions.tsv
EVOLVE_DIR=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_evolve
TRAIT_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_trait_change_summary
GAIN_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_si_gain
NULL_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_recommendation_null
```

#### Case 3: Huanghuazhan base.

Extract `Huanghuazhan__SAMN04505852` from the joint Huanghuazhan–Teqing VCF. The QTN list is the previous Huanghuazhan list with sterility sites removed. The direction file is the south-to-north file: maximize `GYP_YangZ15` and minimize `GYP_LingS15`. Run the shared commands below with these variables. The model remains the 10-fold ensemble set in data preparation.

```bash
PANEL_VCF=HuangHuaZhan_Teqing_Rice655-ref.snp.indel.sv.impute.biallelic.vcf.gz
QTN_LIST=ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list
SAMPLE=Huanghuazhan__SAMN04505852
PARENT_VCF=${SAMPLE}.parent.vcf.gz
DIRECTION_FILE=gyp_south_to_north.directions.tsv
EVOLVE_DIR=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_evolve
TRAIT_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_trait_change_summary
GAIN_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_si_gain
NULL_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_recommendation_null
```

### Evolve!
```bash
bcftools view \
  --samples "$SAMPLE" \
  --output-type z \
  --output-file "$PARENT_VCF" \
  "$PANEL_VCF"

bcftools index --force "$PARENT_VCF"

mkdir -p "${EVOLVE_DIR}/logs"

for seed in $(seq 1 100); do
  aquila_qtn_evolve.py \
    --model-dir "$MODEL_DIR" \
    --vcf "$PARENT_VCF" \
    --qtn-list "$QTN_LIST" \
    --direction-file "$DIRECTION_FILE" \
    --strategy screening \
    --homozygous \
    --save-all-rounds \
    --seed "${seed}" \
    --output-dir "${EVOLVE_DIR}/seed_${seed}" \
    > "${EVOLVE_DIR}/logs/seed_${seed}.log" 2>&1
done

python plot_trait_change_summary.py \
  --evolve-dir "$EVOLVE_DIR" \
  --direction "$DIRECTION_FILE" \
  --out "$TRAIT_PREFIX"

python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir "$EVOLVE_DIR" \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name "${SAMPLE}.parent_evolve.vcf.gz" \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf "$PARENT_VCF" \
  --out-prefix "$GAIN_PREFIX"
```

Each seed shuffles the order in which the same QTN list is screened. The ensemble prediction for a given genome is deterministic, so the 100 runs differ only by that order. The trait summary and the SI-gain panel report the mean and standard deviation across these 100 orders. The genotype after optimization is the majority genotype across orders.

QTN list IDs such as `SNP-1-178766-1` are matched to the parent VCF by exact ID, then by a unique chromosome and position. The resolved markers are written to `${EVOLVE_DIR}/seed_*/qtn_sites.tsv`. Unmatched or ambiguous QTNs are recorded there and are not evolved.

### (Optional) Direction null

Keep the parent and the QTN list fixed. Each permutation draws two regression traits that are not targets in `$DIRECTION_FILE`, assigns one to `maximize` and one to `minimize`, and reruns screening at seed 42. The recorded value is the reported direction's selection index: the signed fold-0 Z change of those target traits. The red line is the mean of that index across the 100 screening orders in `$EVOLVE_DIR`. The histogram is the same index under random directions, and the shaded band is the central 95% of the null. The empirical P is `(1 + number of random directions at least as large) / (N_NULL + 1)`.

```bash
N_NULL=36
DIRECTION_NULL_DIR=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.direction_null
DIRECTION_NULL_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.direction_null
mkdir -p "${DIRECTION_NULL_DIR}/logs"

python prepare_qtn_direction_null.py \
  --preprocessing "${MODEL_DIR}/fold_0/preprocessing.json" \
  --direction-file "$DIRECTION_FILE" \
  --n-directions "$N_NULL" \
  --seed 42 \
  --out-dir "$DIRECTION_NULL_DIR"

for perm in $(seq 1 "$N_NULL"); do
  id=$(printf '%02d' "$perm")
  aquila_qtn_evolve.py \
    --model-dir "$MODEL_DIR" \
    --vcf "$PARENT_VCF" \
    --qtn-list "$QTN_LIST" \
    --direction-file "${DIRECTION_NULL_DIR}/perm_${id}.directions.tsv" \
    --strategy screening \
    --homozygous \
    --seed 42 \
    --output-dir "${DIRECTION_NULL_DIR}/perm_${id}" \
    > "${DIRECTION_NULL_DIR}/logs/perm_${id}.log" 2>&1
done

python plot_qtn_direction_null.py \
  --evolve-dir "$EVOLVE_DIR" \
  --null-dir "$DIRECTION_NULL_DIR" \
  --direction-file "$DIRECTION_FILE" \
  --preprocessing "${MODEL_DIR}/fold_0/preprocessing.json" \
  --out-prefix "$DIRECTION_NULL_PREFIX"
```

## Teqing-to-Huanghuazhan

This case uses `teqing_to_hhz.evolve_direction_2.tsv` with the 10-fold production model `../production_GS_model_train/1171rice.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo`. Minimize heading date, plant height, amylose content, chalkiness, and grain translucency; maximize seed setting rate, gel consistency, and grain shape; leave the other traits in that file neutral. Extract `Teqing__SAMN04505840` and `Huanghuazhan__SAMN04505852` from `HuangHuaZhan_Teqing_Rice1171-ref.snp.indel.sv.impute.biallelic.vcf.gz`, which has the same 17,875 SNPs as `1171rice.final.imputed.panel.vcf.gz`. The SI-gain panel shows the Teqing genotype, the Huanghuazhan genotype, and the evolved majority genotype, and labels whether the evolved genotype matches Huanghuazhan, moves toward it, or matches neither parent.

```bash
PANEL_VCF=HuangHuaZhan_Teqing_Rice1171-ref.snp.indel.sv.impute.biallelic.vcf.gz
QTN_LIST=ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list
MODEL_DIR=../production_GS_model_train/1171rice.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo
SAMPLE=Teqing__SAMN04505840
PARENT_VCF=${SAMPLE}.1171panel.parent.vcf.gz
HHZ_SAMPLE=Huanghuazhan__SAMN04505852
HHZ_VCF=${HHZ_SAMPLE}.1171panel.parent.vcf.gz
DIRECTION_FILE=teqing_to_hhz.evolve_direction_2.tsv
EVOLVE_DIR=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_evolve
TRAIT_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_trait_change_summary
GAIN_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_si_gain
DIRECTION_NULL_DIR=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.direction_null
DIRECTION_NULL_PREFIX=qtn_evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.direction_null

bcftools view \
  --samples "$SAMPLE" \
  --output-type z \
  --output-file "$PARENT_VCF" \
  "$PANEL_VCF"
bcftools index --force "$PARENT_VCF"

bcftools view \
  --samples "$HHZ_SAMPLE" \
  --output-type z \
  --output-file "$HHZ_VCF" \
  "$PANEL_VCF"
bcftools index --force "$HHZ_VCF"

mkdir -p "${EVOLVE_DIR}/logs"

for seed in $(seq 1 100); do
  aquila_qtn_evolve.py \
    --model-dir "$MODEL_DIR" \
    --vcf "$PARENT_VCF" \
    --qtn-list "$QTN_LIST" \
    --direction-file "$DIRECTION_FILE" \
    --strategy screening \
    --homozygous \
    --save-all-rounds \
    --seed "${seed}" \
    --output-dir "${EVOLVE_DIR}/seed_${seed}" \
    > "${EVOLVE_DIR}/logs/seed_${seed}.log" 2>&1
done

python plot_trait_change_summary.py \
  --evolve-dir "$EVOLVE_DIR" \
  --direction "$DIRECTION_FILE" \
  --out "$TRAIT_PREFIX"

python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir "$EVOLVE_DIR" \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name "${SAMPLE}.1171panel.parent_evolve.vcf.gz" \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf "$PARENT_VCF" \
  --hhz-vcf "$HHZ_VCF" \
  --out-prefix "$GAIN_PREFIX"

N_NULL=36
mkdir -p "${DIRECTION_NULL_DIR}/logs"

python prepare_qtn_direction_null.py \
  --preprocessing "${MODEL_DIR}/fold_0/preprocessing.json" \
  --direction-file "$DIRECTION_FILE" \
  --n-directions "$N_NULL" \
  --seed 42 \
  --out-dir "$DIRECTION_NULL_DIR"

for perm in $(seq 1 "$N_NULL"); do
  id=$(printf '%02d' "$perm")
  aquila_qtn_evolve.py \
    --model-dir "$MODEL_DIR" \
    --vcf "$PARENT_VCF" \
    --qtn-list "$QTN_LIST" \
    --direction-file "${DIRECTION_NULL_DIR}/perm_${id}.directions.tsv" \
    --strategy screening \
    --homozygous \
    --seed 42 \
    --output-dir "${DIRECTION_NULL_DIR}/perm_${id}" \
    > "${DIRECTION_NULL_DIR}/logs/perm_${id}.log" 2>&1
done

python plot_qtn_direction_null.py \
  --evolve-dir "$EVOLVE_DIR" \
  --null-dir "$DIRECTION_NULL_DIR" \
  --direction-file "$DIRECTION_FILE" \
  --preprocessing "${MODEL_DIR}/fold_0/preprocessing.json" \
  --out-prefix "$DIRECTION_NULL_PREFIX"
```

# (Ignore., Experimental version) QTN Proxy-Guided Evolution 

Run all commands below from the `evolve/` directory. The workflow has three stages:

1. Extract `Nanjingxiangzhan` as a single-sample parent from the large 300-rice VCF.
2. Use the 10-fold ensemble to map each RiceNavi QTN to a proxy within ±500 kb.
3. Run multi-round directed evolution with the fixed mapping while recalculating each proxy's SI effect in the current genomic background.

### Data preparation
```sh
PANEL_VCF=4lines.Rice655_ref.snp.indel.sv.impute.biallelic.vcf.gz
QTN_LIST=ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list
MODEL_DIR=../production_GS_model_train/655rice.production.more_sv_markers.final.imputed.panel.aquila-snp.multi_trait.per-fold.10folds.hpo
```

#### Case 1: Yield changes from South China to North China.
```bash
SAMPLE=D2295 # D2785
PARENT_VCF=${SAMPLE}.parent.vcf.gz
DIRECTION_FILE=gyp_south_to_north.directions.tsv
PROXY_DIR=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_proxy
EVOLVE_DIR=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.proxy_evolve
TRAIT_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.trait_change_summary
GAIN_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.proxy_si_gain
NULL_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.recommendation_null
```

#### Case 2: Yield changes from North China to South China.
```bash
SAMPLE=D341 # D309
PARENT_VCF=${SAMPLE}.parent.vcf.gz
DIRECTION_FILE=gyp_north_to_south.directions.tsv
PROXY_DIR=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_proxy
EVOLVE_DIR=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.proxy_evolve
TRAIT_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.trait_change_summary
GAIN_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.proxy_si_gain
NULL_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.recommendation_null
```

#### Case 3: Huanghuazhan base.

Extract `Huanghuazhan__SAMN04505852` from the joint Huanghuazhan–Teqing VCF. The QTN list is the previous Huanghuazhan list with sterility sites removed. The direction file is the south-to-north file: maximize `GYP_YangZ15` and minimize `GYP_LingS15`. Run the shared commands below with these variables. The model remains the 10-fold ensemble set in data preparation.

```bash
PANEL_VCF=HuangHuaZhan_Teqing_Rice655-ref.snp.indel.sv.impute.biallelic.vcf.gz
QTN_LIST=ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list
SAMPLE=Huanghuazhan__SAMN04505852
PARENT_VCF=${SAMPLE}.parent.vcf.gz
DIRECTION_FILE=gyp_south_to_north.directions.tsv
PROXY_DIR=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.qtn_proxy
EVOLVE_DIR=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.proxy_evolve
TRAIT_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.trait_change_summary
GAIN_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.proxy_si_gain
NULL_PREFIX=evolve_results/${SAMPLE}/${SAMPLE}_${DIRECTION_FILE%.tsv}.recommendation_null
```

### Evolve!
```bash
bcftools view \
  --samples "$SAMPLE" \
  --output-type z \
  --output-file "$PARENT_VCF" \
  "$PANEL_VCF"

bcftools index --force "$PARENT_VCF"

aquila_qtn_proxy_map.py \
  --model-dir "$MODEL_DIR" \
  --vcf "$PARENT_VCF" \
  --qtn-list "$QTN_LIST" \
  --direction-file "$DIRECTION_FILE" \
  --window-kb 500 \
  --min-cosine 0.8 \
  --cosine-modes maximize,minimize,maintain \
  --si-epsilon 1e-4 \
  --batch-size 16 \
  --output-dir "$PROXY_DIR"

# Run multi-round evolution with the fixed proxy mapping
aquila_evolve_ensemble.py \
  --model-dir "$MODEL_DIR" \
  --vcf "$PARENT_VCF" \
  --direction-file "$DIRECTION_FILE" \
  --qtn-proxy-map "${PROXY_DIR}/qtn_proxy_map.json" \
  --strategy screening \
  --homozygous \
  --save-all-rounds \
  --seed 42 \
  --output-dir "$EVOLVE_DIR"

python plot_trait_change_summary.py \
  --evolve-dir "$EVOLVE_DIR" \
  --direction "$DIRECTION_FILE" \
  --out "$TRAIT_PREFIX"

python plot_qtn_gain_seed_mean_panel_for_web.py \
  --evolve-dir "$EVOLVE_DIR" \
  --screening-name screening_si_per_round.tsv \
  --evolved-vcf-name "${SAMPLE}.parent_evolve.vcf.gz" \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --base-vcf "$PARENT_VCF" \
  --out-prefix "$GAIN_PREFIX"
```

### (Optional) Recommendation null

Compare the model-preferred homozygote at each QTN with the RiceNavi favorable allele. The preferred homozygote is held fixed, and curated favorable labels are permuted within trait. The histogram is that null distribution of match counts; the red line is the observed number of matches, and the shaded band is the central 95% of the null. QTNs without a model homozygote, and QTNs whose curated label is neutral or ambiguous, are excluded and listed in the audit table. The test quantifies whether the allele overlap exceeds this within-trait chance level. It does not test the size of the SI gains, and it is not a validation in an independent population.

```bash
python plot_qtn_recommendation_null.py \
  --qtn-proxy-map "${PROXY_DIR}/qtn_proxy_map.tsv" \
  --ricenavi-file ricenavi_319site_evaluation.csv \
  --out-prefix "$NULL_PREFIX"
```



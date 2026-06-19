#!/usr/bin/env bash

set -euo pipefail

n_seeds="$1"
direction_file="$2"
vcf_file="$3"
output_dir="$4"
model_dir="$5"
ricenavi_switch="$6"

gpu_ids=(0 1 2)
jobs_per_gpu=2
max_jobs=$(( ${#gpu_ids[@]} * jobs_per_gpu ))

common_args=(
    --model-dir "${model_dir}"
    --vcf "${vcf_file}"
    --direction-file "${direction_file}"
    --strategy screening
    --save-all-rounds
    --homozygous
)

if [[ "${ricenavi_switch,,}" == "true" ]]; then
    common_args+=(
        --sites-to-evolve
        ricenavi_gatk3_gatk4_snp_indel_id.exclude_sterility.list
    )
fi

mkdir -p "${output_dir}/logs"

run_one_seed() {
    local seed="$1"
    local gpu_id="$2"

    echo "Running seed ${seed} on GPU ${gpu_id}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" aquila_evolve.py \
        "${common_args[@]}" \
        --seed "${seed}" \
        --output-dir "${output_dir}/seed_${seed}" \
        > "${output_dir}/logs/seed_${seed}.log" 2>&1
}

for seed in $(seq 1 "${n_seeds}"); do

    slot_index=$(( (seed - 1) % max_jobs ))
    gpu_index=$(( slot_index / jobs_per_gpu ))
    gpu_id="${gpu_ids[$gpu_index]}"

    run_one_seed "${seed}" "${gpu_id}" &

    if (( seed % max_jobs == 0 )); then
        wait
    fi

done

wait

echo "All seeds finished."

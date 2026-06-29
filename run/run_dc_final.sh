#!/usr/bin/env bash

set -euo pipefail

# Resolve all project paths from this script, so it can be started from any
# working directory. The final MEDIC implementation lives under train/MEDIC
# and test/MEDIC, unlike the legacy paths previously used here.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/data/seanoh/anaconda3/envs/dirl/bin/python}"
GPU_ID="${GPU_ID:-7}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] Python executable not found: $PYTHON" >&2
    exit 1
fi

function run_experiment() {
    local lr=$1
    local weight_decay=$2
    local bsize=$3
    local lambda_disentangle=$4
    local lambda_consistency=$5
    local lambda_cls=$6
    local lambda_type=$7
    local cls_temperature=$8
    local type_temperature=$9
    local max_iter=50000

    run_name="lr_${lr}-wd_${weight_decay}-b_size_${bsize}-dis_${lambda_disentangle}-con_${lambda_consistency}-cls_${lambda_cls}-type_${lambda_type}-cls_temp_${cls_temperature}-type_temp_${type_temperature}"
    exp_dir="${PROJECT_ROOT}/exp/medic_dc"
    snapshot=$max_iter

    args=(
        --cfg "${PROJECT_ROOT}/configs/dynamic/transformer_medic_dc.yaml"
        --gpu "$GPU_ID"
        --seed 1111
        --exp_dir "$exp_dir"
        --lr "$lr"
        --batch_size "$bsize"
        --weight_decay "$weight_decay"
        --run_name "$run_name"
        --max_iter "$max_iter"
        --lambda_disentangle "$lambda_disentangle"
        --lambda_consistency "$lambda_consistency"
        --lambda_cls "$lambda_cls"
        --lambda_type "$lambda_type"
        --cls_temperature "$cls_temperature"
        --type_temperature "$type_temperature"
    )

    echo "[INFO] Running: $run_name"
    "$PYTHON" "${PROJECT_ROOT}/train/MEDIC/train_dc_std_final.py" "${args[@]}"

    log_file="$exp_dir/$run_name/evaluate_results.log"
    for snapshot in $(seq 40000 2000 "$max_iter"); do
        "$PYTHON" "${PROJECT_ROOT}/test/MEDIC/test_dc_std_final.py" --cfg "${PROJECT_ROOT}/configs/dynamic/transformer_medic_dc.yaml" --snapshot "$snapshot" --gpu "$GPU_ID" --exp_dir "$exp_dir" --run_name "$run_name"

        results_dir="$exp_dir/$run_name/test_output_${snapshot}/captions"
        anno_path="/mnt/disk1/clevr_dc/total_change_captions_reformat.json"
        type_file_path="/mnt/disk1/clevr_dc/type_mapping_dc.json"

        echo "Run Name: $run_name at $snapshot" >> "$log_file"
        "$PYTHON" "${PROJECT_ROOT}/eval/evaluate_dc_std.py" --results_dir "$results_dir" --anno "$anno_path" --type_file "$type_file_path" >> "$log_file"
        echo "------------------------------------------------" >> "$log_file"
    done
}

trap 'echo "[INFO] run cancelled. running next experiment..."; continue_flag=true' SIGINT

for lr in 2e-04; do
    for weight_decay in 0.0; do
        for bsize in 128; do
            for lambda_disentangle in 0.1; do
                for lambda_consistency in 0.01; do
                    for lambda_cls in 0.1; do
                        for lambda_type in 0.1; do
                            for cls_temperature in 1; do
                                for type_temperature in 1; do
                                    continue_flag=false
                                    run_experiment "$lr" "$weight_decay" "$bsize" "$lambda_disentangle" "$lambda_consistency" "$lambda_cls" "$lambda_type" "$cls_temperature" "$type_temperature"
                                    if [ "$continue_flag" = true ]; then
                                        continue
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

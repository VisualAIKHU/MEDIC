#! /bin/bash

export PYTHONPATH=$(pwd)

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
    local seed=${10}
    local max_iter=30000

    run_name="lr_${lr}-wd_${weight_decay}-b_size_${bsize}-dis_${lambda_disentangle}-con_${lambda_consistency}-cls_${lambda_cls}-type_${lambda_type}-cls_temp_${cls_temperature}-type_temp_${type_temperature}-seed_${seed}"
    exp_dir="./exp/medic_chg"
    snapshot=$max_iter

    args=(
        --cfg configs/dynamic/transformer_medic_chg.yaml
        --gpu 0
        --exp_dir $exp_dir
        --lr $lr
        --batch_size $bsize
        --weight_decay $weight_decay
        --run_name $run_name
        --max_iter $max_iter
        --lambda_disentangle $lambda_disentangle
        --lambda_consistency $lambda_consistency
        --lambda_cls $lambda_cls
        --lambda_type $lambda_type
        --cls_temperature $cls_temperature
        --type_temperature $type_temperature
        --seed $seed
    )

    echo "[INFO] Running: $run_name"

    python train/train_chg_final.py "${args[@]}" || return 1

    log_file="$exp_dir/$run_name/evaluate_results.log"
    for snapshot in $(seq 30000 10000 $max_iter); do
        python test/test_chg_final.py --cfg configs/dynamic/transformer_medic_chg.yaml --snapshot $snapshot --gpu 0 --exp_dir $exp_dir --run_name $run_name || return 1

        results_dir="$exp_dir/$run_name/test_output_${snapshot}/captions"
        anno_path="/mnt/disk1/clevr_change/data/total_change_captions_reformat_old.json"
        type_file_path="/mnt/disk1/clevr_change/data/type_mapping.json"

        echo "Run Name: $run_name at $snapshot" >> $log_file
        python eval/evaluate_chg.py --results_dir $results_dir --anno $anno_path --type_file $type_file_path >> $log_file || return 1
        echo "------------------------------------------------" >> $log_file
    done
}

trap 'echo "[INFO] run cancelled. running next experiment..."; continue_flag=true' SIGINT

for lr in 2e-04; do
    for weight_decay in 0.0; do
        for bsize in 64; do
            for lambda_disentangle in 0.1; do
                for lambda_consistency in 0.1; do
                    for lambda_cls in 0.1; do
                        for lambda_type in 0.1; do
                            for cls_temperature in 1; do
                                for type_temperature in 1; do
                                    for seed in 1 2 3 4 5; do
                                        continue_flag=false
                                        run_experiment "$lr" "$weight_decay" "$bsize" "$lambda_disentangle" "$lambda_consistency" "$lambda_cls" "$lambda_type" "$cls_temperature" "$type_temperature" "$seed"
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
done

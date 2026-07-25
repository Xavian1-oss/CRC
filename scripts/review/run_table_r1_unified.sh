#!/usr/bin/env bash
# Table R1 — unified ladder G1–G5 × B1–B3, single protocol (--fix_seed 2021)
# Logs: logs/review/r1u_{g1,g2,g3,g4,g5}_b{1,2,3}/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

FIX_SEED=2021

COMMON=(
  --fix_seed "${FIX_SEED}"
  --crc_encoder_layers 3
  --crc_tau_quantile 0.5
  --crc_val_ratio 1.0
  --warm_start
  --use_gpu True
  --gpu 0
)

run_one() {
  local tag="$1"
  shift
  local log_dir="logs/review/${tag}"
  mkdir -p "$log_dir"
  echo "=== Running ${tag} @ $(date) ===" | tee "${log_dir}/run.log"
  python -u run.py "$@" 2>&1 | tee -a "${log_dir}/run.log"
  echo "=== Done ${tag} ===" | tee -a "${log_dir}/run.log"
}

run_bench() {
  local bench="$1"   # b1 b2 b3
  local gflag="$2"   # g1..g5
  local extra=("${@:3}")

  case "${bench}" in
    b1)
      run_one "r1u_${gflag}_${bench}" \
        --task_name long_term_forecast --is_training 1 \
        --model CRC --baseline_model DLinear \
        --model_id "review_r1u_${gflag}_${bench}" \
        --data ETTh1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
        --features M --seq_len 96 --label_len 48 --pred_len 96 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model 512 --e_layers 2 --d_layers 1 --factor 3 \
        "${COMMON[@]}" "${extra[@]}" \
        --des "r1u_${gflag}_${bench}"
      ;;
    b2)
      run_one "r1u_${gflag}_${bench}" \
        --task_name long_term_forecast --is_training 1 \
        --model CRC --baseline_model DLinear \
        --model_id "review_r1u_${gflag}_${bench}" \
        --data ETTh1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
        --features M --seq_len 96 --label_len 48 --pred_len 336 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model 512 --e_layers 2 --d_layers 1 --factor 3 \
        "${COMMON[@]}" "${extra[@]}" \
        --des "r1u_${gflag}_${bench}"
      ;;
    b3)
      run_one "r1u_${gflag}_${bench}" \
        --task_name long_term_forecast --is_training 1 \
        --model CRC --baseline_model TimesNet \
        --model_id "review_r1u_${gflag}_${bench}" \
        --data custom \
        --root_path ./dataset/weather/ --data_path weather.csv \
        --features M --seq_len 96 --label_len 48 --pred_len 96 \
        --enc_in 21 --dec_in 21 --c_out 21 \
        --d_model 32 --d_ff 32 --top_k 5 \
        --e_layers 2 --d_layers 1 --factor 3 \
        "${COMMON[@]}" "${extra[@]}" \
        --des "r1u_${gflag}_${bench}"
      ;;
    *) echo "Unknown bench ${bench}"; exit 1 ;;
  esac
}

for bench in b1 b2 b3; do
  run_bench "${bench}" g5
  run_bench "${bench}" g1 --crc_use_blend 0
  run_bench "${bench}" g2 --crc_ridge_gate_only 1
  run_bench "${bench}" g3 --crc_use_dir_gate 0
  run_bench "${bench}" g4 \
    --crc_use_mtl 0 --crc_use_stcl 0 --crc_use_minimal_priors 1
done

echo "Table R1 unified run finished. Verify: python scripts/review/verify_table_r1.py"

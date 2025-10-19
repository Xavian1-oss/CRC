#!/usr/bin/env bash
set -euo pipefail

# ================== 基础设置（全局缺省；可被后续覆盖） ==================
export CUDA_VISIBLE_DEVICES=0

SEQ_LEN=96
LABEL_LEN=48

ENC_IN=7
DEC_IN=7
C_OUT=7

EPOCHS=50
PATIENCE=7
BATCH=32
LR=1e-3

Q_VAL=3
TOP_K=5

DMODEL=64
NHEADS=8
ELAYERS=2
DLAYERS=1
DFF=2048
PATCH_LEN=16
FACTOR=1
MOVING_AVG=25

# USE_DTW=--use_dtw
INVERSE=''   # 若要还原尺度评估，设为 '--inverse'

# ================== 想批量跑的组合 ==================
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2)            # 例如：ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Exchange Traffic
PRED_LENS=(96 192 336 720)               # 例如：96 192 336 720
BASELINES=(PatchTST)          # 例如：PatchTST TimesNet DLinear Informer Autoformer

# ================== 数据集路径与通道数 ==================
dataset_setup () {
  local ds="$1"
  case "$ds" in
    ETTh1)       ROOT="./dataset/ETT-small/";     CSV="ETTh1.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    ETTh2)       ROOT="./dataset/ETT-small/";     CSV="ETTh2.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    ETTm1)       ROOT="./dataset/ETT-small/";     CSV="ETTm1.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    ETTm2)       ROOT="./dataset/ETT-small/";     CSV="ETTm2.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    Weather)     ROOT="./dataset/weather/";       CSV="weather.csv";      ENC_IN=21;  DEC_IN=21;  C_OUT=21;;
    Traffic)     ROOT="./dataset/traffic/";       CSV="traffic.csv";      ENC_IN=862; DEC_IN=862; C_OUT=862;;
    Electricity) ROOT="./dataset/electricity/";   CSV="electricity.csv";  ENC_IN=321; DEC_IN=321; C_OUT=321;;
    Exchange)    ROOT="./dataset/exchange_rate/"; CSV="exchange_rate.csv";ENC_IN=8;   DEC_IN=8;   C_OUT=8;;
    *) echo "未知数据集: $ds"; exit 1;;
  esac
}

# ================== 三层默认 + 查表覆盖 ==================
# 0) 全局缺省
declare -A GDEF=(
  [LR]="1e-4" [BATCH]="32" [PATIENCE]="3"
  [DMODEL]="512" [NHEADS]="8" [ELAYERS]="2" [DLAYERS]="1" [DFF]="2048"
  [PATCH_LEN]="16" [FACTOR]="1" [TOP_K]="5" [MOVING_AVG]="25"
)

# 1) baseline × dataset 覆盖（只写与全局不同的）
declare -A BD

BD["PatchTST|ETTh1|ELAYERS"]="1"
BD["PatchTST|ETTh1|DLAYERS"]="1"
BD["PatchTST|ETTh1|FACTOR"]="3"

BD["PatchTST|ETTh2|ELAYERS"]="3"
BD["PatchTST|ETTh2|DLAYERS"]="1"
BD["PatchTST|ETTh2|FACTOR"]="3"
BD["PatchTST|ETTh2|NHEADS"]="4"

BD["PatchTST|ETTm1|DLAYERS"]="1"
BD["PatchTST|ETTm1|FACTOR"]="3"

BD["PatchTST|ETTm2|DLAYERS"]="1"
BD["PatchTST|ETTm2|FACTOR"]="3"

BD["DLinear|ETTh1|ELAYERS"]="2"
BD["DLinear|ETTh1|DLAYERS"]="1"
BD["DLinear|ETTh1|FACTOR"]="3"

BD["TimeXer|ETTh1|FACTOR"]="3"

BD["TimeXer|ETTh2|FACTOR"]="3"
BD["TimeXer|ETTh2|DFF"]="1024"

# 2) baseline × dataset × horizon 覆盖（只写确实因 horizon 不同而变化的项）
declare -A BDH

BDH["PatchTST|ETTh1|96|NHEADS"]="2"
BDH["PatchTST|ETTh1|192|NHEADS"]="8"
BDH["PatchTST|ETTh1|336|NHEADS"]="8"
BDH["PatchTST|ETTh1|720|NHEADS"]="16"

BDH["PatchTST|ETTm1|96|ELAYERS"]="1"
BDH["PatchTST|ETTm1|96|NHEADS"]="2"
BDH["PatchTST|ETTm1|96|BATCH"]="32"

BDH["PatchTST|ETTm1|192|ELAYERS"]="3"
BDH["PatchTST|ETTm1|192|NHEADS"]="2"
BDH["PatchTST|ETTm1|192|BATCH"]="128"

BDH["PatchTST|ETTm1|336|ELAYERS"]="1"
BDH["PatchTST|ETTm1|336|NHEADS"]="4"
BDH["PatchTST|ETTm1|336|BATCH"]="128"

BDH["PatchTST|ETTm1|720|ELAYERS"]="3"
BDH["PatchTST|ETTm1|720|NHEADS"]="4"
BDH["PatchTST|ETTm1|720|BATCH"]="128"

BDH["PatchTST|ETTm2|96|ELAYERS"]="3"
BDH["PatchTST|ETTm2|96|NHEADS"]="16"
BDH["PatchTST|ETTm2|96|BATCH"]="32"

BDH["PatchTST|ETTm2|192|ELAYERS"]="3"
BDH["PatchTST|ETTm2|192|NHEADS"]="2"
BDH["PatchTST|ETTm2|192|BATCH"]="128"

BDH["PatchTST|ETTm2|336|ELAYERS"]="1"
BDH["PatchTST|ETTm2|336|NHEADS"]="4"
BDH["PatchTST|ETTm2|336|BATCH"]="32"

BDH["PatchTST|ETTm2|720|ELAYERS"]="3"
BDH["PatchTST|ETTm2|720|NHEADS"]="4"
BDH["PatchTST|ETTm2|720|BATCH"]="128"

BDH["TimeXer|ETTh1|96|ELAYERS"]="1"
BDH["TimeXer|ETTh1|96|DMODEL"]="256"
BDH["TimeXer|ETTh1|96|BATCH"]="4"

BDH["TimeXer|ETTh1|192|ELAYERS"]="2"
BDH["TimeXer|ETTh1|192|DMODEL"]="128"
BDH["TimeXer|ETTh1|192|BATCH"]="4"

BDH["TimeXer|ETTh1|336|ELAYERS"]="1"
BDH["TimeXer|ETTh1|336|DMODEL"]="512"
BDH["TimeXer|ETTh1|336|DFF"]="1024"
BDH["TimeXer|ETTh1|336|BATCH"]="16"

BDH["TimeXer|ETTh1|720|ELAYERS"]="1"
BDH["TimeXer|ETTh1|720|DMODEL"]="256"
BDH["TimeXer|ETTh1|720|DFF"]="1024"
BDH["TimeXer|ETTh1|720|BATCH"]="16"

BDH["TimeXer|ETTh2|96|ELAYERS"]="1"
BDH["TimeXer|ETTh2|96|DMODEL"]="256"
BDH["TimeXer|ETTh2|96|BATCH"]="16"

BDH["TimeXer|ETTh2|192|ELAYERS"]="1"
BDH["TimeXer|ETTh2|192|DMODEL"]="256"

BDH["TimeXer|ETTh2|336|ELAYERS"]="2"
BDH["TimeXer|ETTh2|336|DMODEL"]="512"
BDH["TimeXer|ETTh2|336|BATCH"]="4"

BDH["TimeXer|ETTh2|720|ELAYERS"]="2"
BDH["TimeXer|ETTh2|720|DMODEL"]="256"
BDH["TimeXer|ETTh2|720|BATCH"]="16"

# 3) 查表函数：horizon 覆盖 > (baseline,dataset) 覆盖 > 全局缺省
getp () {
  local base="$1" ds="$2" h="$3" key="$4"
  local v=""
  v="${BDH[$base|$ds|$h|$key]:-}"
  [[ -z "$v" ]] && v="${BD[$base|$ds|$key]:-}"
  [[ -z "$v" ]] && v="${GDEF[$key]}"
  echo "$v"
}

# ================== 日志 ==================
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/multi_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"
log() { echo "[$(date +'%F %T')] $*" | tee -a "${MASTER_LOG}"; }
trap 'ecode=$?; [ $ecode -ne 0 ] && log "❌ 脚本异常退出，退出码=${ecode}"; exit $ecode' EXIT

# ================== 单次运行 ==================
run_with_log () {
  local baseline="$1"   # PatchTST / TimesNet / ...
  local ds="$2"         # 数据集
  local pred="$3"       # 预测步长

  dataset_setup "${ds}"

  local data_arg="${ds}"
  if [[ "${ds}" == "Weather" || "${ds}" == "Traffic" || "${ds}" == "Exchange" || "${ds}" == "Electricity" ]]; then
    data_arg="custom"
  fi

  # —— 取这次要用的超参（查表合并） ——
  local LRv       ; LRv=$(getp "$baseline" "$ds" "$pred" "LR")
  local BATCHv    ; BATCHv=$(getp "$baseline" "$ds" "$pred" "BATCH")
  local PATIENCEv ; PATIENCEv=$(getp "$baseline" "$ds" "$pred" "PATIENCE")
  local DMODELv   ; DMODELv=$(getp "$baseline" "$ds" "$pred" "DMODEL")
  local NHEADSv   ; NHEADSv=$(getp "$baseline" "$ds" "$pred" "NHEADS")
  local ELAYERSv  ; ELAYERSv=$(getp "$baseline" "$ds" "$pred" "ELAYERS")
  local DLAYERSv  ; DLAYERSv=$(getp "$baseline" "$ds" "$pred" "DLAYERS")
  local DFFv      ; DFFv=$(getp "$baseline" "$ds" "$pred" "DFF")
  local PATCHLENv ; PATCHLENv=$(getp "$baseline" "$ds" "$pred" "PATCH_LEN")
  local FACTORv   ; FACTORv=$(getp "$baseline" "$ds" "$pred" "FACTOR")
  local TOP_Kv    ; TOP_Kv=$(getp "$baseline" "$ds" "$pred" "TOP_K")
  local MOVAVGv   ; MOVAVGv=$(getp "$baseline" "$ds" "$pred" "MOVING_AVG")

  local tag="${ds}_crc_${baseline,,}_pl${pred}_dm${DMODELv}"
  local log_file="${LOG_DIR}/${tag}.log"

  log "▶️  baseline=${baseline} / ds=${ds} / pred=${pred} (log: ${log_file})"

  # ------- 回显命令 -------
  {
    echo "CMD @ $(date +'%F %T')"
    cat <<EOF
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CRC \
  --baseline_model ${baseline} \
  --model_id ${tag} \
  --data ${data_arg} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${pred} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model ${DMODELv} \
  --n_heads ${NHEADSv} \
  --e_layers ${ELAYERSv} \
  --d_layers ${DLAYERSv} \
  --d_ff ${DFFv} \
  --patch_len "${PATCHLENv}" \
  --factor ${FACTORv} \
  --top_k ${TOP_Kv} \
  --moving_avg ${MOVAVGv} \
  --batch_size ${BATCHv} \
  --learning_rate ${LRv} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCEv} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW:-} ${INVERSE} \
  --q_val ${Q_VAL} \
  --des CRC_${baseline}_M_d${DMODELv}
EOF
  } | tee -a "${log_file}" >> "${MASTER_LOG}"

  # ------- 真正执行 -------
  set +e
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model CRC \
    --baseline_model "${baseline}" \
    --model_id "${tag}" \
    --data "${data_arg}" \
    --root_path "${ROOT}" \
    --data_path "${CSV}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${pred}" \
    --enc_in "${ENC_IN}" \
    --dec_in "${DEC_IN}" \
    --c_out "${C_OUT}" \
    --d_model "${DMODELv}" \
    --n_heads "${NHEADSv}" \
    --e_layers "${ELAYERSv}" \
    --d_layers "${DLAYERSv}" \
    --d_ff "${DFFv}" \
    --patch_len "${PATCHLENv}" \
    --factor "${FACTORv}" \
    --top_k "${TOP_Kv}" \
    --moving_avg "${MOVAVGv}" \
    --batch_size "${BATCHv}" \
    --learning_rate "${LRv}" \
    --train_epochs "${EPOCHS}" \
    --patience "${PATIENCEv}" \
    --use_gpu True \
    --gpu_type cuda \
    --gpu 0 \
    ${USE_DTW:-} ${INVERSE} \
    --q_val "${Q_VAL}" \
    --des "CRC_${baseline}_M_d${DMODELv}" \
    2> >(tee -a "${log_file}" >> "${MASTER_LOG}" >&2) \
    | tee -a "${log_file}" >> "${MASTER_LOG}"
  ecode=${PIPESTATUS[0]}
  set -e

  if [ $ecode -eq 0 ]; then
    log "✅ 完成：${ds} / ${baseline} / pred_len=${pred}"
  else
    log "❌ 失败：${ds} / ${baseline} / pred_len=${pred}，退出码=${ecode}（详见 ${log_file}）"
    return $ecode
  fi
}

# ================== 批量调度 ==================
for ds in "${DATASETS[@]}"; do
  for pl in "${PRED_LENS[@]}"; do
    for base in "${BASELINES[@]}"; do
      run_with_log "${base}" "${ds}" "${pl}"
    done
  done
done

log "🎉 全部批量运行完成。日志目录：${LOG_DIR}"

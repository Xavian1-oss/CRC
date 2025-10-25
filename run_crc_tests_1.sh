#!/usr/bin/env bash
set -euo pipefail

# ================== 基础设置 ==================
export CUDA_VISIBLE_DEVICES=0

# --- 以下参数为 GDEF 之外的全局设置 ---
Q_VAL=3
# USE_DTW=--use_dtw
INVERSE=''   # 若要还原尺度评估，设为 '--inverse'

# ================== 想批量跑的组合 ==================
# (只保留 WPMixer 和 ETTh1)
DATASETS=(Electricity Traffic)
PRED_LENS=(96 192 336)
BASELINES=(WPMixer)

# ================== 数据集路径与通道数 ==================
# (保留这个函数，以后可能跑 WPMixer on ETTh2)
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
# 0) 全局缺省 (与 run.py 的 argparse default 保持一致)
# (这个 GDEF 必须保留，WPMixer 依赖它获取 DLAYERS, NHEADS 等默认值)
declare -A GDEF=(
  [LR]="1e-4" [BATCH]="32" [PATIENCE]="3"
  [DMODEL]="512" [NHEADS]="8" [ELAYERS]="2" [DLAYERS]="1" [DFF]="2048"
  [PATCH_LEN]="16" [FACTOR]="1" [TOP_K]="5" [MOVING_AVG]="25"
  [SEQ_LEN]="96"
  [LABEL_LEN]="48"
  [EPOCHS]="50"
  [DROPOUT]="0.1"
  [LRADJ]="type1"
  [USE_AMP]=""
)

# 1) baseline × dataset 覆盖（只保留 WPMixer）
declare -A BD

# WPMixer@ETTh1：官方脚本的共同参数
BD["WPMixer|ETTh1|SEQ_LEN"]="512"
BD["WPMixer|ETTh1|LABEL_LEN"]="0"
BD["WPMixer|ETTh1|BATCH"]="256"
BD["WPMixer|ETTh1|EPOCHS"]="30"
BD["WPMixer|ETTh1|LRADJ"]="type3"
BD["WPMixer|ETTh1|PATIENCE"]="12"

BD["WPMixer|ETTh2|SEQ_LEN"]="512"
BD["WPMixer|ETTh2|LABEL_LEN"]="0"
BD["WPMixer|ETTh2|BATCH"]="256"
BD["WPMixer|ETTh2|EPOCHS"]="30"
BD["WPMixer|ETTh2|LRADJ"]="type3"
BD["WPMixer|ETTh2|PATIENCE"]="12"

BD["WPMixer|ETTm1|SEQ_LEN"]="512"
BD["WPMixer|ETTm1|LABEL_LEN"]="0"
BD["WPMixer|ETTm1|BATCH"]="256"
BD["WPMixer|ETTm1|EPOCHS"]="80"
BD["WPMixer|ETTm1|DROPOUT"]="0.4"
BD["WPMixer|ETTm1|PATCH_LEN"]="48"
BD["WPMixer|ETTm1|LRADJ"]="type3"
BD["WPMixer|ETTm1|PATIENCE"]="12"

BD["WPMixer|ETTm2|SEQ_LEN"]="512"
BD["WPMixer|ETTm2|LABEL_LEN"]="0"
BD["WPMixer|ETTm2|DMODEL"]="256"
BD["WPMixer|ETTm2|BATCH"]="256"
BD["WPMixer|ETTm2|EPOCHS"]="80"
BD["WPMixer|ETTm2|PATCH_LEN"]="48"
BD["WPMixer|ETTm2|LRADJ"]="type3"
BD["WPMixer|ETTm2|PATIENCE"]="12"

BD["WPMixer|Weather|SEQ_LEN"]="512"
BD["WPMixer|Weather|LABEL_LEN"]="0"
BD["WPMixer|Weather|EPOCHS"]="60"
BD["WPMixer|Weather|DROPOUT"]="0.4"
BD["WPMixer|Weather|LRADJ"]="type3"
BD["WPMixer|Weather|PATIENCE"]="12"

BD["WPMixer|Electricity|SEQ_LEN"]="512"
BD["WPMixer|Electricity|LABEL_LEN"]="0"
BD["WPMixer|Electricity|DMODEL"]="32"
BD["WPMixer|Electricity|EPOCHS"]="100"
BD["WPMixer|Electricity|LRADJ"]="type3"
BD["WPMixer|Electricity|PATIENCE"]="12"

BD["WPMixer|Traffic|SEQ_LEN"]="1200"
BD["WPMixer|Traffic|LABEL_LEN"]="0"
BD["WPMixer|Traffic|BATCH"]="16"
BD["WPMixer|Traffic|LRADJ"]="type3"
BD["WPMixer|Traffic|PATIENCE"]="12"

# 2) baseline × dataset × horizon 覆盖（只保留 WPMixer）
declare -A BDH

# WPMixer@ETTh1：按 horizon 变化的参数
BDH["WPMixer|ETTh1|96|LR"]="0.000242438"
BDH["WPMixer|ETTh1|96|DROPOUT"]="0.4"
BDH["WPMixer|ETTh1|96|DMODEL"]="256"

BDH["WPMixer|ETTh1|192|LR"]="0.000201437"
BDH["WPMixer|ETTh1|192|DROPOUT"]="0.05"
BDH["WPMixer|ETTh1|192|DMODEL"]="256"

BDH["WPMixer|ETTh1|336|LR"]="0.000132929"
BDH["WPMixer|ETTh1|336|DROPOUT"]="0.0"
BDH["WPMixer|ETTh1|336|DMODEL"]="256"

BDH["WPMixer|ETTh1|720|LR"]="0.000239762"
BDH["WPMixer|ETTh1|720|DROPOUT"]="0.2"
BDH["WPMixer|ETTh1|720|DMODEL"]="128"

BDH["WPMixer|ETTh2|96|LR"]="0.000466278"
BDH["WPMixer|ETTh2|96|DROPOUT"]="0.0"
BDH["WPMixer|ETTh2|96|DMODEL"]="256"

BDH["WPMixer|ETTh2|192|LR"]="0.000294929"
BDH["WPMixer|ETTh2|192|DROPOUT"]="0.0"
BDH["WPMixer|ETTh2|192|DMODEL"]="256"

BDH["WPMixer|ETTh2|336|LR"]="0.000617476"
BDH["WPMixer|ETTh2|336|DROPOUT"]="0.1"
BDH["WPMixer|ETTh2|336|DMODEL"]="128"

BDH["WPMixer|ETTh2|720|LR"]="0.000810205"
BDH["WPMixer|ETTh2|720|DROPOUT"]="0.4"
BDH["WPMixer|ETTh2|720|DMODEL"]="128"

BDH["WPMixer|ETTh2|720|DROPOUT"]="0.4"
BDH["WPMixer|ETTh2|720|DMODEL"]="128"

BDH["WPMixer|ETTm1|96|LR"]="0.001277976"
BDH["WPMixer|ETTm1|96|DMODEL"]="256"

BDH["WPMixer|ETTm1|192|LR"]="0.002415901"
BDH["WPMixer|ETTm1|192|DMODEL"]="128"

BDH["WPMixer|ETTm1|336|LR"]="0.001594735"
BDH["WPMixer|ETTm1|336|DMODEL"]="256"

BDH["WPMixer|ETTm1|720|LR"]="0.002011441"
BDH["WPMixer|ETTm1|720|DMODEL"]="128"

BDH["WPMixer|ETTm2|96|LR"]="0.00076587"
BDH["WPMixer|ETTm2|96|DROPOUT"]="0.4"

BDH["WPMixer|ETTm2|192|LR"]="0.000275775"
BDH["WPMixer|ETTm2|192|DROPOUT"]="0.2"

BDH["WPMixer|ETTm2|336|LR"]="0.000234608"
BDH["WPMixer|ETTm2|336|DROPOUT"]="0.4"

BDH["WPMixer|ETTm2|720|LR"]="0.001039536"
BDH["WPMixer|ETTm2|720|DROPOUT"]="0.4"

BDH["WPMixer|Weather|96|LR"]="0.000913333"
BDH["WPMixer|Weather|96|BATCH"]="32"
BDH["WPMixer|Weather|96|DMODEL"]="256"

BDH["WPMixer|Weather|192|LR"]="0.001379042"
BDH["WPMixer|Weather|192|BATCH"]="64"
BDH["WPMixer|Weather|192|DMODEL"]="128"

BDH["WPMixer|Weather|336|LR"]="0.000607991"
BDH["WPMixer|Weather|336|BATCH"]="32"
BDH["WPMixer|Weather|336|DMODEL"]="128"

BDH["WPMixer|Weather|720|LR"]="0.001470479"
BDH["WPMixer|Weather|720|BATCH"]="128"
BDH["WPMixer|Weather|720|DMODEL"]="128"

BDH["WPMixer|Electricity|96|LR"]="0.00328086"
BDH["WPMixer|Electricity|96|DROPOUT"]="0.1"

BDH["WPMixer|Electricity|192|LR"]="0.000493286"
BDH["WPMixer|Electricity|192|DROPOUT"]="0.1"

BDH["WPMixer|Electricity|336|LR"]="0.002505375"
BDH["WPMixer|Electricity|336|DROPOUT"]="0.2"

BDH["WPMixer|Electricity|720|LR"]="0.001977516"
BDH["WPMixer|Electricity|720|DROPOUT"]="0.1"

BDH["WPMixer|Traffic|96|LR"]="0.0010385"
BDH["WPMixer|Traffic|96|EPOCHS"]="60"
BDH["WPMixer|Traffic|96|DROPOUT"]="0.05"
BDH["WPMixer|Traffic|96|DMODEL"]="16"

BDH["WPMixer|Traffic|192|LR"]="0.000567053"
BDH["WPMixer|Traffic|192|EPOCHS"]="60"
BDH["WPMixer|Traffic|192|DROPOUT"]="0.05"
BDH["WPMixer|Traffic|192|DMODEL"]="32"

BDH["WPMixer|Traffic|336|LR"]="0.001026715"
BDH["WPMixer|Traffic|336|EPOCHS"]="50"
BDH["WPMixer|Traffic|336|DROPOUT"]="0.0"
BDH["WPMixer|Traffic|336|DMODEL"]="32"

BDH["WPMixer|Traffic|720|LR"]="0.001496217"
BDH["WPMixer|Traffic|720|EPOCHS"]="60"
BDH["WPMixer|Traffic|720|DROPOUT"]="0.05"
BDH["WPMixer|Traffic|720|DMODEL"]="32"

# 3) 查表函数：horizon 覆盖 > (baseline,dataset) 覆盖 > 全局缺省
# (这个函数是核心引擎，必须保留)
getp () {
  local base="$1" ds="$2" h="$3" key="$4"
  local v=""
  v="${BDH[$base|$ds|$h|$key]:-}"
  [[ -z "$v" ]] && v="${BD[$base|$ds|$key]:-}"
  [[ -z "$v" ]] && v="${GDEF[$key]}"
  echo "$v"
}

# ================== 日志 ==================
# (必须保留)
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/multi_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"
log() { echo "[$(date +'%F %T')] $*" | tee -a "${MASTER_LOG}"; }
trap 'ecode=$?; [ $ecode -ne 0 ] && log "❌ 脚本异常退出，退出码=${ecode}"; exit $ecode' EXIT

# ================== 单次运行 ==================
# (这个函数是核心引擎，必须保留，并且保留所有升级后的参数)
run_with_log () {
  local baseline="$1"   # PatchTST / TimesNet / ...
  local ds="$2"         # 数据集
  local pred="$3"       # 预测步长

  dataset_setup "${ds}"

  local data_arg="${ds}"
  if [[ "${ds}" == "Weather" || "${ds}" == "Traffic" || "${ds}" == "Exchange" || "${ds}" == "Electricity" ]]; then
    data_arg="custom"
  fi

  # —— 取这次要用的超参（查表合并） —— (已升级)
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
  local SEQLENv   ; SEQLENv=$(getp "$baseline" "$ds" "$pred" "SEQ_LEN")
  local LABELLENv ; LABELLENv=$(getp "$baseline" "$ds" "$pred" "LABEL_LEN")
  local EPOCHSv   ; EPOCHSv=$(getp "$baseline" "$ds" "$pred" "EPOCHS")
  local DROPOUTv  ; DROPOUTv=$(getp "$baseline" "$ds" "$pred" "DROPOUT")
  local LRADJv    ; LRADJv=$(getp "$baseline" "$ds" "$pred" "LRADJ")
  local USEAMPv   ; USEAMPv=$(getp "$baseline" "$ds" "$pred" "USE_AMP")

  local tag="${ds}_crc_${baseline,,}_pl${pred}_dm${DMODELv}"
  local log_file="${LOG_DIR}/${tag}.log"

  log "▶️  baseline=${baseline} / ds=${ds} / pred=${pred} (log: ${log_file})"

  # ------- 回显命令 ------- (已升级)
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
  --seq_len ${SEQLENv} \
  --label_len ${LABELLENv} \
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
  --train_epochs ${EPOCHSv} \
  --patience ${PATIENCEv} \
  --dropout ${DROPOUTv} \
  --lradj ${LRADJv} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW:-} ${INVERSE} \
  ${USEAMPv} \
  --q_val ${Q_VAL} \
  --des CRC_${baseline}_M_d${DMODELv}
EOF
  } | tee -a "${log_file}" >> "${MASTER_LOG}"

  # ------- 真正执行 ------- (已升级)
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
    --seq_len "${SEQLENv}" \
    --label_len "${LABELLENv}" \
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
    --train_epochs "${EPOCHSv}" \
    --patience "${PATIENCEv}" \
    --dropout "${DROPOUTv}" \
    --lradj "${LRADJv}" \
    --use_gpu True \
    --gpu_type cuda \
    --gpu 0 \
    ${USE_DTW:-} ${INVERSE} \
    ${USEAMPv} \
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
# (这个循环现在只会跑 WPMixer on ETTh1 的 4个 pred_len)
for ds in "${DATASETS[@]}"; do
  for pl in "${PRED_LENS[@]}"; do
    for base in "${BASELINES[@]}"; do
      run_with_log "${base}" "${ds}" "${pl}"
    done
  done
done

log "🎉 全部批量运行完成。日志目录：${LOG_DIR}"
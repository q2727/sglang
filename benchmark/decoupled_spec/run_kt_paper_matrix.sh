#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/qinchong/workspace/code/ktransformers-ssd-maint/.venv-qwen36-latest/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/home/qinchong/models/Qwen3.6-35B-A3B}"
DATASET_DIR="${DATASET_DIR:-/home/qinchong/workspace/experiments/qwen3coder_dflash5_100/matrix-v1-t256/datasets}"
MATRIX_ROOT="${MATRIX_ROOT:-${HOME}/workspace/experiments/qwen36_ssd_qwen35_draft/matrix-k7f4-t75d25-t256}"
LABEL="${LABEL:-ssd-qwen36-qwen35-k7-f4-t75-d25}"
GPU_ID="${GPU_ID:-0}"
TARGET_PORT="${TARGET_PORT:-30030}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

DATASETS=(
  gsm8k
  math500
  aime25
  mbpp
  humaneval
  lcb
  mt-bench
  alpaca
  arena-hard-v2
)

mkdir -p "${MATRIX_ROOT}/records"
exec 9>"${MATRIX_ROOT}/orchestrator.lock"
if ! flock -n 9; then
  echo "Another matrix orchestrator already owns ${MATRIX_ROOT}." >&2
  exit 1
fi

SERVER_RUN_DIR="${MATRIX_ROOT}/server"

stop_server() {
  if [[ -d "${SERVER_RUN_DIR}" ]]; then
    "${SCRIPT_DIR}/stop_kt_same_gpu.sh" "${SERVER_RUN_DIR}" || true
  fi
}
trap stop_server EXIT INT TERM

wait_for_gpu_free() {
  local i
  for ((i = 0; i < 3600; i++)); do
    if ! nvidia-smi -i "${GPU_ID}" \
      --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -q '[0-9]'; then
      return 0
    fi
    sleep 5
  done
  echo "GPU ${GPU_ID} did not become idle." >&2
  return 1
}

wait_for_gpu_free
RUN_DIR="${SERVER_RUN_DIR}" \
GPU_ID="${GPU_ID}" TARGET_PORT="${TARGET_PORT}" \
K=7 F=4 TARGET_SM_PERCENT=75 DRAFT_SM_PERCENT=25 \
KT_CPU_INFER=120 KT_THREADPOOL_COUNT=2 KT_NUM_GPU_EXPERTS=0 \
SGLANG_DECOUPLED_ENUM_WAIT_MS=200 \
SGLANG_ENABLE_DECOUPLED_ADAPTIVE_FANOUT=0 \
TARGET_DISABLE_OVERLAP_SCHEDULE=0 TARGET_STREAM_GATE=0 TARGET_DOORBELL=0 \
DRAFT_PAGE_SIZE=1 DRAFT_MAX_TOKENS=4096 \
"${SCRIPT_DIR}/launch_kt_same_gpu.sh"

warmup_requests=1
for dataset in "${DATASETS[@]}"; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_paper_matrix.py" \
    --url "http://127.0.0.1:${TARGET_PORT}" \
    --model "${TARGET_MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --dataset "${dataset}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --warmup-requests "${warmup_requests}" \
    --max-input-tokens 4096 \
    --request-timeout 300 \
    --apply-chat-template \
    --label "${LABEL}" \
    --output "${MATRIX_ROOT}/records/${dataset}.jsonl"
  warmup_requests=0
done

touch "${MATRIX_ROOT}/DONE"
trap - EXIT INT TERM
stop_server
echo "Matrix complete: ${MATRIX_ROOT}"

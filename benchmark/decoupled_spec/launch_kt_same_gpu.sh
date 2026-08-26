#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/qinchong/workspace/code/ktransformers-ssd-maint/.venv-qwen36-latest/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/home/qinchong/models/Qwen3.6-35B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/qinchong/models/Qwen3.5-0.8B}"
GPU_ID="${GPU_ID:-0}"
TARGET_PORT="${TARGET_PORT:-30030}"
DRAFT_PORT="${DRAFT_PORT:-30031}"
K="${K:-7}"
F="${F:-4}"
TARGET_SM_PERCENT="${TARGET_SM_PERCENT:-75}"
DRAFT_SM_PERCENT="${DRAFT_SM_PERCENT:-25}"
TARGET_MEM_FRACTION="${TARGET_MEM_FRACTION:-0.60}"
DRAFT_MEM_FRACTION="${DRAFT_MEM_FRACTION:-0.24}"
DRAFT_PAGE_SIZE="${DRAFT_PAGE_SIZE:-1}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-8192}"
DRAFT_MAX_TOKENS="${DRAFT_MAX_TOKENS:-4096}"
KT_CPU_INFER="${KT_CPU_INFER:-120}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-0}"
TARGET_DISABLE_OVERLAP_SCHEDULE="${TARGET_DISABLE_OVERLAP_SCHEDULE:-0}"
TARGET_STREAM_GATE="${TARGET_STREAM_GATE:-0}"

if (( K < 1 || F < 1 )); then
  echo "K and F must both be positive." >&2
  exit 2
fi
if (( TARGET_SM_PERCENT < 1 || DRAFT_SM_PERCENT < 1 || TARGET_SM_PERCENT + DRAFT_SM_PERCENT > 100 )); then
  echo "SM percentages must be positive and sum to at most 100." >&2
  exit 2
fi

BRANCH_ROWS=$(((K + 1) * F))
MIN_MAMBA_SLOTS=$((1 + K + BRANCH_ROWS + 2))
DRAFT_MAMBA_SLOTS="${DRAFT_MAMBA_SLOTS:-$((((MIN_MAMBA_SLOTS + 7) / 8) * 8))}"
MIN_DRAFT_MAX_REQUESTS="${MIN_MAMBA_SLOTS}"
if (( MIN_DRAFT_MAX_REQUESTS < 16 )); then
  DEFAULT_DRAFT_MAX_REQUESTS=16
else
  DEFAULT_DRAFT_MAX_REQUESTS=$((((MIN_DRAFT_MAX_REQUESTS + 7) / 8) * 8))
fi
DRAFT_MAX_REQUESTS="${DRAFT_MAX_REQUESTS:-${DEFAULT_DRAFT_MAX_REQUESTS}}"
if (( DRAFT_MAX_REQUESTS < MIN_DRAFT_MAX_REQUESTS )); then
  echo "DRAFT_MAX_REQUESTS=${DRAFT_MAX_REQUESTS} is smaller than the required ${MIN_DRAFT_MAX_REQUESTS}." >&2
  exit 2
fi
if (( DRAFT_MAMBA_SLOTS < MIN_MAMBA_SLOTS )); then
  echo "DRAFT_MAMBA_SLOTS=${DRAFT_MAMBA_SLOTS} is smaller than the required ${MIN_MAMBA_SLOTS}." >&2
  exit 2
fi

RUN_BASE="${RUN_BASE:-${HOME}/workspace/experiments/ssd_enumeration_kt_samegpu}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-k${K}-f${F}-t${TARGET_SM_PERCENT}-d${DRAFT_SM_PERCENT}}"
RUN_DIR="${RUN_DIR:-${RUN_BASE}/${RUN_ID}}"
MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/${USER}-ssd-enum-gpu${GPU_ID}-mps-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/${USER}-ssd-enum-gpu${GPU_ID}-mps-log}"
VERIFY_ENDPOINT="ipc:///tmp/${USER}-ssd-enum-gpu${GPU_ID}-verify"
DRAFT_ENDPOINT="ipc:///tmp/${USER}-ssd-enum-gpu${GPU_ID}-draft"
VERIFY_ENDPOINT_PATH="${VERIFY_ENDPOINT#ipc://}"
DRAFT_ENDPOINT_PATH="${DRAFT_ENDPOINT#ipc://}"
SOURCE_PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

TARGET_SCHEDULER_ARGS=()
if [[ "${TARGET_DISABLE_OVERLAP_SCHEDULE}" == "1" ]]; then
  TARGET_SCHEDULER_ARGS+=(--disable-overlap-schedule)
fi

mkdir -p "${RUN_DIR}" "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
ln -sfn "${RUN_DIR}" "${RUN_BASE}/latest"

if ss -ltn | grep -qE ":(${TARGET_PORT}|${DRAFT_PORT})[[:space:]]"; then
  echo "Target or draft port is already in use." >&2
  exit 1
fi
rm -f "${VERIFY_ENDPOINT_PATH}" "${DRAFT_ENDPOINT_PATH}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
nvidia-cuda-mps-control -d

cleanup_on_error() {
  local pid
  for file in "${RUN_DIR}/target.pid" "${RUN_DIR}/draft.pid"; do
    if [[ -f "${file}" ]]; then
      pid="$(<"${file}")"
      kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    fi
  done
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup_on_error ERR INT TERM

wait_for_ready_log() {
  local pid="$1" log="$2" role="$3" timeout_s="${4:-180}"
  local i
  for ((i = 0; i < timeout_s; i++)); do
    if grep -q "The server is fired up and ready to roll" "${log}" 2>/dev/null; then
      echo "${role} ready."
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${role} exited during startup." >&2
      tail -80 "${log}" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for ${role}." >&2
  tail -80 "${log}" >&2 || true
  return 1
}

: >"${RUN_DIR}/target.log"
setsid env \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  SGLANG_ENABLE_DECOUPLED_STREAM_GATE="${TARGET_STREAM_GATE}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${TARGET_SM_PERCENT}" \
  PYTHONPATH="${SOURCE_PYTHONPATH}" \
  "${PYTHON_BIN}" -m sglang.launch_server \
  --host 127.0.0.1 --port "${TARGET_PORT}" \
  --model-path "${TARGET_MODEL}" --served-model-name "$(basename "${TARGET_MODEL}")-SSD" \
  --tp 1 --attention-backend triton --page-size 1 \
  --mem-fraction-static "${TARGET_MEM_FRACTION}" \
  --max-running-requests 2 --max-total-tokens "${TARGET_MAX_TOKENS}" \
  --chunked-prefill-size 2048 --max-prefill-tokens 4096 \
  --disable-prefill-cuda-graph --skip-server-warmup --trust-remote-code \
  --disable-radix-cache "${TARGET_SCHEDULER_ARGS[@]}" --disable-shared-experts-fusion \
  --kt-weight-path "${TARGET_MODEL}" --kt-method BF16 \
  --kt-cpuinfer "${KT_CPU_INFER}" --kt-threadpool-count "${KT_THREADPOOL_COUNT}" \
  --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS}" \
  --speculative-algorithm STANDALONE --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-num-steps "${K}" --speculative-fanout "${F}" \
  --decoupled-spec-role verifier --decoupled-spec-rank 0 \
  --decoupled-spec-data-transport zmq \
  --decoupled-spec-bind-endpoint "${VERIFY_ENDPOINT}" \
  --decoupled-spec-connect-endpoints "[\"${DRAFT_ENDPOINT}\"]" \
  >"${RUN_DIR}/target.log" 2>&1 < /dev/null &
TARGET_PID=$!
echo "${TARGET_PID}" >"${RUN_DIR}/target.pid"
wait_for_ready_log "${TARGET_PID}" "${RUN_DIR}/target.log" target

GRAPH_BS=(1 2 4 8 12 16 24 32 48 64)
DRAFT_GRAPH_BS=()
for bs in "${GRAPH_BS[@]}"; do
  if (( bs <= DRAFT_MAX_REQUESTS )); then
    DRAFT_GRAPH_BS+=("${bs}")
  fi
done
if (( DRAFT_GRAPH_BS[${#DRAFT_GRAPH_BS[@]} - 1] != DRAFT_MAX_REQUESTS )); then
  DRAFT_GRAPH_BS+=("${DRAFT_MAX_REQUESTS}")
fi

: >"${RUN_DIR}/draft.log"
setsid env \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH=1 \
  SGLANG_DECOUPLED_STRICT_LOCKSTEP=1 \
  SGLANG_DECOUPLED_ENUM_WAIT_MS="${SGLANG_DECOUPLED_ENUM_WAIT_MS:-200}" \
  SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH=1 \
  SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH=1 \
  SGLANG_ENABLE_DECOUPLED_DEVICE_PACK=1 \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${DRAFT_SM_PERCENT}" \
  PYTHONPATH="${SOURCE_PYTHONPATH}" \
  "${PYTHON_BIN}" -m sglang.launch_server \
  --host 127.0.0.1 --port "${DRAFT_PORT}" \
  --model-path "${DRAFT_MODEL}" --tp 1 --attention-backend triton --page-size "${DRAFT_PAGE_SIZE}" \
  --mem-fraction-static "${DRAFT_MEM_FRACTION}" \
  --max-running-requests "${DRAFT_MAX_REQUESTS}" --max-total-tokens "${DRAFT_MAX_TOKENS}" \
  --chunked-prefill-size 1024 --max-prefill-tokens 2048 \
  --disable-prefill-cuda-graph --skip-server-warmup --trust-remote-code \
  --speculative-algorithm STANDALONE --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-num-steps "${K}" --speculative-fanout "${F}" \
  --cuda-graph-bs-decode "${DRAFT_GRAPH_BS[@]}" \
  --max-mamba-cache-size "${DRAFT_MAMBA_SLOTS}" \
  --decoupled-spec-role drafter --decoupled-spec-rank 0 \
  --decoupled-spec-data-transport zmq \
  --decoupled-spec-bind-endpoint "${DRAFT_ENDPOINT}" \
  --decoupled-spec-connect-endpoints "[\"${VERIFY_ENDPOINT}\"]" \
  >"${RUN_DIR}/draft.log" 2>&1 < /dev/null &
DRAFT_PID=$!
echo "${DRAFT_PID}" >"${RUN_DIR}/draft.pid"
wait_for_ready_log "${DRAFT_PID}" "${RUN_DIR}/draft.log" draft

printf '%s\n' \
  "repo=${REPO_ROOT}" \
  "target=${TARGET_MODEL}" \
  "draft=${DRAFT_MODEL}" \
  "gpu=${GPU_ID}" \
  "K=${K}" \
  "F=${F}" \
  "branch_rows=${BRANCH_ROWS}" \
  "target_sm=${TARGET_SM_PERCENT}" \
  "draft_sm=${DRAFT_SM_PERCENT}" \
  "draft_page_size=${DRAFT_PAGE_SIZE}" \
  "target_disable_overlap_schedule=${TARGET_DISABLE_OVERLAP_SCHEDULE}" \
  "target_stream_gate=${TARGET_STREAM_GATE}" \
  "draft_max_requests=${DRAFT_MAX_REQUESTS}" \
  "draft_mamba_slots=${DRAFT_MAMBA_SLOTS}" \
  "mps_pipe=${MPS_PIPE_DIR}" \
  "mps_log=${MPS_LOG_DIR}" \
  >"${RUN_DIR}/provenance.txt"

trap - ERR INT TERM
echo "SSD ready: http://127.0.0.1:${TARGET_PORT}"
echo "Run directory: ${RUN_DIR}"
echo "Target PID ${TARGET_PID}, draft PID ${DRAFT_PID}"

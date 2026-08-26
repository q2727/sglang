#!/usr/bin/env bash
set -euo pipefail

RUN_BASE="${RUN_BASE:-${HOME}/workspace/experiments/ssd_enumeration_kt_samegpu}"
RUN_DIR="${1:-${RUN_DIR:-${RUN_BASE}/latest}}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

for name in draft target; do
  pid_file="${RUN_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(<"${pid_file}")"
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  fi
done
sleep 2

provenance="${RUN_DIR}/provenance.txt"
if [[ -f "${provenance}" ]]; then
  MPS_PIPE_DIR="$(sed -n 's/^mps_pipe=//p' "${provenance}")"
  MPS_LOG_DIR="$(sed -n 's/^mps_log=//p' "${provenance}")"
  if [[ -n "${MPS_PIPE_DIR}" && -n "${MPS_LOG_DIR}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
    export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
fi

echo "Stopped SSD processes from ${RUN_DIR}."

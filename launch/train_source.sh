#!/usr/bin/env bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash launch/train_source.sh <task> [run-id]" >&2
  echo "Tasks: virrg_pirrg pirrg_virrg virrg_prgb prgb_virrg r_u u_r" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
TASK_KEY="${1,,}"
WORKERS="${WORKERS:-4}"

case "${TASK_KEY}" in
  virrg_pirrg)
    CFG="configs/VIRRG_PIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="VIRRG_PIRRG"; LOADER="ISPRS" ;;
  pirrg_virrg)
    CFG="configs/PIRRG_VIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="PIRRG_VIRRG"; LOADER="ISPRS" ;;
  virrg_prgb)
    CFG="configs/VIRRG_PRGB_config/deeplabv3plus_resnet50.yaml"
    EXP="VIRRG_PRGB"; LOADER="ISPRS" ;;
  prgb_virrg)
    CFG="configs/PRGB_VIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="PRGB_VIRRG"; LOADER="ISPRS" ;;
  r_u)
    CFG="configs/R_U_config/deeplabv3plus_resnet50.yaml"
    EXP="R_U"; LOADER="loveda" ;;
  u_r)
    CFG="configs/U_R_config/deeplabv3plus_resnet50.yaml"
    EXP="U_R"; LOADER="loveda" ;;
  *)
    echo "Unknown task: ${TASK_KEY}" >&2
    echo "Tasks: virrg_pirrg pirrg_virrg virrg_prgb prgb_virrg r_u u_r" >&2
    exit 2 ;;
esac

RUN_ID="${2:-00_source_${TASK_KEY}}"
cd "${PROJECT_ROOT}"
export WORLD_SIZE=1

python train.py \
  --fresh-run \
  --dataloader "${LOADER}" \
  --cfg "${CFG}" \
  --exp "${EXP}" \
  --run "${RUN_ID}" \
  --workers "${WORKERS}" \
  --set MODEL.BASELINE True MODEL.LR_TARGET 0.0

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash launch/infer.sh <task> <checkpoint> [split] [run-id]" >&2
  echo "Tasks: virrg_pirrg pirrg_virrg virrg_prgb prgb_virrg r_u u_r" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
TASK_KEY="${1,,}"
CHECKPOINT="$2"
WORKERS="${WORKERS:-8}"

case "${TASK_KEY}" in
  virrg_pirrg)
    CFG="configs/VIRRG_PIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="VIRRG_PIRRG"; LOADER="ISPRS"; DEFAULT_SPLIT="val_potsdam_irrg" ;;
  pirrg_virrg)
    CFG="configs/PIRRG_VIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="PIRRG_VIRRG"; LOADER="ISPRS"; DEFAULT_SPLIT="val_vaihingen" ;;
  virrg_prgb)
    CFG="configs/VIRRG_PRGB_config/deeplabv3plus_resnet50.yaml"
    EXP="VIRRG_PRGB"; LOADER="ISPRS"; DEFAULT_SPLIT="val_potsdam_rgb" ;;
  prgb_virrg)
    CFG="configs/PRGB_VIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="PRGB_VIRRG"; LOADER="ISPRS"; DEFAULT_SPLIT="val_vaihingen" ;;
  r_u)
    CFG="configs/R_U_config/deeplabv3plus_resnet50.yaml"
    EXP="R_U"; LOADER="loveda"; DEFAULT_SPLIT="val_loveda_urban" ;;
  u_r)
    CFG="configs/U_R_config/deeplabv3plus_resnet50.yaml"
    EXP="U_R"; LOADER="loveda"; DEFAULT_SPLIT="val_loveda_rural" ;;
  *)
    echo "Unknown task: ${TASK_KEY}" >&2
    exit 2 ;;
esac

SPLIT="${3:-${DEFAULT_SPLIT}}"
RUN_ID="${4:-infer_${TASK_KEY}}"
OUTPUT_DIR="results/${LOADER}/${EXP}/${RUN_ID}/${SPLIT}"

cd "${PROJECT_ROOT}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

python infer_val.py \
  --dataloader "${LOADER}" \
  --cfg "${CFG}" \
  --exp "${EXP}" \
  --run "${RUN_ID}" \
  --resume "${CHECKPOINT}" \
  --infer-list "${SPLIT}" \
  --mask-output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --eval-original-size \
  --save-logit-stats

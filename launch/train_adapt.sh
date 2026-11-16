#!/usr/bin/env bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash launch/train_adapt.sh <task> <source-checkpoint> [run-id] [module-mode] [avg|minentropy|adaf]" >&2
  echo "Tasks: virrg_pirrg pirrg_virrg virrg_prgb prgb_virrg r_u u_r" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
TASK_KEY="${1,,}"
CHECKPOINT="$2"
WORKERS="${WORKERS:-4}"

case "${TASK_KEY}" in
  virrg_pirrg)
    CFG="configs/VIRRG_PIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="VIRRG_PIRRG"; LOADER="ISPRS"; TARGET_HAS_NIR=True ;;
  pirrg_virrg)
    CFG="configs/PIRRG_VIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="PIRRG_VIRRG"; LOADER="ISPRS"; TARGET_HAS_NIR=True ;;
  virrg_prgb)
    CFG="configs/VIRRG_PRGB_config/deeplabv3plus_resnet50.yaml"
    EXP="VIRRG_PRGB"; LOADER="ISPRS"; TARGET_HAS_NIR=False ;;
  prgb_virrg)
    CFG="configs/PRGB_VIRRG_config/deeplabv3plus_resnet50.yaml"
    EXP="PRGB_VIRRG"; LOADER="ISPRS"; TARGET_HAS_NIR=True ;;
  r_u)
    CFG="configs/R_U_config/deeplabv3plus_resnet50.yaml"
    EXP="R_U"; LOADER="loveda"; TARGET_HAS_NIR=False ;;
  u_r)
    CFG="configs/U_R_config/deeplabv3plus_resnet50.yaml"
    EXP="U_R"; LOADER="loveda"; TARGET_HAS_NIR=False ;;
  *)
    echo "Unknown task: ${TASK_KEY}" >&2
    echo "Tasks: virrg_pirrg pirrg_virrg virrg_prgb prgb_virrg r_u u_r" >&2
    exit 2 ;;
esac

# RSDA-CST experiments start from the plain self-training baseline unless a
# module mode is explicitly provided as the fourth argument.
DEFAULT_MODULE_MODE="baseline"
MODULE_MODE="${4:-${DEFAULT_MODULE_MODE}}"
RUN_ID="${3:-01_${MODULE_MODE}_${TASK_KEY}}"
case "${MODULE_MODE}" in
  baseline)
    USE_ADAF=False; USE_GIR=False; USE_RS_AUG=False ;;
  adaf)
    USE_ADAF=True; USE_GIR=False; USE_RS_AUG=False ;;
  gir)
    USE_ADAF=False; USE_GIR=True; USE_RS_AUG=False ;;
  adaf_gir)
    USE_ADAF=True; USE_GIR=True; USE_RS_AUG=False ;;
  rsaug)
    USE_ADAF=False; USE_GIR=False; USE_RS_AUG=True ;;
  rsaug_adaf)
    USE_ADAF=True; USE_GIR=False; USE_RS_AUG=True ;;
  rsaug_gir)
    USE_ADAF=False; USE_GIR=True; USE_RS_AUG=True ;;
  rsaug_adaf_gir)
    USE_ADAF=True; USE_GIR=True; USE_RS_AUG=True ;;
  *)
    echo "Unknown module mode: ${MODULE_MODE}" >&2
    echo "Modes: baseline adaf gir adaf_gir rsaug rsaug_adaf rsaug_gir rsaug_adaf_gir" >&2
    exit 2 ;;
esac

# Optional fusion override for controlled comparisons. Without it, the
# original module mode and YAML pooling choice remain unchanged.
FUSION_MODE="${5:-}"
FUSION_POOL=""
case "${FUSION_MODE}" in
  "") ;;
  avg)
    USE_ADAF=False
    FUSION_POOL="avg_pool" ;;
  minentropy)
    USE_ADAF=False
    FUSION_POOL="minentropy_pool" ;;
  adaf)
    USE_ADAF=True ;;
  *)
    echo "Unknown fusion mode: ${FUSION_MODE} (expected avg, minentropy, or adaf)" >&2
    exit 2 ;;
esac

MODULE_SET=(
  MODEL.USE_ADAF "${USE_ADAF}"
  MODEL.USE_GIR "${USE_GIR}"
  DATASET.USE_REMOTE_AUG "${USE_RS_AUG}"
  DATASET.REMOTE_HAS_NIR "${TARGET_HAS_NIR}"
)
if [[ -n "${FUSION_MODE}" ]]; then
  MODULE_SET+=(MODEL.CONF_POOL_ON True)
fi
if [[ -n "${FUSION_POOL}" ]]; then
  MODULE_SET+=(MODEL.CONF_POOL "${FUSION_POOL}")
fi
cd "${PROJECT_ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Source checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

export WORLD_SIZE=1

python train.py \
  --fresh-run \
  --dataloader "${LOADER}" \
  --cfg "${CFG}" \
  --exp "${EXP}" \
  --run "${RUN_ID}" \
  --resume "${CHECKPOINT}" \
  --workers "${WORKERS}" \
  --set "${MODULE_SET[@]}"

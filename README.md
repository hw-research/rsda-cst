# RSDA-CST

RSDA-CST is a teacher–student framework for unsupervised domain adaptation in remote-sensing semantic segmentation. It includes remote-sensing-specific augmentation (RS-AUG), adaptive entropy-aware multi-view fusion (ADAF), and geometry integrity regularization (GIR). EviFuse is not included in this release.

## Supported transfers

| Task | Source | Target |
|---|---|---|
| `virrg_pirrg` | Vaihingen IRRG | Potsdam IRRG |
| `pirrg_virrg` | Potsdam IRRG | Vaihingen IRRG |
| `virrg_prgb` | Vaihingen IRRG | Potsdam RGB |
| `prgb_virrg` | Potsdam RGB | Vaihingen IRRG |
| `r_u` | LoveDA Rural | LoveDA Urban |
| `u_r` | LoveDA Urban | LoveDA Rural |

## Contents

- `train.py`, `base_trainer.py`, `infer_val.py`: training, validation, and inference.
- `models/`: segmentation networks, DA-SAC teacher–student model, ADAF, and GIR.
- `datasets/`: remote-sensing data loaders, transforms, and RS-AUG.
- `configs/`: independent task configurations.
- `launch/`: source training, adaptation ablations, and inference entry points.
- `tools/`: project/configuration checks and dataset preprocessing utilities.
- `tests/`: focused unit tests for model and training behavior.

Datasets, split manifests, pretrained weights, checkpoints, TensorBoard logs, and generated results are intentionally not included.

## Setup

Create an environment with a CUDA-compatible PyTorch build, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Prepare dataset files and split manifests under `data/` using the paths expected by each task configuration. Place the pretrained ResNet-50 weights at `models/pretrained/resnet50-19c8e357.pth`, or change `MODEL.INIT_MODEL` in the relevant YAML file.

## Training

Run commands from the repository root in Bash or Git Bash. Supported adaptation modes are `baseline`, `adaf`, `gir`, `adaf_gir`, `rsaug`, `rsaug_adaf`, `rsaug_gir`, and `rsaug_adaf_gir`.

```bash
# Train the source-domain model.
bash launch/train_source.sh virrg_pirrg 00_ABN

# Adapt from a source checkpoint.
WORKERS=4 bash launch/train_adapt.sh \
  virrg_pirrg \
  snapshots/ISPRS/VIRRG_PIRRG/00_ABN/epochXXX_scoreX.XXX.pth \
  01_rsaug_adaf_gir rsaug_adaf_gir
```

Replace the task key and checkpoint path for other transfer directions. Each run writes TensorBoard events and `train.log` beneath `logs/`, and checkpoints beneath `snapshots/`.

## Inference

```bash
bash launch/infer.sh virrg_pirrg path/to/checkpoint.pth
```

The default inference split is the target-domain validation set. An optional third argument selects another split, for example `val_vaihingen`.



# RSDA-CST

RSDA-CST is a teacher–student consistency self-training framework for unsupervised domain adaptation in remote sensing semantic segmentation. It builds on [DA-SAC](https://github.com/visinf/da-sac) and provides adaptive entropy-aware fusion (ADAF), geometry integrity regularization (GIR), and remote-sensing-specific augmentation (RS-AUG). RS-AUG is an auxiliary training strategy; ADAF and GIR are the two method components.

## Supported transfer tasks

| Task | Source | Target |
|---|---|---|
| `virrg_pirrg` | Vaihingen IRRG | Potsdam IRRG |
| `virrg_prgb` | Vaihingen IRRG | Potsdam RGB |
| `pirrg_virrg` | Potsdam IRRG | Vaihingen IRRG |
| `prgb_virrg` | Potsdam RGB | Vaihingen IRRG |
| `r_u` | LoveDA Rural | LoveDA Urban |
| `u_r` | LoveDA Urban | LoveDA Rural |

## Setup

Install a CUDA-compatible PyTorch build and the packages in `requirements.txt`. Obtain the ISPRS and LoveDA datasets from their official sources. See [data/README.md](data/README.md) for the manifest format. Place the pretrained ResNet-50 checkpoint at the path specified by `MODEL.INIT_MODEL` in the task configuration, or adjust that path.

## Training and inference

Run commands from the repository root in Bash or Git Bash:

```bash
# Source-domain training
bash launch/train_source.sh virrg_pirrg 00_base

# Target-domain self-training from a source checkpoint
bash launch/train_adapt.sh virrg_pirrg path/to/source_checkpoint.pth 01_full rsaug_adaf_gir

# Inference
bash launch/infer.sh virrg_pirrg path/to/checkpoint.pth
```

Adaptation modes: `baseline`, `adaf`, `gir`, `adaf_gir`, `rsaug`, `rsaug_adaf`, `rsaug_gir`, and `rsaug_adaf_gir`. The six task-specific YAML files are in `configs/`. Use `python tools/check_project.py` to check configuration and launcher consistency.

Training datasets, split manifests, pretrained weights, logs, and model checkpoints are not redistributed in this repository.

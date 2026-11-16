# Dataset Preparation

This repository does not redistribute datasets, labels, split manifests, or pretrained weights. Obtain each dataset from its official source and follow its license and access terms. Prepare the files locally under this directory before running training or inference.

## Supported data

- **ISPRS Potsdam**: RGB and IRRG imagery, with semantic labels.
- **ISPRS Vaihingen**: IRRG imagery, with semantic labels.
- **LoveDA**: Urban and Rural imagery, with semantic labels.

The supported transfer directions are listed in the repository README. For a given experiment, both source and target train/validation splits are required by the current training setup.

## Expected directory layout

The exact subdirectory names must match the paths written in the split manifests. A typical layout is:

```text
data/
├── train_vaihingen.txt
├── val_vaihingen.txt
├── test_vaihingen.txt
├── vaihingen/
│   ├── images/
│   └── labels/

......

Folder names may be adapted, but then the corresponding paths in the manifest files must be updated as well. The `DATASET.ROOT` setting in each task YAML defaults to `data`.

## Split manifest format

Each manifest is a UTF-8 text file. Use one sample per line, with paths relative to `data/`:

```text
relative/path/to/image.png relative/path/to/label.png
```

Examples:

```text
vaihingen/images/top_mosaic_09cm_area1.png vaihingen/labels/top_mosaic_09cm_area1.png
potsdam/images_IRRG/top_potsdam_2_10_IRRG.png potsdam/labels/top_potsdam_2_10_IRRG.png
loveda/urban/images/2083.png loveda/urban/labels/2083.png
```

For unlabeled target training images, the loader can also accept a line containing only the image path. Validation manifests must include labels. Image and label dimensions must match, and filenames must not contain spaces because entries are whitespace-separated.

## Label requirements

Use the class IDs expected by the data loader and task configuration. ISPRS labels should retain the expected six-class source IDs; LoveDA labels should retain its seven-class IDs. Do not apply an extra remapping unless you also update the dataset label mapping in `tools/remote_sensing_categories.py` and validate the resulting masks.

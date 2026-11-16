"""Read-only consistency checks for the six RSDA-CST experiments."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "virrg_pirrg": (
        "configs/VIRRG_PIRRG_config/deeplabv3plus_resnet50.yaml",
        "train_vaihingen", "train_potsdam_irrg", 6,
    ),
    "pirrg_virrg": (
        "configs/PIRRG_VIRRG_config/deeplabv3plus_resnet50.yaml",
        "train_potsdam_irrg", "train_vaihingen", 6,
    ),
    "virrg_prgb": (
        "configs/VIRRG_PRGB_config/deeplabv3plus_resnet50.yaml",
        "train_vaihingen", "train_potsdam_rgb", 6,
    ),
    "prgb_virrg": (
        "configs/PRGB_VIRRG_config/deeplabv3plus_resnet50.yaml",
        "train_potsdam_rgb", "train_vaihingen", 6,
    ),
    "r_u": (
        "configs/R_U_config/deeplabv3plus_resnet50.yaml",
        "train_loveda_rural", "train_loveda_urban", 7,
    ),
    "u_r": (
        "configs/U_R_config/deeplabv3plus_resnet50.yaml",
        "train_loveda_urban", "train_loveda_rural", 7,
    ),
}

EXPECTED_CHANNEL_ADAPT = {
    "virrg_pirrg": False,
    "pirrg_virrg": False,
    "virrg_prgb": True,
    "prgb_virrg": False,
    "r_u": False,
    "u_r": False,
}

VAL_SPLITS = {
    "train_vaihingen": "val_vaihingen",
    "train_potsdam_irrg": "val_potsdam_irrg",
    "train_potsdam_rgb": "val_potsdam_rgb",
    "train_loveda_rural": "val_loveda_rural",
    "train_loveda_urban": "val_loveda_urban",
}


def deep_merge(base, child):
    merged = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError("cyclic config inheritance: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    base_name = content.pop("_BASE_", None)
    if base_name is None:
        return content
    base = load_config(path.parent / base_name, stack + (path,))
    return deep_merge(base, content)


def check_list(split, errors):
    list_path = ROOT / "data" / (split + ".txt")
    if not list_path.is_file():
        return None
    count = 0
    with list_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.strip().split()
            if not parts:
                continue
            count += 1
            for relative in parts:
                if not (ROOT / "data" / relative).is_file():
                    errors.append(
                        "{}:{} missing {}".format(
                            list_path.relative_to(ROOT), line_no, relative
                        )
                    )
    return count


def main():
    errors = []
    checked_splits = set()
    for key, (relative_cfg, source, target, classes) in TASKS.items():
        config_path = ROOT / relative_cfg
        if not config_path.is_file():
            errors.append("{}: missing config {}".format(key, relative_cfg))
            continue
        config = load_config(config_path)
        train = config.get("TRAIN", {})
        dataset = config.get("DATASET", {})

        if (train.get("TASK"), train.get("TARGET")) != (source, target):
            errors.append("{}: source/target mapping is incorrect".format(key))
        if dataset.get("USE_REMOTE_AUG") is True:
            print("[NOTE] {}: RS-AUG is enabled for this experiment".format(key))
        if not isinstance(dataset.get("CHANNEL_ADAPT", False), bool):
            errors.append("{}: CHANNEL_ADAPT must be a boolean".format(key))
        elif dataset.get("CHANNEL_ADAPT", False) is not EXPECTED_CHANNEL_ADAPT[key]:
            errors.append("{}: CHANNEL_ADAPT differs from the fixed task protocol".format(key))
        if not isinstance(dataset.get("REMOTE_HAS_NIR", False), bool):
            errors.append("{}: REMOTE_HAS_NIR must be a boolean".format(key))
        weights = dataset.get("CLASS_WEIGHTS", [])
        if dataset.get("USE_CLASS_WEIGHTS", True) and len(weights) != classes:
            errors.append(
                "{}: expected {} class weights, found {}".format(
                    key, classes, len(weights)
                )
            )

        for split in (source, target, VAL_SPLITS[source], VAL_SPLITS[target]):
            if split not in checked_splits:
                sample_count = check_list(split, errors)
                if sample_count is None:
                    print("[SKIP] {}: data manifest not found".format(split))
                else:
                    print("{:24s} {:5d} samples".format(split, sample_count))
                checked_splits.add(split)
        print("[OK] {:12s} {} -> {}".format(key, source, target))

    for script in ("launch/train_source.sh", "launch/train_adapt.sh", "launch/infer.sh"):
        if not (ROOT / script).is_file():
            errors.append("missing launcher: {}".format(script))

    if errors:
        print("\nProject check failed:", file=sys.stderr)
        for error in errors:
            print("- " + error, file=sys.stderr)
        return 1

    print("\nAll six tasks passed configuration and launcher checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

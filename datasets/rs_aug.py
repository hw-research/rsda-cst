"""Remote-sensing strong photometric augmentation for unlabeled target views."""

from . import tf_target


def build_rs_aug(cfg, dataset_name):
    has_nir = bool(cfg.DATASET.REMOTE_HAS_NIR)
    if has_nir and dataset_name not in ("potsdam", "vaihingen"):
        raise ValueError(
            "REMOTE_HAS_NIR is only supported for native Potsdam/Vaihingen IRRG inputs"
        )
    # Apply RS-AUG to raw target bands before channel adaptation.
    transforms = [
        tf_target.MaskBandwiseAdjust(
            brightness=cfg.DATASET.BANDWISE_BRIGHTNESS,
            contrast=cfg.DATASET.BANDWISE_CONTRAST,
            p=cfg.DATASET.BANDWISE_PROB,
        ),
        tf_target.MaskCloudSimulation(
            cloud_intensity=cfg.DATASET.CLOUD_INTENSITY,
            nir_factor=cfg.DATASET.CLOUD_NIR_FACTOR,
            p=cfg.DATASET.CLOUD_PROB,
            has_nir=has_nir,
        ),
        tf_target.MaskSoftShadow(
            shadow_dark_ratio=cfg.DATASET.SHADOW_DARK_RATIO,
            env_light=cfg.DATASET.SHADOW_ENV_LIGHT,
            p=cfg.DATASET.SHADOW_PROB,
        ),
    ]
    if has_nir:
        transforms.append(
            tf_target.MaskBandCrosstalk(
                leak_ratio=cfg.DATASET.CROSSTALK_LEAK_RATIO,
                p=cfg.DATASET.CROSSTALK_PROB,
            )
        )
    return transforms

"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
Modified for static remote sensing images_IRRG: skip affine generation when GROUP_SIZE == 1.
"""

import os
import bisect
import random
import copy
import math
import torch
import torch.nn.functional as F
from PIL import Image
from .dataloader_base import DLBase
import datasets.tf_target as tf


class DataTarget(DLBase):

    IGNORE_LABEL = 255

    def __init__(self, cfg, split, num_classes, weights="", ignore_labels=[], root=os.path.expanduser('./data')):
        super(DataTarget, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels
        self.num_classes = num_classes
        self.is_static = (cfg.TRAIN.GROUP_SIZE == 1)   # flag for static images_IRRG

        # load split file
        split_fn = os.path.join(self.root, self.split + ".txt")
        assert os.path.isfile(split_fn)

        self.images = []
        self.masks = []
        self.sample_index = {}

        with open(split_fn, "r") as lines:
            for line in lines:
                parts = line.strip("\n").split(' ')
                _image = parts[0]
                _mask = parts[1] if len(parts) > 1 else None

                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                assert os.path.isfile(_image), f'{_image} not found'
                self.images.append(_image)

                if _mask is None:
                    self.masks.append(None)
                else:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), f'{_mask} not found'
                    self.masks.append(_mask)
                    self.sample_index[os.path.basename(_mask)] = len(self.masks) - 1

        self._num_samples = len(self.images)

        # sampling weights (unchanged)
        sample_weights = None
        if weights and os.path.isfile(weights):
            print(f"Loading sample weights: {weights}")
            sample_weights = torch.load(weights)
        elif weights:
            print(f"Path to sample weights NOT found: {weights}")

        self.sample_groups = self.init_sampling(self._num_samples, sample_weights,
                                                prior_weight=cfg.DATASET.SAMPLE_UNIFORM_PRIOR)

        # data augmentation pipelines
        tfs = [tf.CreateMask()]
        tfs_augm = []

        if not split.startswith("val"):
            print(f"{split}: with data augmentation")
            tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))
            tfs.append(tf.MaskRandScale(cfg.DATASET.TARGET_SCALE[0], cfg.DATASET.TARGET_SCALE[1]))
            tfs.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))

            if cfg.DATASET.RND_HFLIP:
                tfs.append(tf.MaskRandHFlip())
            if cfg.DATASET.GUIDED_HFLIP:
                tfs.append(tf.GuidedRandHFlip())
            if cfg.DATASET.RND_ZOOM[1] - cfg.DATASET.RND_ZOOM[0] > 0:
                tfs.append(tf.MaskRandScaleCrop(cfg.DATASET.RND_ZOOM))

            # photometric augmentations
            if cfg.DATASET.RND_BLUR:
                tfs_augm.append(tf.RandGaussianBlur())
            if cfg.DATASET.RND_JITTER > 0:
                tfs_augm.append(tf.MaskRandJitter(cfg.DATASET.RND_JITTER))
            if cfg.DATASET.RND_GREYSCALE > 0:
                tfs_augm.append(tf.MaskRandGreyscale(cfg.DATASET.RND_GREYSCALE))
        else:
            print(f"{split}: no augmentation")
            if cfg.DATASET.VAL_CROP:
                tfs.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))
            else:
                tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))

        self.tf_pre = tf.Compose(tfs)
        self.tf_augm = tf.Compose(tfs_augm)

        tfs_post = [tf.ToTensorMask(),
                    tf.Normalize(mean=self.MEAN, std=self.STD),
                    tf.ApplyMask(-1)]
        self.tf_post = tf.Compose(tfs_post)

    def init_sampling(self, num_samples, loaded_weights=None, prior_weight=0.7):
        """Same as original, omitted for brevity (unchanged)."""
        prior = 1.0 / num_samples
        sample_groups = {}
        if loaded_weights is not None:
            weights = [prior_weight * prior for _ in range(num_samples)]
            weights_uniform = [prior for _ in range(num_samples)]
            for cid in range(self.num_classes):
                if cid in self.cfg.VAL.IGNORE_CLASS:
                    sample_groups[cid] = weights_uniform.copy()
                else:
                    sample_groups[cid] = weights.copy()
            for name, stat in loaded_weights.items():
                sample_id = self.sample_index[name]
                for cid, val in stat.items():
                    sample_groups[cid][sample_id] += (1.0 - prior_weight) * val
            for cid in self.cfg.VAL.IGNORE_CLASS:
                sample_groups[cid] = [prior] * num_samples
        else:
            for cid in range(self.num_classes):
                sample_groups[cid] = [prior] * num_samples

        # cumulative sum
        for cid in range(self.num_classes):
            group = sample_groups[cid]
            for i in range(1, len(group)):
                group[i] += group[i-1]
            # ensure last element is 1 (floating point tolerance)
            if abs(group[-1] - 1.0) > 1e-3:
                group[-1] = 1.0
        return sample_groups

    def __len__(self):
        return self._num_samples

    def denorm(self, image):
        # unchanged
        if image.dim() == 3:
            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)
        return image

    # ------------------------------------------------------------------
    # Affine transformation helpers (only used when GROUP_SIZE > 1)
    # ------------------------------------------------------------------
    def _get_affine(self, params):
        T = len(params)
        affine = torch.zeros(T, 2, 3)
        crop_h, crop_w = self.cfg.DATASET.CROP_SIZE
        aspect_ratio = crop_w / crop_h
        for i, (dy, dx, alpha, scale, flip) in enumerate(params):
            sin = math.sin(alpha * math.pi / 180.0)
            cos = math.cos(alpha * math.pi / 180.0)
            affine[i, 0, 0] = flip * cos
            affine[i, 0, 1] = sin * aspect_ratio
            affine[i, 1, 0] = -sin / aspect_ratio
            affine[i, 1, 1] = cos
            # translation
            affine[i, 0, 2] = -1.0 * (cos * dx + sin * dy)
            affine[i, 1, 2] = -1.0 * (-sin * dx + cos * dy)
            affine[i, 0, 2] /= (crop_w // 2)
            affine[i, 1, 2] /= (crop_h // 2)
            affine[i] *= scale
        return affine

    def _get_affine_inv(self, affine, params):
        T = affine.size(0)
        affine_inv = affine.clone()
        crop_h, crop_w = self.cfg.DATASET.CROP_SIZE
        aspect_ratio = crop_w / crop_h
        affine_inv[:, 0, 1] = affine[:, 1, 0] * aspect_ratio ** 2
        affine_inv[:, 1, 0] = affine[:, 0, 1] / aspect_ratio ** 2
        affine_inv[:, 0, 2] = -1.0 * (affine_inv[:, 0, 0] * affine[:, 0, 2] + affine_inv[:, 0, 1] * affine[:, 1, 2])
        affine_inv[:, 1, 2] = -1.0 * (affine_inv[:, 1, 0] * affine[:, 0, 2] + affine_inv[:, 1, 1] * affine[:, 1, 2])
        scales = torch.tensor([p[3] for p in params], dtype=affine.dtype, device=affine.device).view(-1,1,1)
        affine_inv /= (scales ** 2)
        return affine_inv

    # ------------------------------------------------------------------
    # Main getitem
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        # Category-balanced sampling
        cat_idx = index % len(self.sample_groups)
        cum_weights = self.sample_groups[cat_idx]
        rand_val = random.uniform(0, cum_weights[-1])
        select_idx = bisect.bisect_left(cum_weights, rand_val)

        # Load image and mask
        image = Image.open(self.images[select_idx]).convert('RGB')
        if self.masks[select_idx] is None:
            mask = Image.new('L', image.size, self.IGNORE_LABEL)
        else:
            mask = Image.open(self.masks[select_idx]).convert('L')

        assert image.size == mask.size, f"Size mismatch: {image.size} vs {mask.size}"

        # For static images_IRRG (GROUP_SIZE=1), we do NOT need multiple copies or affine.
        # For videos, we create GROUP_SIZE copies and generate affine parameters.
        T = self.cfg.TRAIN.GROUP_SIZE

        if self.is_static:
            # ---------- STATIC IMAGE MODE ----------
            # Single image, no temporal augmentation, no affine.
            images = [image]
            masks = [mask]

            # Apply spatial augmentations (scale, crop, flip, etc.) without affine
            # Note: tf_pre may still return affine_params, but we ignore them.
            augms = self.tf_pre(images, masks)
            if len(augms) > 2 and isinstance(augms[-1], list):
                # Last element is affine_params (we discard it)
                augms = augms[:-1]
            augms2 = copy.deepcopy(augms)

            # Apply photometric augmentation to the first copy (student input)
            augms1 = self.tf_augm(*augms)

            images1, masks = self.tf_post(*augms1)
            images2, _ = self.tf_post(*augms2)

            # Remove the batch dimension (T=1) -> shape [C,H,W] for images_IRRG, [H,W] for mask
            images1 = images1[0]   # [C,H,W]
            images2 = images2[0]
            masks = masks[0]       # [H,W]

            # For static images_IRRG, return affine = None, affine_inv = None
            return images1, masks, images2, None, None

        else:
            # ---------- VIDEO MODE (GROUP_SIZE > 1) ----------
            images = [image.copy() for _ in range(T)]
            masks = [mask.copy() for _ in range(T)]

            # tf_pre returns (images_IRRG, masks, affine_params_list)
            augms = self.tf_pre(images, masks)
            affine_params = augms[-1]   # list of T tuples
            augms = augms[:-1]

            # Photometric augmentation only for the first copy (student input)
            augms2 = copy.deepcopy(augms)
            augms1 = self.tf_augm(*augms)

            images1, masks = self.tf_post(*augms1)
            images2, _ = self.tf_post(*augms2)

            # Stack along new dimension (T)
            images1 = torch.stack(images1, dim=0)   # [T, C, H, W]
            images2 = torch.stack(images2, dim=0)
            masks = torch.stack(masks, dim=0)       # [T, H, W]

            # Compute affine matrices
            affine = self._get_affine(affine_params)        # [T, 2, 3]
            affine_inv = self._get_affine_inv(affine, affine_params)

            return images1, masks, images2, affine, affine_inv
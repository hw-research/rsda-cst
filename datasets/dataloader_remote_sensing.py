"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Description: 遥感数据集加载器（Potsdam, Vaihingen, LoveDA）
Modified: Added remote-sensing specific strong augmentation
"""

import os
import random
import math
import bisect
import copy
import numpy as np

from PIL import Image

import torch
from .rs_aug import build_rs_aug
import torch.utils.data as data

from .dataloader_base import DLBase
import datasets.tf_seg as tf
import datasets.tf_target as tf_target


def _adapt_vaihingen_irrg(image):
    """Map native Vaihingen NIR-R-G to the RGB-like [R, G, synthesized-B] input."""
    arr = np.asarray(image.convert('RGB'))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            "CHANNEL_ADAPT expects a three-band Vaihingen NIR-R-G image; "
            "got shape {}".format(arr.shape)
        )
    red, green = arr[:, :, 1], arr[:, :, 2]
    blue = ((red.astype(np.float32) + green.astype(np.float32)) / 2).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=2))


def _channel_adapt_enabled(cfg, dataset_name, split):
    if (dataset_name != 'vaihingen'
            or not cfg.DATASET.CHANNEL_ADAPT
            or cfg.MODEL.BASELINE):
        return False

    if cfg.TRAIN.TARGET == 'train_vaihingen':
        adapted_split = cfg.TRAIN.TARGET
    elif (cfg.TRAIN.TASK == 'train_vaihingen'
          and cfg.TRAIN.TARGET == 'train_potsdam_rgb'):
        adapted_split = cfg.TRAIN.TASK
    else:
        return False

    return split in (adapted_split, adapted_split.replace('train_', 'val_', 1))


def _target_split_has_nir(split, dataset_name):
    """Whether the target split's raw image has NIR before any channel mapping."""
    if dataset_name == 'vaihingen':
        return True
    if dataset_name == 'potsdam':
        if 'potsdam_rgb' in split:
            return False
        if 'potsdam_irrg' in split:
            return True
    if dataset_name == 'loveda':
        return False
    return None


class DLSegRemoteSensing(DLBase):
    """遥感数据集分割加载器（用于源域或验证集）"""
    def _init_palette(self):
        if self.dataset_name == 'loveda':
            self.palette = [
                255, 255, 255,    # background
                255, 0, 0,        # building
                255, 255, 0,      # road
                0, 0, 255,        # water
                159, 129, 183,    # barren
                0, 255, 0,        # forest
                255, 195, 128     # agricultural
            ]
        else:  # potsdam, vaihingen
            self.palette = [
                255, 255, 255,  # impervious_surface
                0, 0, 255,  # building
                0, 255, 255,  # low_vegetation
                0, 255, 0,  # tree
                255, 255, 0,  # car
                255, 0, 0  # clutter
            ]

    def __init__(self, cfg, split, ignore_labels=[],
                 root=os.path.expanduser('./data'), renorm=False):
        # 先根据 split 确定 dataset_name
        if 'potsdam' in split:
            self.dataset_name = 'potsdam'
        elif 'vaihingen' in split:
            self.dataset_name = 'vaihingen'
        elif 'loveda_urban' in split or 'loveda_rural' in split:
            self.dataset_name = 'loveda'
        else:
            raise ValueError(f"Unknown remote sensing dataset in split: {split}")
        super(DLSegRemoteSensing, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels
        # 获取标签映射
        from tools.remote_sensing_categories import get_remote_sensing_labels
        self.labels = get_remote_sensing_labels(self.dataset_name)
        self.id_to_train_id = {l.id: l.train_id for l in self.labels}

        # 读取文件列表
        split_fn = os.path.join(self.root, self.split + ".txt")
        if not os.path.isfile(split_fn):
            split_fn = os.path.join(self.root, self.split, self.split + ".txt")
        print(f"Loading file list from: {split_fn}")
        assert os.path.isfile(split_fn), f"File {split_fn} does not exist"

        self.images = []
        self.masks = []

        with open(split_fn, "r") as lines:
            for line in lines:
                parts = line.strip("\n").split(' ')
                _image = parts[0]
                _mask = parts[1] if len(parts) > 1 else None

                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                assert os.path.isfile(_image), f'{_image} not found'
                self.images.append(_image)

                if _mask is not None:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), f'{_mask} not found'
                    self.masks.append(_mask)
                else:
                    self.masks.append(None)

        print(f"Dataloader: {split} # {len(self.images)}")
        if _channel_adapt_enabled(cfg, self.dataset_name, split):
            print(f"\t {split}: channel adaptation NIR-R-G -> R-G-synthesized-B")

        # 定义数据增强
        tfs = [tf.CreateMask()]

        if split.startswith("train"):
            print(f"\t {split}: w/ data augmentation")
            tfs.append(tf.MaskRandScale(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO))

            if cfg.DATASET.SRC_RND_BLUR:
                tfs.append(tf.RandGaussianBlur())

            if cfg.DATASET.RND_HFLIP:
                tfs.append(tf.MaskRandHFlip())

            # Keep source-domain jitter for the supervised/ABN source stage,
            # but disable it during target-domain self-training so that the
            # source branch follows the fixed source-distribution protocol.
            if cfg.MODEL.BASELINE and cfg.DATASET.SRC_RND_JITTER > 0:
                tfs.append(tf.MaskRandJitter(cfg.DATASET.SRC_RND_JITTER))

            if cfg.DATASET.RND_CROP:
                tfs.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))
        else:
            print(f"\t {split}: no augmentation")
            if cfg.DATASET.VAL_CROP:
                tfs.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))
            else:
                tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))

        tfs_post = [tf.ToTensorMask()]

        if renorm:
            mean = self.MEAN
            std = self.STD
            tfs_post += [tf.Normalize(mean=mean, std=std)]
        else:
            tfs_post += [tf.Normalize(mean=self.MEAN, std=self.STD)]

        tfs_post += [tf.ApplyMask(255)]

        self.tf_augm = tf.Compose(tfs)
        self.tf_post = tf.Compose(tfs_post)

        self._num_samples = len(self.images)

    def set_num_samples(self, n):
        print(f"Re-setting # of samples: {self._num_samples} -> {n}")
        self._num_samples = n

    def __len__(self):
        return self._num_samples

    def _convert_label(self, mask):
        """将原始标签转换为训练用的连续ID"""
        mask = np.array(mask)
        converted = np.ones_like(mask) * 255
        for l in self.labels:
            if l.train_id != 255:
                converted[mask == l.id] = l.train_id
        return Image.fromarray(converted.astype(np.uint8))

    def denorm(self, image):
        """反归一化图像（用于可视化）"""
        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

    def __getitem__(self, index):
        index = index % len(self.images)
        image = Image.open(self.images[index])          # 注意：不要立即 convert('RGB')

        # Map Vaihingen inputs selected for adaptation; source pretraining
        # always retains native bands.
        if _channel_adapt_enabled(self.cfg, self.dataset_name, self.split):
            image = _adapt_vaihingen_irrg(image)
        else:
            image = image.convert('RGB')
        # 后续处理保持不变
        if self.masks[index] is None:
            mask = Image.new('L', image.size, 255)
        else:
            mask = Image.open(self.masks[index])
            mask = self._convert_label(mask)

        assert image.size == mask.size, f"Image & mask shape mismatch: {image.size} vs {mask.size}"

        augms = self.tf_augm(image, mask)
        image, mask = self.tf_post(*augms)

        return image, mask


class DataTargetRemoteSensing(DLBase):
    """遥感数据集目标域加载器（用于自监督训练）"""
    def _init_palette(self):
        if self.dataset_name == 'loveda':
            self.palette = [
                255, 255, 255,    # background
                255, 0, 0,        # building
                255, 255, 0,      # road
                0, 0, 255,        # water
                159, 129, 183,    # barren
                0, 255, 0,        # forest
                255, 195, 128     # agricultural
            ]
        else:  # potsdam, vaihingen
            self.palette = [
                255, 255, 255,  # impervious_surface
                0, 0, 255,  # building
                0, 255, 255,  # low_vegetation
                0, 255, 0,  # tree
                255, 255, 0,  # car
                255, 0, 0  # clutter
            ]

    IGNORE_LABEL = 255

    def __init__(self, cfg, split, num_classes, weights="", ignore_labels=[], root=os.path.expanduser('./data')):
        # 先根据 split 确定 dataset_name
        if 'potsdam' in split:
            self.dataset_name = 'potsdam'
        elif 'vaihingen' in split:
            self.dataset_name = 'vaihingen'
        elif 'loveda_urban' in split or 'loveda_rural' in split:
            self.dataset_name = 'loveda'
        else:
            raise ValueError(f"Unknown remote sensing dataset in split: {split}")
        super(DataTargetRemoteSensing, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels
        self.num_classes = num_classes
        # 获取标签映射
        from tools.remote_sensing_categories import get_remote_sensing_labels
        self.labels = get_remote_sensing_labels(self.dataset_name)
        self.id_to_train_id = {l.id: l.train_id for l in self.labels}

        # 读取文件列表
        split_fn = os.path.join(self.root, self.split + ".txt")
        if not os.path.isfile(split_fn):
            split_fn = os.path.join(self.root, self.split, self.split + ".txt")
        print(f"Loading file list from: {split_fn}")
        assert os.path.isfile(split_fn), f"File {split_fn} does not exist"

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

                if _mask is not None:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), f'{_mask} not found'
                    self.masks.append(_mask)
                    mindex = len(self.masks) - 1
                    self.sample_index[os.path.basename(_mask)] = mindex
                else:
                    self.masks.append(None)

        self._num_samples = len(self.images)

        # 初始化采样权重
        sample_weights = None
        if len(weights):
            if os.path.isfile(weights):
                print(f"Loading sample weights: {weights}")
                sample_weights = torch.load(weights)
            else:
                print(f"Path to sample weights NOT found: {weights}")

        self.sample_groups = self._init_sampling(self._num_samples, sample_weights,
                                                  prior_weight=cfg.DATASET.SAMPLE_UNIFORM_PRIOR)

        # ==================== 定义几何增强（tf_pre） ====================
        tfs = [tf_target.CreateMask()]
        print(f"Dataloader: {split} # {len(self.images)}")
        if _channel_adapt_enabled(cfg, self.dataset_name, split):
            print(f"\t {split}: channel adaptation NIR-R-G -> R-G-synthesized-B")

        if split.startswith("train"):
            print(f"\t {split}: w/ data augmentation")
            tfs.append(tf_target.MaskScale(cfg.DATASET.CROP_SIZE))
            tfs.append(tf_target.MaskRandScale(cfg.DATASET.TARGET_SCALE[0], cfg.DATASET.TARGET_SCALE[1]))
            tfs.append(tf_target.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))

            if cfg.DATASET.RND_HFLIP:
                tfs.append(tf_target.MaskRandHFlip())

            if cfg.DATASET.GUIDED_HFLIP:
                tfs.append(tf_target.GuidedRandHFlip())

            if cfg.DATASET.RND_ZOOM[1] - cfg.DATASET.RND_ZOOM[0] > 0:
                tfs.append(tf_target.MaskRandScaleCrop(cfg.DATASET.RND_ZOOM))

            # 构建强增强（光度部分）
            use_remote_aug = getattr(cfg.DATASET, 'USE_REMOTE_AUG', False)
            self.tf_augm = []

            if use_remote_aug:
                has_nir = bool(getattr(cfg.DATASET, 'REMOTE_HAS_NIR', False))
                actual_has_nir = _target_split_has_nir(split, self.dataset_name)
                if actual_has_nir is not None and has_nir != actual_has_nir:
                    raise ValueError(
                        "REMOTE_HAS_NIR={} does not match raw target split {} "
                        "(expected {}).".format(has_nir, split, actual_has_nir)
                    )
                self.tf_augm.extend(build_rs_aug(cfg, self.dataset_name))
                print("\t {}: w/ remote-sensing strong augmentation (RS-AUG)".format(split))
                print("\t RS-AUG input: raw target bands; REMOTE_HAS_NIR={}".format(has_nir))
            else:
                # 保留原始自然图像增强（用于对比）
                if cfg.DATASET.RND_BLUR:
                    self.tf_augm.append(tf_target.RandGaussianBlur())
                if cfg.DATASET.RND_JITTER > 0:
                    self.tf_augm.append(tf_target.MaskRandJitter(cfg.DATASET.RND_JITTER))
                if cfg.DATASET.RND_GREYSCALE > 0:
                    self.tf_augm.append(tf_target.MaskRandGreyscale(cfg.DATASET.RND_GREYSCALE))

            self.tf_augm = tf_target.Compose(self.tf_augm)
        else:
            print(f"\t {split}: no augmentation")
            if cfg.DATASET.VAL_CROP:
                tfs.append(tf_target.MaskCenterCrop(cfg.DATASET.CROP_SIZE))
            else:
                tfs.append(tf_target.MaskScale(cfg.DATASET.CROP_SIZE))
            self.tf_augm = None

        # 关键：必须将几何增强序列保存为 self.tf_pre（无论 train 还是 val）
        self.tf_pre = tf_target.Compose(tfs)

        # 后处理：归一化、转张量等
        tfs_post = [tf_target.ToTensorMask(),
                    tf_target.Normalize(mean=self.MEAN, std=self.STD),
                    tf_target.ApplyMask(-1)]
        self.tf_post = tf_target.Compose(tfs_post)

    def _convert_label(self, mask):
        """将原始标签转换为训练用的连续ID"""
        mask = np.array(mask)
        converted = np.ones_like(mask) * 255
        for l in self.labels:
            if l.train_id != 255:
                converted[mask == l.id] = l.train_id
        return Image.fromarray(converted.astype(np.uint8))

    def _init_sampling(self, num_samples, loaded_weights=None, prior_weight=0.7):
        """初始化类别平衡采样"""
        prior = 1. / num_samples
        sample_groups = {}

        if loaded_weights is not None:
            weights = [prior_weight * prior for _ in range(num_samples)]
            weights_uniform = [prior for _ in range(num_samples)]

            existing_classes = set()
            for name, stat in loaded_weights.items():
                existing_classes.update(stat.keys())
            print(f"Classes present in weight file: {sorted(existing_classes)}")

            for cid in range(self.num_classes):
                if cid in self.cfg.VAL.IGNORE_CLASS:
                    sample_groups[cid] = weights_uniform.copy()
                else:
                    sample_groups[cid] = weights.copy()

            for name, stat in loaded_weights.items():
                if name not in self.sample_index:
                    continue
                sample_id = self.sample_index[name]
                for cid, val in stat.items():
                    if cid < self.num_classes:
                        sample_groups[cid][sample_id] += (1. - prior_weight) * val

            for cid in range(self.num_classes):
                if cid not in existing_classes and cid not in self.cfg.VAL.IGNORE_CLASS:
                    for sample_id in range(num_samples):
                        sample_groups[cid][sample_id] += (1. - prior_weight) * 0.001
        else:
            weights = [prior for _ in range(num_samples)]
            for cid in range(self.num_classes):
                sample_groups[cid] = weights.copy()

        # 累积分布
        for cid in range(self.num_classes):
            sample_group = sample_groups[cid]
            sample_group = np.array(sample_group)
            sample_group = sample_group / sample_group.sum()
            cumsum = np.cumsum(sample_group)
            sample_groups[cid] = cumsum.tolist()

        return sample_groups

    def set_num_samples(self, n):
        print(f"Re-setting # of samples: {self._num_samples} -> {n}")
        self._num_samples = n

    def __len__(self):
        return self._num_samples

    def _get_affine(self, params):
        """构建仿射变换矩阵"""
        affine = torch.zeros(self.cfg.TRAIN.GROUP_SIZE, 2, 3)
        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / float(self.cfg.DATASET.CROP_SIZE[1])

        for i, (dy, dx, alpha, scale, flip) in enumerate(params):
            sin = math.sin(alpha * math.pi / 180.)
            cos = math.cos(alpha * math.pi / 180.)

            affine[i, 0, 0], affine[i, 0, 1] = flip * cos, sin * aspect_ratio
            affine[i, 1, 0], affine[i, 1, 1] = -sin / aspect_ratio, cos

            affine[i, 0, 2] = -1. * (cos * dx + sin * dy)
            affine[i, 1, 2] = -1. * (-sin * dx + cos * dy)

            affine[i, 0, 2] /= float(self.cfg.DATASET.CROP_SIZE[1] // 2)
            affine[i, 1, 2] /= float(self.cfg.DATASET.CROP_SIZE[0] // 2)

            affine[i] *= scale

        return affine

    def _get_affine_inv(self, affine, params):
        """计算逆仿射变换"""
        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / float(self.cfg.DATASET.CROP_SIZE[1])
        affine_inv = affine.clone()
        affine_inv[:, 0, 1] = affine[:, 1, 0] * aspect_ratio ** 2
        affine_inv[:, 1, 0] = affine[:, 0, 1] / aspect_ratio ** 2
        affine_inv[:, 0, 2] = -1 * (affine_inv[:, 0, 0] * affine[:, 0, 2] + affine_inv[:, 0, 1] * affine[:, 1, 2])
        affine_inv[:, 1, 2] = -1 * (affine_inv[:, 1, 0] * affine[:, 0, 2] + affine_inv[:, 1, 1] * affine[:, 1, 2])
        affine_inv /= torch.Tensor(params)[:, 3].view(-1, 1, 1) ** 2
        return affine_inv

    def __getitem__(self, index):
        # 类别平衡采样
        cat_idx = index % len(self.sample_groups)
        sample_weights = self.sample_groups[cat_idx]
        rand_sample = random.uniform(0, sample_weights[-1])
        select_idx = bisect.bisect_left(sample_weights, rand_sample)

        image = Image.open(self.images[select_idx])   # 注意：不要立即 convert('RGB')
        channel_adapt = _channel_adapt_enabled(self.cfg, self.dataset_name, self.split)
        # When RS-AUG uses native NIR, keep it intact through augmentation,
        # then map both teacher and student views to the model's RGB-like input.
        defer_channel_adapt = (
            channel_adapt
            and self.tf_augm is not None
            and bool(getattr(self.cfg.DATASET, 'USE_REMOTE_AUG', False))
            and bool(getattr(self.cfg.DATASET, 'REMOTE_HAS_NIR', False))
        )
        if channel_adapt and not defer_channel_adapt:
            image = _adapt_vaihingen_irrg(image)
        elif not defer_channel_adapt:
            # 其他数据集（如 Potsdam 已经是 RGB）直接转换
            image = image.convert('RGB')
        if self.masks[select_idx] is None:
            mask = Image.new('L', image.size, self.IGNORE_LABEL)
        else:
            mask = Image.open(self.masks[select_idx])
            mask = self._convert_label(mask)

        assert image.size == mask.size

        # 复制 N 个副本
        images = [image.copy() for _ in range(self.cfg.TRAIN.GROUP_SIZE)]
        masks = [mask.copy() for _ in range(self.cfg.TRAIN.GROUP_SIZE)]

        # 应用几何增强（包含可能的仿射参数）
        augms = self.tf_pre(images, masks)
        if len(augms) == 4:
            affine_params = augms[-1]
            augms = augms[:-1]
        else:
            affine_params = None

        # 深拷贝弱增强副本
        augms2 = copy.deepcopy(augms)
        # 应用光度增强（强增强）
        if self.tf_augm is not None:
            augms1 = self.tf_augm(*augms)
        else:
            augms1 = augms

        if defer_channel_adapt:
            augms1 = list(augms1)
            augms1[0] = [_adapt_vaihingen_irrg(img) for img in augms1[0]]
            augms2 = list(augms2)
            augms2[0] = [_adapt_vaihingen_irrg(img) for img in augms2[0]]

        # 后处理（归一化、转张量等）
        images1, masks = self.tf_post(*augms1)
        images2, _ = self.tf_post(*augms2)

        # 合并批次维度
        images1 = torch.cat([x.unsqueeze(0) for x in images1], 0)
        images2 = torch.cat([x.unsqueeze(0) for x in images2], 0)
        masks = torch.cat([x.unsqueeze(0) for x in masks], 0)

        if affine_params is not None:
            affine = self._get_affine(affine_params)
            affine_inv = self._get_affine_inv(affine, affine_params)
        else:
            # 默认单位仿射变换
            affine = torch.zeros(self.cfg.TRAIN.GROUP_SIZE, 2, 3)
            affine[:, 0, 0] = 1
            affine[:, 1, 1] = 1
            affine_inv = affine.clone()

        return images1, masks, images2, affine, affine_inv

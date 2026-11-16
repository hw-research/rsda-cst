"""
Copyright (c) 2021 TU Darmstadt
Modified for ABN stage: target domain loader returning (image, mask) with mask=255.
"""

import os
import numpy as np
from PIL import Image
import torch

from .dataloader_base import DLBase
import datasets.tf_seg as tf


class DataTargetABN(DLBase):
    """
    用于 ABN 阶段的目标域加载器。
    返回 (image, mask)，其中 mask 全为 255 (ignore label)。
    不生成多视图，只进行常规的几何变换（缩放、裁剪、翻转）和归一化。
    """
    IGNORE_LABEL = 255

    def __init__(self, cfg, split, root=os.path.expanduser('./data')):
        super(DataTargetABN, self).__init__()
        self.cfg = cfg
        self.root = root
        self.split = split

        # 读取文件列表（目标域的图像文件，不读取标签）
        split_fn = os.path.join(self.root, self.split + ".txt")
        if not os.path.isfile(split_fn):
            split_fn = os.path.join(self.root, self.split, self.split + ".txt")
        print(f"Loading target file list from: {split_fn}")
        assert os.path.isfile(split_fn), f"File {split_fn} does not exist"

        self.images = []
        with open(split_fn, "r") as lines:
            for line in lines:
                parts = line.strip("\n").split(' ')
                _image = parts[0]
                # 忽略 mask 文件（目标域无标签）
                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                assert os.path.isfile(_image), f'{_image} not found'
                self.images.append(_image)

        self._num_samples = len(self.images)

        # 定义数据增强（与源域 DLSegRemoteSensing 类似）
        tfs = [tf.CreateMask()]
        if split.startswith("train"):
            print(f"\t {split}: w/ data augmentation")
            tfs.append(tf.MaskRandScale(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO))
            if cfg.DATASET.SRC_RND_BLUR:
                tfs.append(tf.RandGaussianBlur())
            if cfg.DATASET.RND_HFLIP:
                tfs.append(tf.MaskRandHFlip())
            if cfg.DATASET.SRC_RND_JITTER > 0:
                tfs.append(tf.MaskRandJitter(cfg.DATASET.SRC_RND_JITTER))
            if cfg.DATASET.RND_CROP:
                tfs.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))
        else:
            print(f"\t {split}: no augmentation")
            if cfg.DATASET.VAL_CROP:
                tfs.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))
            else:
                tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))

        tfs_post = [tf.ToTensorMask(),
                    tf.Normalize(mean=self.MEAN, std=self.STD),
                    tf.ApplyMask(self.IGNORE_LABEL)]

        self.tf_augm = tf.Compose(tfs)
        self.tf_post = tf.Compose(tfs_post)

    def __len__(self):
        return self._num_samples

    def set_num_samples(self, n):
        """调整数据集的样本数量（用于与源域对齐）"""
        print(f"Re-setting # of samples: {self._num_samples} -> {n}")
        self._num_samples = n

    def denorm(self, image):
        if image.dim() == 3:
            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)
        return image

    def __getitem__(self, index):
        index = index % len(self.images)
        image = Image.open(self.images[index]).convert('RGB')
        # 创建一个全 IGNORE_LABEL 的 mask
        mask = Image.new('L', image.size, self.IGNORE_LABEL)

        # 应用数据增强（图像和 mask 同时变换，mask 始终为 ignore）
        augms = self.tf_augm(image, mask)
        image, mask = self.tf_post(*augms)

        return image, mask
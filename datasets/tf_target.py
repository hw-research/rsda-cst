"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import random
import math
import numpy as np
import numbers
import collections
import torch
from PIL import Image, ImageFilter

import torchvision.transforms as tf
import torchvision.transforms.functional as F

from functools import partial

class Compose:
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, *result):

        # allow for intermediate representations
        for t in self.segtransform:
            result = t(*result)

        return result

class ToTensorMask:

    def __toByteTensor(self, pic):
        return torch.from_numpy(np.array(pic, np.int32, copy=False))

    def __call__(self, images, labels, masks):

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            images[i] = F.to_tensor(image)
            labels[i] = self.__toByteTensor(label)
            masks[i] = self.__toByteTensor(mask)

        return images, labels, masks

class CreateMask:

    def __call__(self, images, labels):

        masks = []
        for i, label in enumerate(labels):
            # create mask as labels
            masks.append(Image.new("L", label.size))

        return images, labels, masks

class Normalize:

    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):

        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)

        self.mean = mean
        self.std = std

    def __call__(self, images, labels, masks):

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):

            if self.std is None:
                for t, m in zip(image, self.mean):
                    t.sub_(m)
            else:
                for t, m, s in zip(image, self.mean, self.std):
                    t.sub_(m).div_(s)

        return images, labels, masks

class ApplyMask:

    def __init__(self, ignore_label):
        self.ignore_label = ignore_label

    def __call__(self, images, labels, masks):

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            mask = mask > 0.

            images[i] *= (1. - mask.type_as(image))
            labels[i][mask] = self.ignore_label
            labels[i] = labels[i].squeeze(0).long()

        return images, labels

class MaskRandAffine(object):

    def __init__(self, degree, scale):
        self.degrees = (-degree, degree)
        self.scale = scale
        self.translate = None
        self.shear = None

    def __call__(self, images, labels, masks):

        # getting the parameters
        ret = tf.RandomAffine.get_params(self.degrees, \
                                         self.translate, \
                                         self.scale, \
                                         self.shear, \
                                         images[0].size)

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            # fill values will be replaced later
            images[i] = F.affine(image, *ret, resample=Image.BILINEAR, fillcolor=0)
            labels[i] = F.affine(label, *ret, resample=Image.NEAREST, fillcolor=0)

            # keep track of the values to ignore later
            masks[i] = F.affine(mask, *ret, resample=Image.NEAREST, fillcolor=1)

        return image, labels, mask

class MaskScale(object):

    def __init__(self, size):
        self.scaled_size = (size[1], size[0])

    def __call__(self, images, labels, masks):

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            images[i] = image.resize(self.scaled_size, Image.BILINEAR)
            labels[i] = label.resize(self.scaled_size, Image.NEAREST)
            masks[i] = mask.resize(self.scaled_size, Image.NEAREST)

        return images, labels, masks

class GuidedRandHFlip:

    def __call__(self, images, labels, masks, affine=None):

        if affine is None:
            affine = [[0.,0.,0.,1.,1.] for _ in range(len(images))]

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):

            if random.random() > 0.5:
                images[i] = F.hflip(image)
                labels[i] = F.hflip(label)
                masks[i] = F.hflip(mask)
                affine[i][4] *= -1

        return images, labels, masks, affine

class MaskRandScaleCrop(object):

    def __init__(self, scale_range):
        self.scale_from, self.scale_to = scale_range

    def get_params(self, h, w):
        # generating random crop
        # preserves aspect ratio
        new_scale = random.uniform(self.scale_from, self.scale_to)

        new_h = int(new_scale * h)
        new_w = int(new_scale * w)

        # generating
        if new_scale < 1.:
            assert w >= new_w, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
        else:
            assert w <= new_w, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(h - new_h, 0)
            j = random.randint(w - new_w, 0)

        return i, j, new_h, new_w, new_scale

    def __call__(self, images, labels, masks, affine=None):

        if affine is None:
            affine = [[0.,0.,0.,1.,1.] for _ in range(len(images))]

        W, H = images[0].size

        i2 = H / 2
        j2 = W / 2

        for k, (image, label, mask) in enumerate(zip(images, labels, masks)):

            if k == 0:
                continue # no change in the original copy

            ii, jj, h, w, s = self.get_params(H, W)

            if s == 1.:
                continue # no change in scale

            # displacement of the centre
            dy = ii + h / 2 - i2
            dx = jj + w / 2 - j2

            affine[k][0] = dy
            affine[k][1] = dx
            affine[k][3] = 1 / s # scale

            if s < 1.: # zooming in -> crop
                assert ii >= 0 and jj >= 0

                image_crop = F.crop(image, ii, jj, h, w)
                images[k] = image_crop.resize((W, H), Image.BILINEAR)

                label_crop = F.crop(label, ii, jj, h, w)
                labels[k] = label_crop.resize((W, H), Image.NEAREST)

                mask_crop = F.crop(mask, ii, jj, h, w)
                masks[k] = mask_crop.resize((W, H), Image.NEAREST)
            else: # zooming out -> pad
                assert ii <= 0 and jj <= 0

                pad_l = abs(jj)
                pad_r = w - W - pad_l
                pad_t = abs(ii)
                pad_b = h - H - pad_t

                image_pad = F.pad(image, (pad_l, pad_t, pad_r, pad_b))
                images[k] = image_pad.resize((W, H), Image.BILINEAR)

                label_pad = F.pad(label, (pad_l, pad_t, pad_r, pad_b), 1)
                labels[k] = label_pad.resize((W, H), Image.NEAREST)

                mask_pad = F.pad(mask, (pad_l, pad_t, pad_r, pad_b), 1)
                masks[k] = mask_pad.resize((W, H), Image.NEAREST)

        return images, labels, masks, affine

class MaskRandScale(object):

    def __init__(self, scale_from, scale_to):
        self.scale = (scale_from, scale_to)
        self.aspect_ratio = None

    def __call__(self, images, labels, masks):

        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()

        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        scaled_size = (int(masks[0].size[0] * scale_factor_y), \
                       int(masks[0].size[1] * scale_factor_x))

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            images[i] = image.resize(scaled_size, Image.BILINEAR)
            labels[i] = label.resize(scaled_size, Image.NEAREST)
            masks[i] = mask.resize(scaled_size, Image.NEAREST)

        return images, labels, masks

class MaskRandCrop:

    def __init__(self, size, pad_if_needed=False):
        self.size = size # (h, w)
        self.pad_if_needed = pad_if_needed

    def __pad(self, img, padding_mode='constant', fill=0):

        # pad the width if needed
        pad_width = self.size[1] - img.size[0]
        pad_height = self.size[0] - img.size[1]
        if self.pad_if_needed and (pad_width > 0 or pad_height > 0):
            pad_l = max(0, pad_width // 2)
            pad_r = max(0, pad_width - pad_l)
            pad_t = max(0, pad_height // 2)
            pad_b = max(0, pad_height - pad_t)
            img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), fill, padding_mode)

        return img

    def __call__(self, images, labels, masks):

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            images[i] = self.__pad(image)
            labels[i] = self.__pad(label)
            masks[i] = self.__pad(mask, fill=1)

        i, j, h, w = tf.RandomCrop.get_params(images[0], self.size)

        for k, (image, label, mask) in enumerate(zip(images, labels, masks)):
            images[k] = F.crop(image, i, j, h, w)
            labels[k] = F.crop(label, i, j, h, w)
            masks[k] = F.crop(mask, i, j, h, w)

        return images, labels, masks

class MaskCenterCrop:

    def __init__(self, size):
        self.size = size # (h, w)

    def __call__(self, images, labels, masks):

        for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
            images[i] = F.center_crop(image, self.size)
            labels[i] = F.center_crop(label, self.size)
            masks[i] = F.center_crop(mask, self.size)

        return images, labels, masks

class MaskRandHFlip:

    def __call__(self, images, labels, masks):

        if random.random() > 0.5:

            for i, (image, label, mask) in enumerate(zip(images, labels, masks)):
                images[i] = F.hflip(image)
                labels[i] = F.hflip(label)
                masks[i] = F.hflip(mask)

        return images, labels, masks

class RandGaussianBlur:

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, radius=[.1, 2.]):
        self.radius = radius

    def __call__(self, images, labels, masks, crop_jitter=None):

        for i, image in enumerate(images):
            radius = random.uniform(self.radius[0], self.radius[1])
            images[i] = image.filter(ImageFilter.GaussianBlur(radius))

        if crop_jitter is None:
            return images, labels, masks

        return images, labels, masks, crop_jitter

class MaskRandGreyscale:

    def __init__(self, p = 0.1):
        self.p = p

    def __call__(self, images, labels, masks, crop_jitter=None):

        for i, image in enumerate(images):
            if self.p > random.random():
                images[i] = F.to_grayscale(images[i], num_output_channels=3)

        if crop_jitter is None:
            return images, labels, masks

        return images, labels, masks, crop_jitter

class MaskRandJitter:

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, jitter, p=0.5):
        self.p = p
        self.jitter = tf.ColorJitter(brightness=jitter, \
                                     contrast=jitter, \
                                     saturation=jitter, \
                                     hue=min(0.1, jitter))

    def __call__(self, images, labels, masks, crop_jitter=None):

        for i, image in enumerate(images):

            if random.random() < self.p:
                images[i] = self.jitter(image)

        if crop_jitter is None:
            return images, labels, masks

        return images, labels, masks, crop_jitter
# ==================== 以下为新增的遥感专用增强类 ====================
# ==================== 以下为新增的遥感专用增强类（纯 PIL 实现，无需 cv2） ====================

class MaskBandwiseAdjust(object):
    """独立调整每个波段的亮度与对比度（IRRG/RGB）。"""
    def __init__(self, brightness=(0.7, 1.3), contrast=(0.7, 1.3), p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, images, labels, masks, crop_jitter=None):
        import numpy as np
        from PIL import Image

        for i, image in enumerate(images):
            if np.random.rand() > self.p:
                continue
            img = np.array(image).astype(np.float32)
            h, w, c = img.shape
            for ch in range(c):
                brightness = np.random.uniform(*self.brightness)
                contrast = np.random.uniform(*self.contrast)
                band_mean = img[:, :, ch].mean()
                img[:, :, ch] = (img[:, :, ch] - band_mean) * contrast + band_mean * brightness
            img = np.clip(img, 0, 255).astype(np.uint8)
            images[i] = Image.fromarray(img)

        if crop_jitter is None:
            return images, labels, masks
        else:
            return images, labels, masks, crop_jitter


class MaskCloudSimulation(object):
    """模拟低频薄云；仅在具有近红外波段时应用 NIR 特定衰减。"""
    def __init__(self, cloud_intensity=(0.1, 0.4), nir_factor=0.3, p=0.7, has_nir=False):
        self.cloud_intensity = cloud_intensity
        self.nir_factor = nir_factor
        self.p = p
        self.has_nir = bool(has_nir)

    def __call__(self, images, labels, masks, crop_jitter=None):
        import numpy as np
        from PIL import Image, ImageFilter

        for i, image in enumerate(images):
            if np.random.rand() > self.p:
                continue
            img = np.array(image).astype(np.float32)
            h, w, c = img.shape

            # 生成低频噪声（高斯模糊白噪声）
            noise = np.random.rand(h, w).astype(np.float32)
            noise_img = Image.fromarray((noise * 255).astype(np.uint8))
            noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=15))
            noise = np.array(noise_img).astype(np.float32) / 255.0

            intensity = np.random.uniform(*self.cloud_intensity)

            if c == 3 and self.has_nir:
                # ISPRS IRRG: band 0 is NIR; remaining bands are visible.
                for ch in range(c):
                    if ch == 0:  # NIR
                        factor = self.nir_factor
                    else:
                        factor = 1.0
                    img[:,:,ch] = img[:,:,ch] * (1 - intensity * factor * noise) + \
                                   255 * intensity * factor * noise
            else:
                img = img * (1 - intensity * noise[:,:,np.newaxis]) + \
                      255 * intensity * noise[:,:,np.newaxis]

            img = np.clip(img, 0, 255).astype(np.uint8)
            images[i] = Image.fromarray(img)

        if crop_jitter is None:
            return images, labels, masks
        else:
            return images, labels, masks, crop_jitter


class MaskSoftShadow(object):
    """随机多边形软阴影，纯 PIL 实现"""
    def __init__(self, shadow_dark_ratio=(0.3, 0.7), env_light=(30,40,50), p=0.6):
        self.shadow_dark_ratio = shadow_dark_ratio
        self.env_light = env_light
        self.p = p

    def __call__(self, images, labels, masks, crop_jitter=None):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter

        for i, image in enumerate(images):
            if np.random.rand() > self.p:
                continue
            img = np.array(image).astype(np.float32)
            h, w = img.shape[:2]

            # 随机多边形顶点
            pts = []
            n_pts = np.random.randint(3, 6)
            for _ in range(n_pts):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                pts.append((x, y))

            # 创建阴影掩膜
            mask_img = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(pts, fill=255)
            blur_radius = np.random.choice([5, 9, 15])
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask_shadow = np.array(mask_img).astype(np.float32) / 255.0

            dark_ratio = np.random.uniform(*self.shadow_dark_ratio)
            env = np.array(self.env_light, dtype=np.float32)
            dark_img = img * dark_ratio + env

            mask_shadow = mask_shadow[:,:,np.newaxis]
            img = img * (1 - mask_shadow) + dark_img * mask_shadow
            img = np.clip(img, 0, 255).astype(np.uint8)
            images[i] = Image.fromarray(img)

        if crop_jitter is None:
            return images, labels, masks
        else:
            return images, labels, masks, crop_jitter


class MaskBandCrosstalk(object):
    """模拟光谱串扰，通常是近红外泄漏到红波段"""
    def __init__(self, leak_ratio=(0.0, 0.15), p=0.5):
        self.leak_ratio = leak_ratio
        self.p = p

    def __call__(self, images, labels, masks, crop_jitter=None):
        import numpy as np
        from PIL import Image

        for i, image in enumerate(images):
            if np.random.rand() > self.p:
                continue
            img = np.array(image).astype(np.float32)
            if img.shape[2] == 3:
                alpha = np.random.uniform(*self.leak_ratio)
                img[:,:,1] = img[:,:,1] + alpha * img[:,:,0]
                img = np.clip(img, 0, 255).astype(np.uint8)
                images[i] = Image.fromarray(img)

        if crop_jitter is None:
            return images, labels, masks
        else:
            return images, labels, masks, crop_jitter

"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Modified for remote sensing datasets:
- Use DLSegRemoteSensing for inference.
- Save predictions as trainId PNGs.
- Support dataset-specific palette for visualization.
"""

import os
import sys
import copy
import numpy as np
import imageio
import time
import csv

from tqdm import tqdm
from PIL import Image

import torch
from utils.torch_compat import torch_load_compat
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model
from utils.timer import Timer
from datasets import get_class_names, get_num_classes
from datasets.dataloader_infer import get_dataloader as get_dataloader_general
from datasets.dataloader_remote_sensing import DLSegRemoteSensing

# deterministic inference
from torch.backends import cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

# ========== Configuration Flags ==========
D_VIS = False               # Generate overlay visualizations
D_NOGT = False              # True if ground truth not available (test set)
# ========================================

def ensure_dir(path):
    """Create a full directory path if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path

def mask2rgb(mask, palette):
    """Convert a single-channel mask to RGB using a palette."""
    im = Image.fromarray(mask).convert("P")
    im.putpalette(palette)
    mask_rgb = np.array(im.convert("RGB"))
    return mask_rgb / 255.0

def mask_overlay(mask, image, palette):
    """Overlay prediction mask on original image."""
    mask_rgb = mask2rgb(mask, palette)
    return 0.3 * image + 0.7 * mask_rgb


def resize_mask_nearest(mask, pil_size):
    """Resize a class-id mask to PIL (width, height) size with nearest neighbor."""
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(pil_size, Image.NEAREST))


def is_remote_sensing_infer(args):
    name = f"{args.dataloader}_{args.infer_list}".lower()
    return any(key in name for key in ("potsdam", "vaihingen", "loveda"))


def load_snapshot(model, snapshot_path):
    assert os.path.isfile(snapshot_path), f"Snapshot not found: {snapshot_path}"
    print(f"Loading snapshot from: {snapshot_path}")
    checkpoint = torch_load_compat(
        snapshot_path, map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Snapshot key used: model")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Snapshot key used: state_dict")
        else:
            state_dict = checkpoint
            print("Snapshot has no model/state_dict key; using checkpoint as state_dict")
    else:
        state_dict = checkpoint
        print("Snapshot object is not a dict; using it as state_dict")

    state_dict = convert_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(f"Snapshot loaded. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    if missing:
        print("First missing keys:", missing[:10])
    if unexpected:
        print("First unexpected keys:", unexpected[:10])
    return checkpoint.get("stage") if isinstance(checkpoint, dict) else None


def inference_dataset_config(config, checkpoint_stage):
    if checkpoint_stage not in ("source", "adapt"):
        return config
    dataset_config = copy.deepcopy(config)
    dataset_config.MODEL.BASELINE = checkpoint_stage == "source"
    return dataset_config


class SegmentationMeter:
    """Accumulate per-class semantic segmentation metrics."""
    def __init__(self, num_classes, class_names=None, ignore_index=255):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.tp = np.zeros(num_classes, dtype=np.float64)
        self.fp = np.zeros(num_classes, dtype=np.float64)
        self.fn = np.zeros(num_classes, dtype=np.float64)

    def update(self, pred, target):
        pred = np.asarray(pred, dtype=np.int64)
        target = np.asarray(target, dtype=np.int64)
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")

        valid = target != self.ignore_index
        valid &= target >= 0
        valid &= target < self.num_classes
        if not valid.any():
            return

        pred = pred[valid]
        target = target[valid]
        pred_valid = (pred >= 0) & (pred < self.num_classes)
        pred = pred[pred_valid]
        target = target[pred_valid]

        for class_id in range(self.num_classes):
            pred_i = pred == class_id
            target_i = target == class_id
            self.tp[class_id] += np.logical_and(pred_i, target_i).sum()
            self.fp[class_id] += np.logical_and(pred_i, ~target_i).sum()
            self.fn[class_id] += np.logical_and(~pred_i, target_i).sum()

    def summarise(self):
        eps = 1e-10
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        support = self.tp + self.fn
        valid_classes = support > 0
        return iou, precision, recall, f1, support, valid_classes

    def format_report(self):
        iou, precision, recall, f1, support, valid_classes = self.summarise()
        lines = []
        lines.append("")
        lines.append("=" * 86)
        lines.append("Per-class segmentation metrics")
        lines.append("-" * 86)
        lines.append(f"{'ID':>3s} {'Class':<20s} {'IoU':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Pixels':>12s}")
        lines.append("-" * 86)
        for idx, name in enumerate(self.class_names):
            lines.append(
                f"{idx:>3d} {name:<20s} "
                f"{iou[idx]:>10.4f} {precision[idx]:>10.4f} "
                f"{recall[idx]:>10.4f} {f1[idx]:>10.4f} {int(support[idx]):>12d}"
            )
        lines.append("-" * 86)
        if valid_classes.any():
            lines.append(f"mIoU      : {iou[valid_classes].mean():.4f}")
            lines.append(f"mPrecision: {precision[valid_classes].mean():.4f}")
            lines.append(f"mRecall   : {recall[valid_classes].mean():.4f}")
            lines.append(f"mF1       : {f1[valid_classes].mean():.4f}")
            lines.append(f"Pixel Acc : {self.tp.sum() / max(1.0, support.sum()):.4f}")
        else:
            lines.append("No valid ground-truth pixels were found; metrics are unavailable.")
        lines.append("=" * 86)
        return "\n".join(lines)


class ResultWriter:
    def __init__(self, palette, out_path, verbose=False, raw=False):
        self.palette = palette
        self.out_path = out_path
        self.color_dir = ensure_dir(os.path.join(out_path, "color_masks"))
        self.verbose = verbose
        self.raw = raw

    def save(self, image, gt_mask, masks, im_name):
        """
        image: [C,H,W] tensor, denormalized (0-1)
        gt_mask: [H,W] tensor (may be all 255 if not available)
        masks: [C,H,W] numpy array (logits)
        im_name: string (without extension)
        """
        masks_raw = masks
        pred = np.argmax(masks_raw, 0).astype(np.uint8)  # trainId
        # Save prediction as PNG (trainId)
        filepath = os.path.join(self.out_path, im_name + '.png')
        imageio.imwrite(filepath, pred)

        color_mask = Image.fromarray(pred, mode="P")
        color_mask.putpalette(self.palette)
        color_mask.convert("RGB").save(os.path.join(self.color_dir, im_name + '.png'))

        # Save raw logits if requested
        if self.raw:
            raw_dir = os.path.join(self.out_path, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            np.savez_compressed(os.path.join(raw_dir, im_name), raw=masks_raw)

        # Generate overlay visualization if verbose
        if self.verbose:
            vis_dir = os.path.join(self.out_path, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            # Convert image tensor to numpy HWC (0-1)
            img_np = image.permute(1,2,0).cpu().numpy()
            # Ground truth mask (if available)
            if not D_NOGT and gt_mask is not None:
                gt_np = gt_mask.cpu().numpy().astype(np.uint8)
                # Stack prediction and ground truth horizontally
                masks_vis = np.concatenate([pred, gt_np], axis=1)
                images_vis = np.concatenate([img_np, img_np], axis=1)
            else:
                masks_vis = pred
                images_vis = img_np
            overlay = mask_overlay(masks_vis, images_vis, self.palette)
            filepath_vis = os.path.join(vis_dir, im_name + '.png')
            imageio.imwrite(filepath_vis, (overlay * 255).astype(np.uint8))


def summarise_prediction_maps(score_maps, pred, gt_mask=None, ignore_index=255):
    """Compute compact uncertainty statistics for downstream plotting."""
    score_maps = np.asarray(score_maps, dtype=np.float32)
    channel_sum = score_maps.sum(axis=0, keepdims=True)
    is_probability_map = (
        np.nanmin(score_maps) >= -1e-6
        and np.nanmax(score_maps) <= 1.0 + 1e-6
        and np.allclose(float(np.nanmean(channel_sum)), 1.0, atol=1e-3)
    )
    if is_probability_map:
        probs = score_maps / np.clip(channel_sum, 1e-12, None)
    else:
        logits = score_maps - score_maps.max(axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.clip(exp_logits.sum(axis=0, keepdims=True), 1e-12, None)

    pred = np.asarray(pred, dtype=np.int64)
    top1 = probs.max(axis=0)
    sorted_probs = np.sort(probs, axis=0)
    top2 = sorted_probs[-2] if probs.shape[0] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=0)

    stats = {
        "pixels": int(pred.size),
        "mean_confidence": float(top1.mean()),
        "std_confidence": float(top1.std()),
        "mean_margin": float(margin.mean()),
        "std_margin": float(margin.std()),
        "mean_entropy": float(entropy.mean()),
        "std_entropy": float(entropy.std()),
        "max_entropy": float(entropy.max()),
    }

    if gt_mask is not None:
        gt = np.asarray(gt_mask, dtype=np.int64)
        valid = (gt != ignore_index) & (gt >= 0) & (gt < probs.shape[0])
        stats["valid_pixels"] = int(valid.sum())
        if valid.any():
            correct = pred[valid] == gt[valid]
            stats["pixel_accuracy"] = float(correct.mean())
            stats["mean_confidence_correct"] = float(top1[valid][correct].mean()) if correct.any() else float("nan")
            stats["mean_confidence_error"] = float(top1[valid][~correct].mean()) if (~correct).any() else float("nan")
            stats["mean_entropy_correct"] = float(entropy[valid][correct].mean()) if correct.any() else float("nan")
            stats["mean_entropy_error"] = float(entropy[valid][~correct].mean()) if (~correct).any() else float("nan")
            boundary = compute_label_boundary_mask(gt, valid=valid, dilation=3)
            stats["boundary_pixels"] = int(boundary.sum())
            stats["mean_boundary_entropy"] = float(entropy[boundary].mean()) if boundary.any() else float("nan")
            stats["mean_boundary_confidence"] = float(top1[boundary].mean()) if boundary.any() else float("nan")
        else:
            stats["pixel_accuracy"] = float("nan")
            stats["mean_confidence_correct"] = float("nan")
            stats["mean_confidence_error"] = float("nan")
            stats["mean_entropy_correct"] = float("nan")
            stats["mean_entropy_error"] = float("nan")
            stats["boundary_pixels"] = 0
            stats["mean_boundary_entropy"] = float("nan")
            stats["mean_boundary_confidence"] = float("nan")

    return stats


def compute_label_boundary_mask(mask, valid=None, dilation=3):
    if valid is None:
        valid = mask != 255
    edge = np.zeros(mask.shape, dtype=bool)
    center = mask[1:-1, 1:-1]
    center_valid = valid[1:-1, 1:-1]
    for neighbor, neighbor_valid in (
        (mask[:-2, 1:-1], valid[:-2, 1:-1]),
        (mask[2:, 1:-1], valid[2:, 1:-1]),
        (mask[1:-1, :-2], valid[1:-1, :-2]),
        (mask[1:-1, 2:], valid[1:-1, 2:]),
    ):
        edge[1:-1, 1:-1] |= center_valid & neighbor_valid & (center != neighbor)

    dilated = edge
    for _ in range(max(0, dilation)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        grown = dilated.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grown |= padded[1 + dx:1 + dx + dilated.shape[0], 1 + dy:1 + dy + dilated.shape[1]]
        dilated = grown
    return dilated & valid

def convert_dict(state_dict):
    """Remove 'module.' prefix from keys."""
    new_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict


def load_original_remote_gt(dataset, index):
    mask_path = dataset.masks[index]
    if mask_path is None:
        return None
    return np.asarray(dataset._convert_label(Image.open(mask_path)))


def make_original_size_prediction(pred, dataset, index):
    original_size = Image.open(dataset.images[index]).size
    return resize_mask_nearest(pred, original_size)


def save_original_size_prediction(pred_original, output_dir, image_name):
    original_dir = ensure_dir(os.path.join(output_dir, "original_size"))
    imageio.imwrite(os.path.join(original_dir, image_name + ".png"), pred_original.astype(np.uint8))

if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    if args.infer_list is None:
        raise ValueError("infer_val.py requires --infer-list")
    if args.resume is None:
        raise ValueError("infer_val.py requires --resume with a trained checkpoint")
    if args.mask_output_dir is None:
        raise ValueError("infer_val.py requires --mask-output-dir")
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # Create output directories
    ensure_dir(args.mask_output_dir)
    if D_VIS:
        ensure_dir(os.path.join(args.mask_output_dir, "vis"))
    if args.save_raw_logits:
        ensure_dir(os.path.join(args.mask_output_dir, "raw"))
    # if cfg.DATASET.USE_CLASS_WEIGHTS:
    #     class_weights = torch.tensor(cfg.DATASET.CLASS_WEIGHTS, dtype=torch.float).cuda(gpu)
    #     print("Using class weights:", class_weights)
    #     criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights, reduction="none")
    # else:
    #     criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
    num_classes = get_num_classes(args)
    class_names = get_class_names(args)

    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Keep the model wrapper selected by the experiment configuration so old
    # source/adaptation checkpoints retain their established loading behavior.
    model = get_model(cfg, 0, num_classes=num_classes, criterion=None)
    checkpoint_stage = load_snapshot(model, args.resume)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = nn.DataParallel(model).cuda()

    # Determine dataset type and create dataset instance.
    # For remote sensing, the concrete split is carried by --infer-list
    # (e.g. val_loveda_rural), so --dataloader may be simply "loveda".
    if is_remote_sensing_infer(args):
        # Use remote sensing dataset loader (supports label conversion and palette)
        dataset_config = inference_dataset_config(cfg, checkpoint_stage)
        infer_dataset = DLSegRemoteSensing(
            dataset_config, args.infer_list, root=dataset_config.DATASET.ROOT
        )
        # For remote sensing, we need to manually extract image names from dataset.images_IRRG
        image_names = [os.path.splitext(os.path.basename(p))[0] for p in infer_dataset.images]
        has_gt = all(m is not None for m in infer_dataset.masks)  # check if ground truth exists
        if not has_gt:
            print("No ground truth masks found. Setting D_NOGT=True")
            D_NOGT = True
    else:
        # Use general inference loader (e.g., Cityscapes)
        infer_dataset = get_dataloader_general(args.dataloader, cfg, args.infer_list)
        image_names = None  # will be provided by dataset's __getitem__
        has_gt = True

    # Get palette for visualization (remote sensing dataset provides get_palette)
    if hasattr(infer_dataset, 'get_palette'):
        palette = infer_dataset.get_palette()
    else:
        # Fallback to a default grayscale palette (not ideal but works)
        palette = list(range(256)) * 3
        palette = tuple(palette)

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.workers,
                                   pin_memory=True, batch_size=1)

    timer = Timer()
    N = len(infer_data_loader)
    writer = ResultWriter(palette, args.mask_output_dir, verbose=D_VIS, raw=args.save_raw_logits)
    meter = None if D_NOGT else SegmentationMeter(num_classes, class_names)
    meter_original = None
    logit_stats_rows = []
    if args.eval_original_size:
        if D_NOGT or image_names is None:
            print("Original-size evaluation requested but original remote-sensing GT is unavailable; skipping.")
        else:
            meter_original = SegmentationMeter(num_classes, class_names)
    if args.save_original_size:
        ensure_dir(os.path.join(args.mask_output_dir, "original_size"))

    for idx, batch in enumerate(tqdm(infer_data_loader)):
        # Handle different dataset output formats
        if image_names is not None:
            # Remote sensing dataset: returns (image, mask)
            image, gt_mask = batch
            im_name = image_names[idx]
        else:
            # General dataset: returns (image, gt_mask, im_name)
            image, gt_mask, im_name = batch
            im_name = im_name[0]

        with torch.no_grad():
            # Use the student branch for inference.
            _, logits = model(image, teacher=False)
            masks_pred = F.softmax(logits, 1)

        pred = torch.argmax(masks_pred, dim=1)
        pred_np = pred[0].cpu().numpy().astype(np.uint8)
        gt_np = gt_mask[0].cpu().numpy()
        if meter is not None:
            meter.update(pred_np, gt_np)

        if args.save_logit_stats:
            row = {
                "image": im_name,
                "has_gt": int(not D_NOGT),
            }
            row.update(summarise_prediction_maps(
                masks_pred[0].cpu().numpy(),
                pred_np,
                gt_np if not D_NOGT else None
            ))
            logit_stats_rows.append(row)

        pred_original = None
        if args.save_original_size or meter_original is not None:
            if image_names is not None:
                pred_original = make_original_size_prediction(pred_np, infer_dataset, idx)
                if args.save_original_size:
                    save_original_size_prediction(pred_original, args.mask_output_dir, im_name)
            else:
                print("Original-size saving is only implemented for remote-sensing inference.")

        if meter_original is not None:
            gt_original = load_original_remote_gt(infer_dataset, idx)
            if gt_original is not None and pred_original is not None:
                meter_original.update(pred_original, gt_original)

        # Denormalize image for visualization
        image_denorm = infer_dataset.denorm(image)

        writer.save(image_denorm[0], gt_mask[0], masks_pred[0].cpu().numpy(), im_name)

        timer.update_progress(float(idx + 1) / N)
        if idx % 50 == 0:
            msg = f"Finish time: {timer.str_est_finish()}"
            tqdm.write(msg)
            sys.stdout.flush()
            if D_VIS and idx > 0:
                print("\nSleeping 60 seconds...")
                time.sleep(90.)

    if meter is not None:
        report = meter.format_report()
        print(report)
        metrics_path = os.path.join(args.mask_output_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"Saved metrics to: {metrics_path}")
    else:
        print("No ground truth masks found. Predictions were saved without metric evaluation.")

    if meter_original is not None:
        report_original = meter_original.format_report()
        report_original = report_original.replace(
            "Per-class segmentation metrics",
            "Per-class segmentation metrics (original image size)"
        )
        print(report_original)
        metrics_original_path = os.path.join(args.mask_output_dir, "metrics_original_size.txt")
        with open(metrics_original_path, "w", encoding="utf-8") as f:
            f.write(report_original + "\n")
        print(f"Saved original-size metrics to: {metrics_original_path}")

    if args.save_logit_stats and logit_stats_rows:
        stats_path = os.path.join(args.mask_output_dir, "logit_stats.csv")
        fieldnames = list(logit_stats_rows[0].keys())
        with open(stats_path, "w", encoding="utf-8", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(logit_stats_rows)
        print(f"Saved logit stats to: {stats_path}")

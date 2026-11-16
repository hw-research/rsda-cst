"""Geometry integrity regularization for pseudo-labeled target images."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_boundary_weight(teacher_probs, sobel_x, sobel_y, ignore_mask=None):
    """Compute probability-domain Sobel weights without padding-induced edges."""
    channels = teacher_probs.shape[1]
    content_mask = teacher_probs.sum(dim=1) > 0
    if ignore_mask is not None:
        content_mask = content_mask & (~ignore_mask)
    padded_probs = F.pad(teacher_probs, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(
        padded_probs, sobel_x.expand(channels, 1, 3, 3), groups=channels
    )
    grad_y = F.conv2d(
        padded_probs, sobel_y.expand(channels, 1, 3, 3), groups=channels
    )
    # Only a complete valid 3x3 stencil can provide a semantic boundary signal.
    invalid_neighborhood = F.max_pool2d(
        (~content_mask).unsqueeze(1).to(teacher_probs.dtype), 3, stride=1, padding=1
    )
    gradient_energy = (grad_x ** 2 + grad_y ** 2) * (invalid_neighborhood == 0)
    gradient = torch.sqrt(gradient_energy + 1e-6)
    weight = (1.0 + gradient.max(dim=1)[0]) * content_mask.to(teacher_probs.dtype)

    positive = weight > 0
    if positive.any():
        mean_weight = weight[positive].mean()
        if torch.isfinite(mean_weight) and mean_weight > 0:
            weight = weight / (mean_weight + 1e-8)
    return weight


def geometry_integrity_loss(pred_logits, teacher_probs, pseudo_labels,
                            num_classes, sobel_x, sobel_y,
                            boundary_loss_weight, dice_loss_weight,
                            ignore_mask=None):
    """Combine boundary-weighted cross-entropy with class-balanced Dice loss."""
    pred_probs = F.softmax(pred_logits, dim=1)
    boundary_weight = compute_boundary_weight(
        teacher_probs, sobel_x, sobel_y, ignore_mask
    )

    content_mask = teacher_probs.detach().sum(dim=1) > 0
    valid_mask = (pseudo_labels != 255) & content_mask
    if ignore_mask is not None:
        valid_mask = valid_mask & (~ignore_mask)
    valid_weight = boundary_weight * valid_mask.to(boundary_weight.dtype)
    coverage = (
        valid_mask.to(pred_logits.dtype).sum()
        / content_mask.to(pred_logits.dtype).sum().clamp_min(1.0)
    )

    log_probs = F.log_softmax(pred_logits, dim=1)
    ce_per_pixel = F.nll_loss(
        log_probs, pseudo_labels, ignore_index=255, reduction="none"
    )
    ce_loss = (ce_per_pixel * valid_weight).sum() / valid_weight.sum().clamp_min(1e-8)

    valid_mask = valid_mask.float().unsqueeze(1)

    target_one_hot = F.one_hot(
        pseudo_labels.clamp(min=0, max=num_classes - 1), num_classes=num_classes
    ).permute(0, 3, 1, 2).float() * valid_mask

    intersection = (pred_probs * target_one_hot).sum(dim=(0, 2, 3))
    union = (
        (pred_probs * valid_mask).sum(dim=(0, 2, 3))
        + target_one_hot.sum(dim=(0, 2, 3))
    )
    dice_per_class = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
    active_classes = target_one_hot.sum(dim=(0, 2, 3)) > 0
    dice_loss = (
        dice_per_class[active_classes].mean()
        if active_classes.any()
        else dice_per_class.sum() * 0.0
    )
    # DA-SAC's consistency loss also decreases when fewer pixels retain pseudo labels.
    return coverage * (boundary_loss_weight * ce_loss + dice_loss_weight * dice_loss)

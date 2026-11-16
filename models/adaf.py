"""Adaptive entropy-aware fusion of aligned teacher predictions."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def entropy_aware_fusion(probs, num_views, temperature, min_temperature,
                         curriculum_progress=None):
    batch_views, channels, height, width = probs.shape
    if num_views is None or num_views <= 0 or batch_views % num_views != 0:
        raise ValueError("ADAF expects a batch arranged as [groups * views, C, H, W]")

    probs_by_view = probs.view(-1, num_views, channels, height, width)
    valid_views = probs_by_view.sum(dim=2) > 0
    entropy = -(probs_by_view * torch.log(probs_by_view + 1e-8)).sum(dim=2)

    if curriculum_progress is None:
        current_temperature = float(temperature)
    else:
        progress = min(max(float(curriculum_progress), 0.0), 1.0)
        current_temperature = temperature * (1.0 - progress) + min_temperature * progress
        current_temperature = max(float(current_temperature), float(min_temperature))
    if current_temperature <= 0:
        raise ValueError("ADAF temperature must be positive")

    logits = (-entropy / current_temperature).masked_fill(
        ~valid_views, torch.finfo(entropy.dtype).min
    )
    weights = F.softmax(logits, dim=1)
    fused = (probs_by_view * weights.unsqueeze(2)).sum(dim=1)
    fused = fused.unsqueeze(1).expand(-1, num_views, -1, -1, -1)
    fused = fused.reshape(batch_views, channels, height, width)

    valid = valid_views.any(dim=1, keepdim=True).to(dtype=probs.dtype)
    valid = valid.expand(-1, num_views, -1, -1).reshape(batch_views, 1, height, width)
    return fused, valid

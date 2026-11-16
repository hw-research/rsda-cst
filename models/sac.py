"""
Copyright (c) 2021 TU Darmstadt
Modified & Improved (2026-06-17):
- Fixed visualize: teacher_conf kept full batch (no slicing) to preserve dimensions
- Restored diags in _refine for teacher_aligned visualization
- Binarize refined_valid to preserve full valid region
- Hard pseudo-labels auto-ignore zero-prob pixels (fill regions)
- Fixed teacher momentum: use fixed EMA (no dynamic)
- Fixed GIR Dice loss: equal class weights (no running_conf weighting)
- Fixed running_conf update: uses valid mask from aligned views
- Fixed gather: no truncation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from models.basenet import BaseNet
from models.adaf import entropy_aware_fusion
from models.gir import compute_boundary_weight, geometry_integrity_loss


class SAC_Baseline(BaseNet):
    def __init__(self, cfg, backbone, rank, **kwargs):
        super(SAC_Baseline, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        self.rank = rank
        if "criterion" in kwargs:
            self.criterion = kwargs["criterion"]

    def forward(self, x=None, y=None, x2=None, use_teacher=False, update_teacher=False):
        return self.backbone(x, y)

    def parameter_groups(self, base_lr, wd):
        return self.backbone.parameter_groups(base_lr, wd)


class SAC(SAC_Baseline):
    def __init__(self, cfg, backbone, slow_copy, rank, **kwargs):
        super(SAC, self).__init__(cfg, backbone, rank, **kwargs)
        self.cfg = cfg
        self.backbone = backbone
        self.use_adaf = getattr(cfg.MODEL, 'USE_ADAF', True)
        self.pool_func = None if self.use_adaf else self._get_op(cfg.MODEL.CONF_POOL)

        running_conf = torch.zeros(kwargs["num_classes"])
        self.register_buffer("running_conf", running_conf)

        self.slow_net = slow_copy
        self.slow_net.eval()
        for p in self.slow_net.parameters():
            p.requires_grad = False

        self.register_buffer("slow_init", torch.Tensor([False]))
        self.num_classes = kwargs["num_classes"]

        # ---------- ADAF ----------
        if self.use_adaf:
            self.adaf_temp = getattr(cfg.MODEL, 'ADAF_TEMP', 1.5)
            self.adaf_temp_min = getattr(cfg.MODEL, 'ADAF_TEMP_MIN', 0.5)

        # ---------- GIR ----------
        self.use_gir = getattr(cfg.MODEL, 'USE_GIR', False)
        if self.use_gir:
            self.gir_boundary_weight = getattr(cfg.MODEL, 'GIR_BOUNDARY_WEIGHT', 0.2)
            self.gir_dice_weight = getattr(cfg.MODEL, 'GIR_DICE_WEIGHT', 0.2)
            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                                    dtype=torch.float32).view(1,1,3,3)
            sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                                    dtype=torch.float32).view(1,1,3,3)
            self.register_buffer("sobel_x", sobel_x, persistent=True)
            self.register_buffer("sobel_y", sobel_y, persistent=True)

        self.lower_bound_init = getattr(cfg.MODEL, 'RUN_CONF_LOWER', 0.5)
        self.lower_bound_final = getattr(cfg.MODEL, 'RUN_CONF_UPPER', 0.95)

    def _get_op(self, name):
        op_name = "_{}".format(name)
        assert hasattr(self, op_name), f"Operation {name} not found"
        return getattr(self, op_name)

    # ========================== 教师动量更新（固定动量） ==========================
    @torch.no_grad()
    def _momentum_update(self, update=False, teacher_probs=None, ignore_mask=None):
        if not self.slow_init[0]:
            print(">>> Re-initialising teacher from student")
            self.running_conf.data.fill_(self.cfg.MODEL.THRESHOLD_BETA)
            self.slow_init[0] = True
            self.slow_net.load_state_dict(self.backbone.state_dict())
            return torch.Tensor([0.]).type_as(self.running_conf)

        effective_momentum = self.cfg.MODEL.NET_MOMENTUM
        diff_sum = 0.0
        total_norm = 0.0
        slow_net_dict = self.slow_net.state_dict()
        backbone_dict = self.backbone.state_dict()
        for key, val in backbone_dict.items():
            if key.split(".")[-1] in ("weight", "bias", "running_mean", "running_var"):
                diff = torch.norm(slow_net_dict[key] - val)
                total_norm += torch.norm(val)
                if update:
                    slow_net_dict[key].mul_(effective_momentum)
                    slow_net_dict[key].add_(val * (1. - effective_momentum))
                diff_sum += diff
        rel_diff = diff_sum / (total_norm + 1e-8)
        if update and self.rank == 0:
            print(f"[Teacher Update] momentum={effective_momentum:.4f}, relative diff={rel_diff.item():.4f}")
        return rel_diff.view(1)

    # ========================== 运行置信度更新（仅有效区域） ==========================
    @torch.no_grad()
    def _update_running_conf(self, probs, valid_mask, tolerance=1e-8):
        B, C, H, W = probs.size()
        valid_pixels = valid_mask.sum()
        if valid_pixels > 0:
            probs_avg = (probs * valid_mask).sum(dim=(0, 2, 3)) / valid_pixels.clamp(min=1.0)
        else:
            probs_avg = torch.zeros(C, device=probs.device)
        update_index = probs_avg > tolerance
        new_index = update_index & (self.running_conf == self.cfg.MODEL.THRESHOLD_BETA)
        self.running_conf.data[new_index] = probs_avg[new_index]
        self.running_conf.data.mul_(self.cfg.MODEL.STAT_MOMENTUM)
        self.running_conf.data.add_((1 - self.cfg.MODEL.STAT_MOMENTUM) * probs_avg)
        self.running_conf.data.clamp_(max=0.95)

    # ========================== 损失函数 ==========================
    def _focal_ce_conf(self, logits, pseudo_gt, teacher_probs, p=3, pixel_weight=None):
        focal_weight = (1 - self.running_conf.detach().clamp(0.)) ** p
        loss_ce = F.cross_entropy(logits, pseudo_gt, weight=focal_weight,
                                  ignore_index=255, reduction="none")
        if pixel_weight is not None:
            if pixel_weight.dim() == 4:
                pixel_weight = pixel_weight.squeeze(1)
            loss_ce = loss_ce * pixel_weight

        if teacher_probs is not None:
            if teacher_probs.dim() == 4:
                teacher_probs = teacher_probs.squeeze(1)
            weighted_loss = loss_ce * teacher_probs
            valid_count = (teacher_probs > 0).sum() + 1e-8
            loss = weighted_loss.sum() / valid_count
        else:
            loss = loss_ce.mean()

        with torch.no_grad():
            C = logits.size(1)
            B, H, W = loss_ce.size()
            loss_per_class = torch.zeros_like(logits)
            loss_idx = pseudo_gt.clone()
            loss_idx[pseudo_gt == 255] = 0
            loss_per_class.scatter_(1, loss_idx[:, None], loss_ce[:, None])
            loss_per_class = loss_per_class.view(B, C, -1).mean(-1).mean(0)
        return loss, loss_per_class

    def _threshold_discount(self):
        return 1. - torch.exp(-self.running_conf / self.cfg.MODEL.THRESHOLD_BETA)

    # ========================== 硬伪标签生成（自动忽略填充区域） ==========================
    @torch.no_grad()
    def _pseudo_labels_probs(self, probs, ignore_augm, discount=True, curriculum=None,
                             stable_mask=None):
        B, C, H, W = probs.size()
        prob_sum = probs.sum(dim=1, keepdim=True)
        valid_mask = prob_sum > 1e-6

        max_conf, max_idx = probs.max(1, keepdim=True)
        # DA-SAC uses the strongest predicted pixel of each class in each image.
        probs_peaks = torch.zeros_like(probs)
        probs_peaks.scatter_(1, max_idx, max_conf * valid_mask)
        top_peaks = probs_peaks.view(B, C, -1).amax(dim=-1)
        top_peaks *= self.cfg.MODEL.RUN_CONF_UPPER
        if discount:
            top_peaks *= self._threshold_discount().view(1, C)
        thresh = top_peaks[:, :, None, None].expand(B, C, H, W)
        thresh = torch.gather(thresh, 1, max_idx)

        if curriculum is not None:
            current_lower = self.lower_bound_init + curriculum * (self.lower_bound_final - self.lower_bound_init)
            current_lower = min(current_lower, self.lower_bound_final)
        else:
            current_lower = self.cfg.MODEL.RUN_CONF_LOWER
        thresh.clamp_(min=current_lower)

        keep_mask = (max_conf > thresh) & valid_mask
        if stable_mask is not None:
            keep_mask = keep_mask & stable_mask[:, None]
        pseudo_labels = max_idx.clone()
        pseudo_labels[~keep_mask] = 255
        pseudo_labels = pseudo_labels.squeeze(1)
        if ignore_augm is not None:
            pseudo_labels = pseudo_labels.masked_fill(ignore_augm, 255)
        return pseudo_labels, max_conf, max_idx

    def _entropy(self, probs, eps=1e-5):
        p = (probs + eps) / (1 + eps)
        return -(p * torch.log(p)).sum(1, keepdim=True)

    @torch.no_grad()
    def _gather(self, tensor, T):
        if not dist.is_initialized() or self.world_size == 1:
            return tensor
        out_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(out_list, tensor)
        return torch.cat(out_list, dim=0)

    @torch.no_grad()
    def _avg_pool(self, probs, T, tolerance=0.1):
        T0 = min(T, probs.size(0))
        probs = self._gather(probs, T)
        _, C, H, W = probs.size()
        probs_T = probs.view(-1, T, C, H, W)
        probs_sum = probs_T.sum(dim=1)
        view_count = probs_sum.sum(dim=1, keepdim=True)
        probs_avg = probs_sum / view_count.clamp_min(1e-3)
        fused = probs_avg.unsqueeze(1).expand(-1, T0, -1, -1, -1).reshape(-1, C, H, W)
        valid = (view_count > tolerance).to(probs.dtype)
        valid = valid.unsqueeze(1).expand(-1, T0, -1, -1, -1).reshape(-1, 1, H, W)
        return fused, valid

    @torch.no_grad()
    def _minentropy_pool(self, probs, T, tolerance=0.1):
        probs = self._gather(probs, T)
        BT, C, H, W = probs.size()
        entropy = self._entropy(probs)
        entropy = entropy.masked_fill(
            probs.sum(dim=1, keepdim=True) <= tolerance,
            torch.finfo(entropy.dtype).max,
        )
        entropy_T = entropy.view(-1, T, 1, H, W)
        entropy_min = entropy_T.argmin(1, keepdim=True)
        entropy_min = entropy_min.expand(-1, -1, C, -1, -1)
        probs_T = probs.view(-1, T, C, H, W)
        masks = probs_T.sum(1, keepdim=True).sum(2, keepdim=True) > tolerance
        selected = probs_T.gather(1, entropy_min)
        selected = selected.expand(-1, T, -1, -1, -1)
        masks = masks.expand(-1, T, -1, -1, -1).type_as(probs_T)
        return selected.reshape(BT, C, H, W), masks.reshape(BT, 1, H, W)

    @torch.no_grad()
    def _adaf_pool(self, probs, T, curriculum_progress=None):
        return entropy_aware_fusion(
            probs,
            num_views=T,
            temperature=self.adaf_temp,
            min_temperature=self.adaf_temp_min,
            curriculum_progress=curriculum_progress,
        )

    # ========================== GIR ==========================
    @torch.no_grad()
    def _compute_boundary_weight(self, teacher_probs, ignore_mask=None):
        return compute_boundary_weight(
            teacher_probs, self.sobel_x, self.sobel_y, ignore_mask
        )

    def _gir_loss(self, pred_logits, teacher_probs, pseudo_labels, ignore_mask=None):
        if not self.use_gir or pseudo_labels is None:
            return torch.tensor(0.0, device=pred_logits.device)

        return geometry_integrity_loss(
            pred_logits=pred_logits,
            teacher_probs=teacher_probs,
            pseudo_labels=pseudo_labels,
            num_classes=self.num_classes,
            sobel_x=self.sobel_x,
            sobel_y=self.sobel_y,
            boundary_loss_weight=self.gir_boundary_weight,
            dice_loss_weight=self.gir_dice_weight,
            ignore_mask=ignore_mask,
        )

    # ========================== 多视图融合入口（返回 diags 用于可视化） ==========================
    @torch.no_grad()
    def _refine(self, frames, pred_logits, T, affine, affine_inv,
                ignore_mask, pool=True, curriculum_progress=None, debug=True):
        _, C, h, w = frames.size()
        pred_logits = F.interpolate(pred_logits, (h, w), mode="bilinear", align_corners=True)
        pred_probs = F.softmax(pred_logits, dim=1)

        # Zero-out ignored regions
        valid_input = 1 - ignore_mask[:, None].type_as(pred_probs)
        pred_probs = pred_probs * valid_input

        diags = {}
        if not pool or T is None or T <= 1 or affine is None or affine_inv is None:
            return pred_probs, diags

        # Align views
        affine_grid = F.affine_grid(affine, size=pred_probs.size(), align_corners=False)
        probs_aligned = F.grid_sample(pred_probs, affine_grid, align_corners=False)
        diags["teacher_aligned"] = probs_aligned
        if debug:
            affine_frames = F.affine_grid(affine, size=frames.size(), align_corners=False)
            diags["frames_aligned"] = F.grid_sample(frames, affine_frames, align_corners=False)

        # The predictions above are warped from each view into the shared
        # reference frame with ``affine``. Warp each view's actual valid-pixel
        # mask with that same grid; using affine_inv here mislabels crop/pad
        # regions as valid (notably when RND_ZOOM is enabled).
        valid_view = (~ignore_mask).unsqueeze(1).to(dtype=pred_probs.dtype)
        valid_aligned = F.grid_sample(
            valid_view,
            affine_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        affine_grid_inv = F.affine_grid(affine_inv, size=probs_aligned.size(), align_corners=False)

        # Update running_conf only on valid (aligned) pixels
        if self.training:
            # valid_aligned has one copy per class because it was created from
            # the probability tensor. Confidence statistics must divide by the
            # number of spatial pixels, not by pixels times num_classes.
            valid_pixels = valid_aligned[:, :1]
            self._update_running_conf(probs_aligned, valid_pixels)

        if self.use_adaf:
            refined_aligned, valid = self._adaf_pool(probs_aligned * valid_aligned, T,
                                                      curriculum_progress=curriculum_progress)
        else:
            refined_aligned, valid = self.pool_func(probs_aligned * valid_aligned, T)

        refined = F.grid_sample(refined_aligned, affine_grid_inv, align_corners=False)
        refined_valid = F.grid_sample(valid, affine_grid_inv, align_corners=False)

        # Binarize valid mask to preserve full region after interpolation
        refined_valid = (refined_valid > 0.5).float()
        refined *= refined_valid

        return refined, diags

    # ========================== 主前向函数 ==========================
    def forward(self, x, y=None, x2=None, affine=None, affine_inv=None,
                use_teacher=False, update_teacher=False, reset_teacher=False,
                T=None, teacher=False, adaf_temperature_progress=None,
                pseudo_threshold_progress=None, return_feat=False):
        if y is None:
            if teacher:
                return self.slow_net(x)
            else:
                return self.backbone(x)

        if reset_teacher:
            self.slow_init[0] = False

        ignore_mask = (y == -1)
        y[ignore_mask] = 255

        # Student forward
        student_losses, student_outs = self.backbone(x, y)

        if not use_teacher:
            if update_teacher:
                student_losses["teacher_diff"] = self._momentum_update(True)
            return student_losses, student_outs

        # Teacher forward
        # For a fresh adaptation run (including a source-only initialization),
        # synchronize the teacher before it creates the first pseudo labels.
        # A resumed adaptation checkpoint restores slow_init=True and therefore
        # preserves its saved EMA teacher.
        if not self.slow_init[0]:
            self._momentum_update(False)
        self.slow_net.eval()
        with torch.no_grad():
            teacher_out = self.slow_net(x2)
            if isinstance(teacher_out, tuple):
                slow_logits = teacher_out[0]
                slow_logits_up = teacher_out[1] if len(teacher_out) > 1 else slow_logits
            else:
                slow_logits = teacher_out
                slow_logits_up = slow_logits

        probs_teacher, diags = self._refine(
            x2, slow_logits, T, affine, affine_inv,
            ignore_mask, pool=self.cfg.MODEL.CONF_POOL_ON,
            curriculum_progress=adaf_temperature_progress
        )
        probs_teacher = probs_teacher.detach()
        # Generate hard pseudo-labels and optimize with confidence-weighted CE.
        pseudo_labels, teacher_conf, _ = self._pseudo_labels_probs(
            probs_teacher, ignore_mask, self.cfg.MODEL.CONF_DISCOUNT,
            curriculum=pseudo_threshold_progress,
        )
        content_mask = teacher_conf.squeeze(1) > 0
        pseudo_coverage = (
            ((pseudo_labels != 255) & content_mask).float().sum()
            / content_mask.float().sum().clamp_min(1.0)
        )
        student_losses["pseudo_coverage"] = pseudo_coverage.detach().view(1)
        loss_ce, _ = self._focal_ce_conf(
            student_outs["logits_up"], pseudo_labels,
            teacher_conf, self.cfg.MODEL.FOCAL_P
        )
        student_losses["loss_consistency"] = loss_ce.view(1)
        total_loss = loss_ce
        if self.use_gir:
            gir_loss = self._gir_loss(
                student_outs["logits_up"], probs_teacher,
                pseudo_labels, ignore_mask
            )
            total_loss = total_loss + gir_loss
            student_losses["loss_gir"] = gir_loss.view(1)
        student_losses["self_ce"] = total_loss.view(1)
        student_outs["teacher_conf"] = teacher_conf.detach()

        # Update teacher EMA
        if update_teacher:
            student_losses["teacher_diff"] = self._momentum_update(True)

        # Store outputs for visualization
        student_outs["teacher_init"] = slow_logits_up.detach()
        student_outs["teacher_refined"] = probs_teacher.detach()
        student_outs["teacher_labels"] = pseudo_labels.detach()
        student_outs["running_conf"] = self.running_conf
        student_outs.update(diags)

        student_losses.pop("loss_ce", None)
        return student_losses, student_outs

    def parameter_groups(self, base_lr, wd):
        return self.backbone.parameter_groups(base_lr, wd)

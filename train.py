"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
Modified for Remote Sensing: class weights, cosine annealing (optional), gradient clipping (optional),
signal handling, robust checkpoint loading.
"""

from __future__ import print_function

import os
import copy
import sys
import signal
import numpy as np
import time
import math
import random
import builtins
import setproctitle

from functools import partial

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

from datasets import *
from models import get_model

from base_trainer import BaseTrainer

from opts import get_arguments, prepare_run_directory
from core.config import cfg, cfg_from_file, cfg_from_list
from utils.timer import Timer
from utils.stat_manager import StatManager
from utils.metrics import Jaccard
from utils.sys_tools import find_free_port
from utils.torch_compat import torch_load_compat
from utils.validation_report import write_source_target_report

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DEBUG = False
EVALUATE = not DEBUG
EVALUATE_SOURCE = False
SNAPSHOT = not DEBUG


def epoch_plan(cfg, stage, start_epoch, stage_start_epoch=None):
    """Return epoch indices for the selected training stage."""
    if stage not in ("source", "adapt"):
        raise ValueError("Unknown training stage: {}".format(stage))
    count = (cfg.TRAIN.SOURCE_EPOCHS if stage == "source"
             else cfg.TRAIN.ADAPT_EPOCHS)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("TRAIN.{}_EPOCHS must be a positive integer".format(
            "SOURCE" if stage == "source" else "ADAPT"))
    if stage == "source":
        if start_epoch > count:
            raise ValueError("Source checkpoint epoch exceeds TRAIN.SOURCE_EPOCHS")
        return range(start_epoch, count)
    if stage_start_epoch is None:
        stage_start_epoch = start_epoch
    end_epoch = stage_start_epoch + count
    if start_epoch > end_epoch:
        raise ValueError("Adaptation checkpoint epoch exceeds the configured stage end")
    return range(start_epoch, end_epoch)


def cleanup_handler(signum, frame):
    # Keep signal handlers lightweight. Resource cleanup is centralized in the
    # main_worker finally block; doing distributed/CUDA work inside a signal
    # handler can deadlock on Windows.
    print(f"\nReceived signal {signum}, stopping after the current operation...")
    raise KeyboardInterrupt


def set_student_bn_stats_mode(net, freeze):
    """Switch student BN statistics without changing affine trainability."""
    model = net.module if hasattr(net, "module") else net
    for layer in model.backbone.modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d,
                              nn.BatchNorm3d, nn.SyncBatchNorm)):
            layer.train(not freeze)


class TeeStream:
    """Write console output to both the terminal and this run's text log."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class Trainer(BaseTrainer):

    def __init__(self, args, cfg, gpu, ngpus_per_node):
        main_process = args.rank % ngpus_per_node == 0
        super(Trainer, self).__init__(args, cfg, main_process)

        self.gpu = gpu
        self.world_size = args.world_size
        self.device = torch.device("cuda", gpu)
        self.stage_length = (cfg.TRAIN.SOURCE_EPOCHS if args.stage == "source"
                             else cfg.TRAIN.ADAPT_EPOCHS)
        self.abn_target_bn = bool(getattr(cfg.MODEL, "ABN_TARGET_BN", True))

        self.nclass = get_num_classes(args)
        self.classNames = get_class_names(args)
        assert self.nclass == len(self.classNames)

        self.classIndex = {}
        for i, cname in enumerate(self.classNames):
            print("{:>5} -> {}".format(cname, i))
            self.classIndex[cname] = i

        # Data loaders
        train_source, train_target = get_dataloader(args, cfg, 'train', self.nclass, ngpus_per_node)
        self.loader_source, self.sampler_source = train_source
        self.loader_target, self.sampler_target = train_target

        self.denorm = self.loader_source.dataset.denorm

        # Validation loaders
        self.valloaders = get_dataloader(args, cfg, 'val', self.nclass, ngpus_per_node)

        val_sets, self.testset = get_val_sets(cfg.TRAIN.TASK, cfg.TRAIN.TARGET)
        source_val_sets = [val_set for val_set in val_sets if val_set != self.testset]
        if len(source_val_sets) != 1:
            raise ValueError("Expected exactly one source validation set")
        self.source_val_set = source_val_sets[0]
        self.writer_val = {}
        for val_set in val_sets:
            logdir_val = os.path.join(args.logdir, val_set)
            self.writer_val[val_set] = SummaryWriter(logdir_val) if self.main_process else None

        # ---------- Class weights ----------
        if cfg.DATASET.USE_CLASS_WEIGHTS:
            if len(cfg.DATASET.CLASS_WEIGHTS) != self.nclass:
                raise ValueError(
                    "DATASET.CLASS_WEIGHTS has {} entries, but task {} has {} classes".format(
                        len(cfg.DATASET.CLASS_WEIGHTS), cfg.TRAIN.TASK, self.nclass
                    )
                )
            class_weights = torch.tensor(cfg.DATASET.CLASS_WEIGHTS, dtype=torch.float).cuda(gpu)
            print("Using class weights:", class_weights)
            criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights, reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")

        self.net = get_model(cfg, self.gpu, num_classes=self.nclass, criterion=criterion)
        print(self.net)

        # Optimizer
        net_params = self.net.parameter_groups(cfg.MODEL.LR, cfg.MODEL.WEIGHT_DECAY)
        self.optim = self.get_optim(net_params, cfg.MODEL)

        # ---------- Cosine Annealing Scheduler (optional) ----------
        self.use_scheduler = getattr(cfg, 'USE_SCHEDULER', False)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=self.stage_length, eta_min=cfg.MODEL.LR * 0.01
            )
            print("Cosine annealing scheduler initialized (eta_min = {:.2e})".format(cfg.MODEL.LR * 0.01))
        else:
            self.scheduler = None

        # ---------- Gradient clipping ----------
        self.grad_clip_norm = getattr(cfg, 'GRAD_CLIP_NORM', None)

        print("# of params: ", len(list(self.net.parameters())))

        self.fixed_batch = None

        print(">>> Distributed training: ", args.rank, gpu)
        torch.cuda.set_device(gpu)
        self.net.cuda(gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[gpu])

        # Checkpoint management
        self.checkpoint.create_model(self.net, self.optim)
        # ---------- Robust checkpoint loading ----------
        if args.resume is not None:
            try:
                self.start_epoch, self.best_score = self.checkpoint.load(args.resume, self.device)
            except ValueError as e:
                print(f"Warning: Optimizer state mismatch: {e}")
                print("Loading only model weights and resetting optimizer state...")
                checkpoint = torch_load_compat(args.resume, map_location=self.device)
                model_state = checkpoint.get('model', checkpoint.get('state_dict'))
                if model_state is None:
                    raise KeyError("Checkpoint does not contain 'model' or 'state_dict' key")
                self.net.load_state_dict(model_state, strict=False)
                self.start_epoch = checkpoint.get('epoch', 0)
                self.best_score = checkpoint.get('score', checkpoint.get('best_score', 0.0))
                self.checkpoint.loaded_stage = checkpoint.get('stage')
                self.checkpoint.loaded_stage_start_epoch = checkpoint.get('stage_start_epoch')
                self.checkpoint.loaded_selection_metric = checkpoint.get('selection_metric')
                print(f"Successfully loaded model weights from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
            self.best_score = 0.0

        self.stage_start_epoch = 0
        if args.stage == "source" and self.checkpoint.loaded_selection_metric != "target_val_miou":
            self.best_score = -1e16
        if args.stage == "adapt":
            if self.checkpoint.loaded_stage == "adapt":
                self.stage_start_epoch = self.checkpoint.loaded_stage_start_epoch
                if self.stage_start_epoch is None:
                    raise ValueError("Adaptation checkpoint is missing stage_start_epoch")
            else:
                self.stage_start_epoch = self.start_epoch
                if self.checkpoint.loaded_stage is None:
                    print("Checkpoint has no stage metadata; treating it as a source checkpoint.")
                # Source and adaptation runs are separate experiments.
                self.best_score = -1e16
        self.checkpoint.stage = args.stage
        self.checkpoint.stage_start_epoch = self.stage_start_epoch

        print(self.optim)
        print("Checkpoint saving ({}): {}".format(
            args.stage, "best target validation mIoU (up to five checkpoints)"
        ))
        print("RS-AUG: {}".format(
            bool(self.cfg.DATASET.USE_REMOTE_AUG) if not self.cfg.MODEL.BASELINE else False
        ))
        if args.stage == "adapt":
            print("Student BN running statistics: {}".format(
                "frozen" if cfg.MODEL.Target_Freeze_BN else "updating"
            ))
        if self.cfg.MODEL.BASELINE:
            print("ABN target BN statistics: {}".format(
                "enabled (target images update running statistics)"
                if self.abn_target_bn
                else "disabled (source images only)"
            ))
            print("ABN target labels/loss: not used")

        if self.cfg.MODEL.BASELINE:
            self.step_target = partial(self.step, train=False)
        else:
            self.step_target = partial(self._step_target, train=True, visualise=False)

    def step(self, epoch, batch_source, train=False, visualise=False,
             save_batch=False, writer=None, tag="train", **kwargs):
        image, masks_gt = batch_source
        image = image.cuda(self.gpu, non_blocking=True)
        masks_gt = masks_gt.cuda(self.gpu, non_blocking=True)

        losses, logits = self.net(image, masks_gt)

        if train:
            self.optim.zero_grad()
            source_loss = losses["loss_ce"].mean()
            if not torch.isfinite(source_loss):
                raise FloatingPointError("Non-finite source loss at epoch {}".format(epoch))
            source_loss.backward()
            if self.cfg.MODEL.BASELINE:
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                self.optim.step()

        if visualise:
            self._visualise(epoch, image, masks_gt, logits, writer, tag)

        if save_batch:
            self.save_fixed_batch(tag, batch_source)

        losses_ret = {}
        for key, val in losses.items():
            losses_ret[key] = val.mean().item()
        logits["mask_gt"] = masks_gt
        return losses_ret, logits

    @torch.no_grad()
    def _update_target_bn(self, batch_target):
        """Forward unlabeled target images in train mode to update student BN.

        The ABN loader returns an all-ignore placeholder mask for compatibility;
        it is intentionally discarded and never enters a loss computation.
        """
        image = batch_target[0].cuda(self.gpu, non_blocking=True)
        self.net(image)

    def _prep_batch(self, tensor):
        N = self.cfg.TRAIN.NUM_GROUPS
        L = self.cfg.TRAIN.GROUP_SIZE
        assert (N * L) % self.world_size == 0, "Batch size does not fit world size"
        batch_per_gpu = N * L // self.world_size
        tensor = tensor.cuda(self.gpu)
        if batch_per_gpu >= L:
            return tensor.flatten(0, 1)
        T = tensor.size(1)
        assert T == L, "Loaded sequence is incorrect {} vs. {}".format(T, L)
        out_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(out_list, tensor)
        batch_index = self.gpu * batch_per_gpu
        index0 = batch_index // T
        index1 = batch_index % T
        index1_end = index1 + batch_per_gpu
        flat_tensor = out_list[index0].flatten(0, 1)
        return flat_tensor[index1:index1_end]

    def _step_target(self, epoch, batch_target,
                     train=False, update_teacher=False,
                     visualise=False, save_batch=False, writer=None, tag="train_target",
                     **kwargs):
        frames1, frames_gt, frames2, affine, affine_inv = [self._prep_batch(t) for t in batch_target]

        # Forward pass: student (frames1) and teacher (frames2)
        # Model returns only (losses, logits) – prototype alignment fully removed
        losses, logits = self.net(frames1, frames_gt, frames2,
                                  affine, affine_inv, use_teacher=True,
                                  update_teacher=update_teacher,
                                  T=self.cfg.TRAIN.GROUP_SIZE,
                                  adaf_temperature_progress=kwargs.get("adaf_temperature_progress"),
                                  pseudo_threshold_progress=kwargs.get("pseudo_threshold_progress"))

        if train:
            if self.cfg.TRAIN.TARGET_ONLY:
                self.optim.zero_grad()

            # Base self-supervised loss
            loss_total = self.cfg.MODEL.LR_TARGET * losses["self_ce"].mean()
            if not torch.isfinite(loss_total):
                raise FloatingPointError("Non-finite target loss at epoch {}".format(epoch))

            # # Add teacher_diff as regularization (optional)
            # if "teacher_diff" in losses:
            #     loss_total +=losses["teacher_diff"].mean()

            loss_total.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            self.optim.step()
            losses["tgt_total"] = loss_total.detach().view(1)

        if visualise:
            self._visualise(epoch, frames1, frames_gt, logits, writer, tag, image2=frames2)

        if save_batch:
            self.save_fixed_batch(tag, batch_target)

        # Synchronize losses for logging
        for key, val in losses.items():
            if val is not None:
                dist.all_reduce(val)
                val /= self.world_size
                losses[key] = val.item()
            else:
                losses[key] = 0.0

        logits["mask_gt"] = frames_gt
        return losses, logits

    def train_epoch(self, epoch):
        stat = StatManager()
        stat.add_val("loss_ce")
        timer = Timer("[{}] New Epoch: ".format(self.gpu))
        step = partial(self.step, visualise=False)

        # Schedule ADAF temperature and pseudo-label threshold independently.
        adaf_temperature_progress = None
        pseudo_threshold_progress = None
        if (getattr(self.cfg, 'ADAF_TEMP_SCHEDULE_ENABLED', False)
                or getattr(self.cfg, 'PSEUDO_THRESHOLD_SCHEDULE_ENABLED', False)):
            stage_epoch = epoch - self.stage_start_epoch if self.args.stage == "adapt" else epoch
            progress = min(1.0, stage_epoch / (self.stage_length * 0.7))
            if getattr(self.cfg, 'ADAF_TEMP_SCHEDULE_ENABLED', False):
                adaf_temperature_progress = progress
            if getattr(self.cfg, 'PSEUDO_THRESHOLD_SCHEDULE_ENABLED', False):
                pseudo_threshold_progress = progress
            if self.main_process:
                print(">>> Schedule progress: {:.3f} (ADAF temperature: {}; pseudo-label threshold: {})".format(
                    progress,
                    "on" if adaf_temperature_progress is not None else "off",
                    "on" if pseudo_threshold_progress is not None else "off",
                ))

        self.sampler_source.set_epoch(epoch)
        # The ABN source-training stage also consumes target images for BN
        # statistics, so its target sampler must be reshuffled each epoch too.
        self.sampler_target.set_epoch(epoch)
        self.net.train()
        if self.args.stage == "adapt":
            set_student_bn_stats_mode(
                self.net, self.cfg.MODEL.Target_Freeze_BN
            )

        # Both stages pair source and target batches. During source baseline
        # (ABN), target images update BN statistics only when enabled; during
        # adaptation they provide self-training data.
        epoch_batches = zip(self.loader_source, self.loader_target)

        for i, batches in enumerate(epoch_batches):
            save_batch = i == 0
            batch_source, batch_target = batches

            # Source pass
            if not self.cfg.TRAIN.TARGET_ONLY:
                losses_src, _ = step(epoch, batch_source, train=True,
                                     save_batch=save_batch, tag="train")

            # Target pass
            if self.cfg.MODEL.BASELINE:
                if self.abn_target_bn:
                    self._update_target_bn(batch_target)
                losses = dict(losses_src)
            else:
                update_teacher = i % self.cfg.MODEL.NET_MOMENTUM_ITER == 0
                losses, _ = self.step_target(epoch, batch_target,
                                             update_teacher=update_teacher,
                                             save_batch=save_batch,
                                             adaf_temperature_progress=adaf_temperature_progress,
                                             pseudo_threshold_progress=pseudo_threshold_progress,
                                             tag="train_target")

            if not self.cfg.TRAIN.TARGET_ONLY and not self.cfg.MODEL.BASELINE:
                for key, val in losses_src.items():
                    losses["src_{}".format(key)] = val

            if self.main_process:
                src_ce = losses.get("src_loss_ce", losses.get("loss_ce", 0.0))
                tgt_total = losses.get("tgt_total", 0.0)
                stat.update_stats("train_total", src_ce + tgt_total)
                for loss_key, loss_val in losses.items():
                    stat.update_stats(loss_key, loss_val)

                display_interval = max(1, int(self.cfg.LOG.ITER_DISPLAY))
                if i % display_interval == 0:
                    parts = ["Iter [{:04d}]".format(i)]
                    if not self.cfg.TRAIN.TARGET_ONLY:
                        parts.append("src_ce {:.4f}".format(src_ce))
                    if not self.cfg.MODEL.BASELINE:
                        parts.append("tgt_consistency {:.4f}".format(
                            losses["loss_consistency"]))
                        if "loss_gir" in losses:
                            parts.append("tgt_gir {:.4f}".format(losses["loss_gir"]))
                        parts.append("tgt_total {:.4f}".format(tgt_total))
                    parts.append("Im/Sec {:.1f}".format(
                        i * self.cfg.TRAIN.BATCH_SIZE / max(0.01, timer.get_stage_elapsed())))
                    msg = " | ".join(parts)
                    print(msg)
                    sys.stdout.flush()

            if DEBUG and i > 20:
                break

        if self.main_process:
            epoch_metrics = dict(stat.items())
            for name, val in epoch_metrics.items():
                self.writer.add_scalar('train_epoch/{}'.format(name), val, epoch)

            parts = ["Epoch [{:03d}]".format(epoch)]
            src_ce = epoch_metrics.get("src_loss_ce", epoch_metrics.get("loss_ce", 0.0))
            if not self.cfg.TRAIN.TARGET_ONLY:
                parts.append("src_ce {:.4f}".format(src_ce))
            if not self.cfg.MODEL.BASELINE:
                parts.append("tgt_consistency {:.4f}".format(
                    epoch_metrics["loss_consistency"]))
                if "loss_gir" in epoch_metrics:
                    parts.append("tgt_gir {:.4f}".format(epoch_metrics["loss_gir"]))
                parts.append("tgt_total {:.4f}".format(epoch_metrics.get("tgt_total", 0.0)))
            print(" | ".join(parts))

            for ii, l in enumerate(self.optim.param_groups):
                print("Learning rate [{}]: {:.4e}".format(ii, l['lr']))
                self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

        if self.use_scheduler and self.scheduler is not None:
            self.scheduler.step()

        if not self.cfg.TRAIN.TARGET_ONLY:
            self.visualise_results(epoch, self.writer, "train", self.step)
        if not self.cfg.MODEL.BASELINE:
            self.visualise_results(epoch, self.writer_target, "train_target", self.step_target)

    def validation(self, epoch, writer, loader, tag=None, step_func=None, max_iter=None,
                   return_metrics=False):
        ignore_eval_ids = set(self.cfg.VAL.IGNORE_CLASS)
        print("Ignoring IDs from mIoU: ", ignore_eval_ids)

        if step_func is None:
            step_func = self.step
        if max_iter is None:
            max_iter = len(loader)

        stat = StatManager()

        def eval_batch(batch):
            loss, masks = step_func(epoch, batch, train=False, visualise=False)
            for loss_key, loss_val in loss.items():
                stat.update_stats(loss_key, loss_val)
            return masks

        self.net.eval()

        def pp_mask(logits_raw):
            return torch.argmax(logits_raw, 1)

        def ignore_indices(x):
            return torch.Tensor([v for i, v in enumerate(x) if not i in ignore_eval_ids])

        jaccard_stats = {}

        for n, batch in enumerate(loader):
            with torch.no_grad():
                masks_all = eval_batch(batch)
            masks_gt = batch[1]
            masks_gt = masks_gt.view(-1, *masks_gt.size()[-2:])

            for masks_layer, masks_raw in masks_all.items():
                if masks_layer not in ("logits_up", "teacher_init", "teacher_refined", "teacher_labels"):
                    continue
                if masks_layer not in jaccard_stats:
                    jaccard_stats[masks_layer] = Jaccard(self.nclass, self.gpu)
                if masks_layer in ("logits_up", "teacher_init", "teacher_refined"):
                    masks_pred = pp_mask(masks_raw)
                else:
                    masks_pred = masks_raw
                jaccard_stats[masks_layer].add_sample(masks_pred, masks_all["mask_gt"])

            if tag is not None and not self.has_fixed_batch(tag):
                self.save_fixed_batch(tag, batch)
            if n > max_iter:
                break

        checkpoint_score = 0.0
        primary_metrics = None

        if self.main_process:
            for stat_key, stat_val in stat.items():
                writer.add_scalar('all/{}'.format(stat_key), stat_val, epoch)
                print('Loss: {:.4f}'.format(stat_val))

        for mask_layer, jaccard_stat in jaccard_stats.items():
            for ni, className in enumerate(self.classNames):
                dist.all_reduce(jaccard_stat.tps[ni])
                dist.all_reduce(jaccard_stat.fns[ni])
                dist.all_reduce(jaccard_stat.fps[ni])

        if self.main_process:
            for mask_layer, jaccard_stat in jaccard_stats.items():
                print("Layer >>> ", mask_layer)
                jaccards, precision, recall = jaccard_stat.summarise()
                assert len(jaccards) == self.nclass
                class_ious = jaccards.clone()

                for ni, className in enumerate(self.classNames):
                    label = "{}_{:02d}_{}".format(mask_layer, ni, className)
                    writer.add_scalar('%s/IoU' % label, jaccards[ni], epoch)
                    print("IoU_{}: {:.4f}".format(className, jaccards[ni]))
                    writer.add_scalar('%s/Precision' % label, precision[ni], epoch)
                    print("Pr_{}: {:.4f}".format(className, precision[ni]))
                    writer.add_scalar('%s/Recall' % label, recall[ni], epoch)
                    print("Re_{}: {:.4f}".format(className, recall[ni]))

                jaccards = ignore_indices(jaccards)
                precision = ignore_indices(precision)
                recall = ignore_indices(recall)

                print("Averaging {} classes: ".format(len(jaccards)))
                meanIoU = jaccards.mean().item()
                if mask_layer == "logits_up":
                    # F1 = 2 * IoU / (1 + IoU) for the same class confusion counts.
                    class_f1 = 2 * class_ious / (1 + class_ious)
                    included = [index for index in range(self.nclass)
                                if index not in ignore_eval_ids]
                    primary_metrics = {
                        "miou": meanIoU,
                        "mf1": class_f1[included].mean().item(),
                        "ignored_class_ids": sorted(ignore_eval_ids),
                        "classes": [
                            {"name": name, "iou": class_ious[index].item(),
                             "f1": class_f1[index].item()}
                            for index, name in enumerate(self.classNames)
                        ],
                    }
                writer.add_scalar('%s_all/mIoU' % mask_layer, meanIoU, epoch)
                print('IoU: {:.4f}'.format(meanIoU))
                meanPr = precision.mean().item()
                writer.add_scalar('%s_all/Precision' % mask_layer, meanPr, epoch)
                print(' Pr: {:.4f}'.format(meanPr))
                meanRe = recall.mean().item()
                writer.add_scalar('%s_all/Recall' % mask_layer, meanRe, epoch)
                print(' Re: {:.4f}'.format(meanRe))
                checkpoint_score = max(meanIoU, checkpoint_score)

        if tag is not None:
            self.visualise_results(epoch, writer, tag, step_func)
        if return_metrics:
            if self.main_process and primary_metrics is None:
                raise RuntimeError("Validation did not produce logits_up metrics")
            return checkpoint_score, primary_metrics
        return checkpoint_score


def main_worker(gpu, ngpus_per_node, argss, cfg):
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    print("GPU / N ", gpu, ngpus_per_node)
    global args
    args = argss
    setproctitle.setproctitle("DA-SAC | {} | {}".format(args.run, gpu))
    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu
    log_file = None
    stdout, stderr = sys.stdout, sys.stderr
    if args.rank == 0:
        log_file = open(os.path.join(args.logdir, "train.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(stdout, log_file)
        sys.stderr = TeeStream(stderr, log_file)
        print("Training text log:", log_file.name)
    trainer = None
    try:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        trainer = Trainer(args, cfg, gpu, ngpus_per_node)
        main_process = args.rank % ngpus_per_node == 0
        timer = Timer()

        def time_call(func, msg, *args, **kwargs):
            timer.reset_stage()
            val = func(*args, **kwargs)
            print("[{}] ".format(gpu) + msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))
            return val

        log_val = 1 if DEBUG else cfg.LOG.ITER_VAL
        log_train = 1 if DEBUG else cfg.LOG.ITER_TRAIN
        log_target = 1 if DEBUG else cfg.LOG.ITER_TARGET
        epochs = epoch_plan(cfg, args.stage, trainer.start_epoch,
                            trainer.stage_start_epoch)
        print("Stage {}: {} training epochs ({}..{})".format(
            args.stage, len(epochs), epochs.start, epochs.stop - 1
        ))

        def evaluate_epoch(epoch, final=False):
            if EVALUATE:
                if not final and epoch > trainer.start_epoch and epoch % log_target == 0:
                    time_call(trainer.validation, "Target / Train: ", epoch, trainer.writer_target,
                              trainer.loader_target, tag="train_target",
                              step_func=trainer.step_target, max_iter=300)
                if final or (epoch > trainer.start_epoch and epoch % log_val == 0):
                    source_score = None
                    target_score = None
                    target_metrics = None
                    for val_set, loader in trainer.valloaders.items():
                        is_target_report = args.stage == "source" and val_set == trainer.testset
                        result = time_call(trainer.validation,
                                           "Validation / {} /  Val: ".format(val_set),
                                           epoch, trainer.writer_val[val_set], loader,
                                           tag=val_set, return_metrics=is_target_report)
                        if val_set == trainer.source_val_set:
                            source_score = result
                        if is_target_report:
                            target_score, target_metrics = result
                        elif val_set == trainer.testset:
                            target_score = result
                    if main_process and SNAPSHOT:
                        if target_score is None:
                            raise RuntimeError("Target validation score is unavailable")
                        checkpoint_path = trainer.checkpoint_best(target_score, epoch)
                        if args.stage == "source" and checkpoint_path is not None:
                            if source_score is None or target_metrics is None:
                                raise RuntimeError("Source or target validation metrics are unavailable")
                            report_path = os.path.join(
                                args.logdir, "best_target_selected_checkpoint_validation.md"
                            )
                            write_source_target_report(
                                report_path, checkpoint_path, epoch, source_score,
                                trainer.source_val_set, trainer.testset, target_metrics
                            )
                            print(">>> Validation report for target-selected source checkpoint: "
                                  "{}".format(report_path))
                if EVALUATE_SOURCE and not cfg.TRAIN.TARGET_ONLY:
                    if not final and epoch > trainer.start_epoch and epoch % log_train == 0:
                        time_call(trainer.validation, "Validation / Train: ", epoch, trainer.writer,
                                  trainer.loader_source, tag="train", max_iter=100)

        for epoch in epochs:
            print("Epoch >>> {:02d} <<< ".format(epoch))
            evaluate_epoch(epoch)
            time_call(trainer.train_epoch, "Train epoch: ", epoch)
        # A checkpoint at the stage boundary contains all requested updates.
        evaluate_epoch(epochs.stop, final=True)
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print("Distributed process group destroyed")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        if log_file is not None:
            sys.stdout, sys.stderr = stdout, stderr
            log_file.close()


def main():
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    args = get_arguments(sys.argv[1:])
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    args.stage = "source" if cfg.MODEL.BASELINE else "adapt"
    if args.stage == "adapt" and args.resume is None:
        raise ValueError("Adaptation requires --resume with a source checkpoint")
    prepare_run_directory(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node < 1:
        raise RuntimeError("CUDA training requires at least one visible GPU")
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.world_size = ngpus_per_node * args.world_size
    args.dist_url = "tcp://127.0.0.1:{}".format(find_free_port())

    print("World size: ", args.world_size, " / URL: {}".format(args.dist_url))
    print("# GPUs per node: ", ngpus_per_node)
    try:
        if ngpus_per_node == 1:
            # Avoid an unnecessary parent/child spawn layer on the common
            # single-GPU Windows setup; this makes Ctrl+C shutdown immediate.
            main_worker(0, ngpus_per_node, args, cfg)
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args, cfg))
    except KeyboardInterrupt:
        print("Training interrupted by user.")


if __name__ == "__main__":
    main()

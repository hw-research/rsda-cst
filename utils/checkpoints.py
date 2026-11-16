"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import re
import torch

from utils.torch_compat import torch_load_compat


class Checkpoint(object):

    def __init__(self, path, max_n=10):
        self.path = path
        self.max_n = max_n
        self.models = {}
        self.checkpoints = []
        self.stage = None
        self.stage_start_epoch = None
        self.loaded_stage = None
        self.loaded_stage_start_epoch = None
        self.loaded_selection_metric = None
        self._target_checkpoints = None

    def create_model(self, model, opt):
        self.models = {}
        self.models['model'] = model
        self.models['opt'] = opt

    def limit(self):
        return self.max_n

    def __len__(self):
        return len(self.checkpoints)

    def _get_full_path(self, suffix):
        filename = self._filename(suffix)
        return os.path.join(self.path, filename)

    def clean(self):
        n_remove = max(0, len(self.checkpoints) - self.max_n)
        for i in range(n_remove):
            self._rm(self.checkpoints[i])
        self.checkpoints = self.checkpoints[n_remove:]

    def _rm(self, suffix):
        path = self._get_full_path(suffix)
        if os.path.isfile(path):
            os.remove(path)

    def _filename(self, suffix):
        return "{}.pth".format(suffix)

    def load(self, path, location):
        if len(path) > 0 and not os.path.isfile(path):
            print("Snapshot {} not found".format(path))

        data = torch.load(path, map_location=location)
        self.models["model"].load_state_dict(data["model"], strict=False)
        self.models["opt"].load_state_dict(data["opt"])
        self.loaded_stage = data.get("stage")
        self.loaded_stage_start_epoch = data.get("stage_start_epoch")
        self.loaded_selection_metric = data.get("selection_metric")

        if not "score" in data:
            data["score"] = -1e16

        return data["epoch"], data["score"]

    def checkpoint(self, score, epoch, suffix=None, selection_metric=None):
        if suffix is None:
            suffix = "epoch{:03d}_score{:4.3f}".format(epoch, score)
        self.checkpoints.append(suffix)

        path = self._get_full_path(suffix)
        if not os.path.isfile(path):
            torch.save({"model": self.models["model"].state_dict(),
                        "opt": self.models["opt"].state_dict(),
                        "score": score,
                        "epoch": epoch,
                        "stage": self.stage,
                        "stage_start_epoch": self.stage_start_epoch,
                        "selection_metric": selection_metric}, path)

        # removing if more than allowed number of snapshots
        self.clean()

    def _get_target_checkpoints(self):
        if self._target_checkpoints is None:
            pattern = re.compile(r"epoch\d+_score-?\d+(?:\.\d+)?\.pth$")
            checkpoints = []
            for filename in os.listdir(self.path):
                if not pattern.fullmatch(filename):
                    continue
                data = torch_load_compat(
                    os.path.join(self.path, filename), map_location="cpu"
                )
                if data.get("stage") != self.stage:
                    continue
                metric = data.get("selection_metric")
                if metric == "target_val_miou" or (self.stage == "adapt" and metric is None):
                    checkpoints.append((float(data["score"]), filename[:-4]))
            self._target_checkpoints = sorted(checkpoints)
        return self._target_checkpoints

    def target_best_score(self):
        checkpoints = self._get_target_checkpoints()
        return checkpoints[-1][0] if checkpoints else -1e16

    def checkpoint_target_best(self, score, epoch):
        """Save a target-validation best and retain up to max_n comparable snapshots."""
        if score <= self.target_best_score():
            return None
        suffix = "epoch{:03d}_score{:.3f}".format(epoch, score)
        if os.path.isfile(self._get_full_path(suffix)):
            suffix = "epoch{:03d}_score{:.9f}".format(epoch, score)
            if os.path.isfile(self._get_full_path(suffix)):
                suffix = "epoch{:03d}_score{:.17f}".format(epoch, score)
        checkpoints = self._get_target_checkpoints()
        self.checkpoints = [name for _, name in checkpoints]
        self.checkpoint(score, epoch, suffix=suffix,
                        selection_metric="target_val_miou")
        checkpoints.append((float(score), suffix))
        checkpoints[:] = checkpoints[-self.max_n:]
        return self._get_full_path(suffix)

"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from __future__ import print_function

import os
import sys
import torch
import argparse
import re
import shutil
from pathlib import Path
from core.config import cfg

def add_global_arguments(parser):

    #
    # Model details
    #
    parser.add_argument("--snapshots-dir", type=str, default='./snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="Where to save log files of the model.")
    parser.add_argument("--exp", type=str, default="main",
                        help="ID of the experiment (multiple runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to snapshots to continue training from")
    parser.add_argument("--run", type=str, help="ID of the run")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow reuse of an existing experiment directory.")
    parser.add_argument("--fresh-run", action="store_true",
                        help="Ask before deleting an existing run's logs and checkpoints.")

    parser.add_argument('--mask-output-dir', type=str, default=None, help='path where to save masks')
    parser.add_argument('--split', type=str, default=None, help='[train|val|test] split to use')
    parser.add_argument('--infer-list', dest='infer_list', type=str, default=None,
                        help='Dataset split/list name used by infer_val.py')
    parser.add_argument('--save-original-size', action='store_true',
                        help='Also save predictions resized back to the original image size.')
    parser.add_argument('--eval-original-size', action='store_true',
                        help='Evaluate predictions at the original ground-truth size when labels are available.')
    parser.add_argument('--save-raw-logits', action='store_true',
                        help='Save per-image raw logits as compressed npz files for post-hoc analysis.')
    parser.add_argument('--save-logit-stats', action='store_true',
                        help='Save per-image confidence, entropy, and margin statistics as CSV.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')

    #
    # Dataset details
    #
    parser.add_argument("--dataloader", type=str,
                        help="Specifies dataloader to use.")
    # parser.add_argument("--infer-list", default="data/val_cityscapes.txt", type=str)

    #
    # Distributed Training
    #
    parser.add_argument('--world-size', default=-1, type=int,
                                help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://1.2.3.4:56789', type=str,
                                help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                             help='distributed backend')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                                help='distributed backend')
    parser.add_argument('--rank', default=0, type=int,
                                help='node rank for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # Seed
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

def check_global_arguments(args):

    args.cuda = torch.cuda.is_available()
    if not args.cuda:
        print("CPU mode is not supported.")
        sys.exit(1)

    args.logdir = os.path.join(args.logdir, args.dataloader, args.exp, args.run)
    args.snapshot_dir = os.path.join(args.snapshots_dir, args.dataloader, args.exp, args.run)
    args.run_exists = os.path.exists(args.logdir) or os.path.exists(args.snapshot_dir)


def _validate_run_paths(args):
    for component in (args.dataloader, args.exp, args.run):
        if not component or component in (".", "..") or Path(component).name != component:
            raise ValueError("Invalid experiment path component: {!r}".format(component))

    for directory in (args.logdir, args.snapshot_dir):
        path = Path(directory)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Run path must be a regular directory: {}".format(path))
        root = path.parents[2].resolve()
        resolved = path.resolve()
        if (resolved != (root / args.dataloader / args.exp / args.run).resolve()
                or resolved == root
                or os.path.commonpath((str(root), str(resolved))) != str(root)):
            raise ValueError("Run path escapes its configured root: {}".format(path))
    if Path(args.logdir).resolve() == Path(args.snapshot_dir).resolve():
        raise ValueError("Log and checkpoint run directories must be different")


def _existing_run_stage(args):
    stages = set()
    for directory in (args.logdir, args.snapshot_dir):
        marker = Path(directory) / ".run_stage"
        if marker.is_file():
            stages.add(marker.read_text(encoding="utf-8").strip())

    if not stages:
        log_file = Path(args.logdir) / "train.log"
        if log_file.is_file():
            with log_file.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    match = re.match(r"^Stage (source|adapt):", line)
                    if match:
                        stages.add(match.group(1))

    if len(stages) > 1 or (stages and args.stage not in stages):
        raise RuntimeError(
            "Existing run belongs to another or mixed training stage; "
            "use a different run name to protect the source checkpoint."
        )
    if not stages and any(
        Path(directory).exists() and any(Path(directory).iterdir())
        for directory in (args.logdir, args.snapshot_dir)
    ):
        raise RuntimeError(
            "Cannot determine the existing run's training stage; "
            "use a different run name rather than risk deleting source weights."
        )


def _create_run_directories(args):
    for directory in (args.logdir, args.snapshot_dir):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / ".run_stage").write_text(args.stage + "\n", encoding="utf-8")


def prepare_run_directory(args):
    """Delete an existing run only after stage checks and confirmation."""
    if args.fresh_run and args.overwrite:
        raise ValueError("--fresh-run and --overwrite cannot be used together")

    _validate_run_paths(args)

    if args.run_exists:
        _existing_run_stage(args)

    if args.fresh_run and args.run_exists:
        if args.resume is not None:
            resume_path = Path(args.resume).resolve()
            for directory in (args.logdir, args.snapshot_dir):
                run_dir = Path(directory).resolve()
                try:
                    same_run = os.path.commonpath((str(run_dir), str(resume_path))) == str(run_dir)
                except ValueError:
                    same_run = False
                if same_run:
                    raise ValueError(
                        "Cannot delete the checkpoint used to start this run; "
                        "use --overwrite for a continuation."
                    )

        if not sys.stdin.isatty():
            raise RuntimeError("Deleting an existing run requires an interactive terminal")
        print("\nExisting {} run will be permanently deleted:".format(args.stage))
        print("  logs:      {}".format(Path(args.logdir).resolve()))
        print("  snapshots: {}".format(Path(args.snapshot_dir).resolve()))
        try:
            answer = input("Delete these files and retrain? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            raise RuntimeError("Experiment cancelled; existing files were preserved.")
        for directory in (args.logdir, args.snapshot_dir):
            if Path(directory).is_dir():
                shutil.rmtree(directory)
        _create_run_directories(args)
        return

    if not args.run_exists:
        _create_run_directories(args)
        return

    if args.overwrite:
        print("Existing experiment detected; reusing its log and checkpoint directories.")
        _create_run_directories(args)
        return

    if not sys.stdin.isatty():
        raise RuntimeError(
            "An existing run directory was found. Re-run with --overwrite "
            "to reuse it, or choose a new --run name."
        )

    print("\nExisting experiment found:")
    print("  logs:      {}".format(args.logdir))
    print("  snapshots: {}".format(args.snapshot_dir))
    answer = input("Reuse this experiment directory? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("Experiment cancelled; existing files were preserved.")
    print("Existing experiment accepted; old files will not be deleted.")
    _create_run_directories(args)

def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args

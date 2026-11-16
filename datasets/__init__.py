from torch.utils import data
from tools.category import labels as SegLabels
from core.config import cfg
from .dataloader_seg import DLSeg
from .dataloader_target import DataTarget
from .dataloader_target_abn import DataTargetABN
# ---------- 尝试导入遥感数据集加载器 ----------
REMOTE_SENSING_AVAILABLE = False
try:
    from .dataloader_remote_sensing import DLSegRemoteSensing, DataTargetRemoteSensing
    from tools.remote_sensing_categories import (
        get_remote_sensing_labels,
        get_remote_sensing_class_names,
        get_remote_sensing_num_classes
    )
    REMOTE_SENSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Remote sensing dataset loader not available. Error: {e}")
    print("Only GTA5/SYNTHIA/Cityscapes datasets are supported.")
    # 定义占位函数，避免后续调用出错
    def get_remote_sensing_class_names(*args, **kwargs):
        raise NotImplementedError("Remote sensing module not available")
    def get_remote_sensing_num_classes(*args, **kwargs):
        raise NotImplementedError("Remote sensing module not available")


def get_class_names(args):
    task = cfg.TRAIN.TASK
    if REMOTE_SENSING_AVAILABLE and is_remote_sensing_dataset(task):
        dataset_name = task.split('_')[1]
        if dataset_name == 'loveda':
            dataset_name = task.split('_')[1] + '_' + task.split('_')[2] if len(task.split('_')) > 2 else task.split('_')[1]
            print("dataset_name is:",dataset_name)
        return get_remote_sensing_class_names(dataset_name)
    return [l.name for l in SegLabels if l.name != "unlabeled"]


def get_num_classes(args):
    task = cfg.TRAIN.TASK
    if REMOTE_SENSING_AVAILABLE and is_remote_sensing_dataset(task):
        dataset_name = task.split('_')[1]
        print("dataset_name is:",dataset_name)
        if dataset_name == 'loveda':
            dataset_name = task.split('_')[1] + '_' + task.split('_')[2] if len(task.split('_')) > 2 else task.split('_')[1]
        return get_remote_sensing_num_classes(dataset_name)
    class_names = get_class_names(args)
    return len(class_names)


def get_val_sets(train_split, target_split=None):
    # GTA5 / SYNTHIA
    if train_split == "train_game_9K":
        return ("val_game_1K", "train_cityscapes", "val_cityscapes", "val2_cityscapes"), "val2_cityscapes"
    if train_split == "train_synthia_9K":
        return ("val_synthia_1K", "train_cityscapes", "val_cityscapes", "val2_cityscapes"), "val2_cityscapes"
    # 遥感数据集：源域、目标域分别映射到各自验证集。
    # 这种写法按配置中的实际源域/目标域选择验证集。
    remote_val_map = {
        "train_vaihingen": "val_vaihingen",
        "train_potsdam_irrg": "val_potsdam_irrg",
        "train_potsdam_rgb": "val_potsdam_rgb",
        # The unqualified Potsdam split uses IRRG.
        "train_potsdam": "val_potsdam_irrg",
        "train_loveda_urban": "val_loveda_urban",
        "train_loveda_rural": "val_loveda_rural",
    }
    if train_split in remote_val_map and target_split in remote_val_map:
        source_val = remote_val_map[train_split]
        target_val = remote_val_map[target_split]
        return (source_val, target_val), target_val

    raise ValueError(
        "No validation-set mapping for source={!r}, target={!r}".format(
            train_split, target_split
        )
    )

def is_remote_sensing_dataset(task):
    if not REMOTE_SENSING_AVAILABLE:
        return False
    return task.startswith((
        'train_potsdam', 'val_potsdam',
        'train_vaihingen', 'val_vaihingen',
        'train_loveda_urban', 'val_loveda_urban',
        'train_loveda_rural', 'val_loveda_rural',
    ))


def get_dataloader(args, cfg, split, num_classes, ngpus_per_node):
    assert split in ("train", "val")

    task = cfg.TRAIN.TASK
    target = cfg.TRAIN.TARGET

    # 支持的数据集列表
    supported_tasks = ('train_game_9K', 'train_synthia_9K')
    if REMOTE_SENSING_AVAILABLE:
        supported_tasks += (
            'train_potsdam', 'train_potsdam_irrg', 'train_potsdam_rgb',
            'train_vaihingen', 'train_loveda_urban', 'train_loveda_rural'
        )
    assert task in supported_tasks, f"Unknown task: {task}. Supported tasks: {supported_tasks}"

    remote_sensing = is_remote_sensing_dataset(task)

    batch_size = int(cfg.TRAIN.BATCH_SIZE / ngpus_per_node)
    workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    print(f"Dataloader: # workers {workers} / split {split}")
    kwargs = {'num_workers': workers, 'pin_memory': True} if args.cuda else {}

    def _dataloader(dataset, batch_size, sampler=None, drop_last=False):
        sampler = data.distributed.DistributedSampler(dataset)
        return data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            sampler=sampler, drop_last=drop_last, **kwargs), sampler

    if split == "train":
        # 源域
        if remote_sensing:
            dataset = DLSegRemoteSensing(cfg, task, renorm=cfg.DATASET.RENORM_SOURCE)
        else:
            dataset = DLSeg(cfg, task, renorm=cfg.DATASET.RENORM_SOURCE)

        # 目标域
        if cfg.MODEL.BASELINE:
            # ABN 阶段：使用简单的单视图加载器，返回 (image, mask) 且 mask=255
            from .dataloader_target_abn import DataTargetABN
            dataset_target = DataTargetABN(cfg, target)
        else:
            # 自训练阶段：使用多视图加载器（返回 5 元组）
            if remote_sensing:
                dataset_target = DataTargetRemoteSensing(cfg, target, num_classes,
                                                         weights=cfg.DATASET.SAMPLE_WEIGHTS)
            else:
                dataset_target = DataTarget(cfg, target, num_classes,
                                            weights=cfg.DATASET.SAMPLE_WEIGHTS)

        dataset_target.set_num_samples(len(dataset))
        # ABN uses one ordinary target image batch per source batch so target
        # images participate in BN running-stat updates throughout the epoch.
        # Self-training keeps its grouped multi-view target batch size.
        batch_target = (
            batch_size if cfg.MODEL.BASELINE
            else max(1, cfg.TRAIN.NUM_GROUPS // ngpus_per_node)
        )
        print(f"Batch size SOURCE: {batch_size}")
        print(f"Batch size TARGET: {batch_target} / {batch_target * cfg.TRAIN.GROUP_SIZE}")

        return (_dataloader(dataset, batch_size, drop_last=True),
                _dataloader(dataset_target, batch_target, drop_last=True))

    else:  # validation
        val_sets, _ = get_val_sets(task, target)
        loaders = {}
        for val_set in val_sets:
            if remote_sensing:
                dataset = DLSegRemoteSensing(cfg, val_set)
            else:
                dataset = DLSeg(cfg, val_set)
            loader, sampler = _dataloader(dataset, batch_size)
            loaders[val_set] = loader
        return loaders

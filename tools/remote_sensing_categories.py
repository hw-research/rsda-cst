"""
遥感数据集类别定义
支持 Potsdam、Vaihingen、LoveDA (Urban/Rural)
"""

from collections import namedtuple

# 定义标签结构
Label = namedtuple('Label', [
    'name',        # 类别名称
    'id',          # 原始数据集中的ID
    'train_id',    # 训练时的ID（连续整数）
    'category',    # 类别
    'category_id', # 类别ID
    'has_instances', # 是否有实例
    'ignore_in_eval', # 是否在评估中忽略
    'color',       # 可视化颜色
])

# ========== Potsdam / Vaihingen 数据集（ISPRS 格式，6类）==========
POTSDAM_LABELS = [
    Label('impervious_surface', 1, 0, 'surface', 0, False, False, (255, 255, 255)),  # 修正颜色
    Label('building',          2, 1, 'construction', 1, False, False, (0, 0, 255)),
    Label('low_vegetation',    3, 2, 'nature', 2, False, False, (0, 255, 255)),
    Label('tree',              4, 3, 'nature', 3, False, False, (0, 255, 0)),
    Label('car',               5, 4, 'vehicle', 4, True, False, (255, 255, 0)),
    Label('clutter',           6, 5, 'void', 5, False, False, (255, 0, 0)),
]
# ========== LoveDA 数据集（7类，Urban 和 Rural 共享相同标签）==========
LOVEDA_LABELS = [
    Label('background',   1, 0, 'void', 0, False, False, (255, 255, 255)),
    Label('building',     2, 1, 'construction', 1, False, False, (255, 0, 0)),
    Label('road',         3, 2, 'flat', 2, False, False, (255, 255, 0)),
    Label('water',        4, 3, 'nature', 3, False, False, (0, 0, 255)),
    Label('barren',       5, 4, 'nature', 4, False, False, (159, 129, 183)),
    Label('forest',       6, 5, 'nature', 5, False, False, (0, 255, 0)),
    Label('agricultural', 7, 6, 'nature', 6, False, False, (255, 195, 128)),
]


def get_remote_sensing_labels(dataset_name):
    """根据数据集名称获取标签列表"""
    if dataset_name in ['potsdam', 'vaihingen']:
        return POTSDAM_LABELS
    elif dataset_name in ['loveda', 'loveda_urban', 'loveda_rural']:
        return LOVEDA_LABELS
    else:
        raise ValueError(f"Unknown remote sensing dataset: {dataset_name}")


def get_remote_sensing_class_names(dataset_name):
    """获取类别名称列表（排除忽略类）"""
    labels = get_remote_sensing_labels(dataset_name)
    return [l.name for l in labels if not l.ignore_in_eval]


def get_remote_sensing_num_classes(dataset_name):
    """获取类别数量（不包括忽略类）"""
    return len(get_remote_sensing_class_names(dataset_name))


def get_remote_sensing_palette(dataset_name):
    """获取遥感数据集的调色板（用于可视化）"""
    labels = get_remote_sensing_labels(dataset_name)
    palette = []
    for label in labels:
        if not label.ignore_in_eval:
            palette.extend(label.color)
        else:
            palette.extend([0, 0, 0])
    return palette

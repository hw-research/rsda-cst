"""
遥感数据集预处理脚本
为每个数据集独立生成训练集、验证集、测试集文件列表
自动检测文件格式并修复路径
"""

import os
import argparse
import random
from pathlib import Path

def find_image_label_pairs(data_dir):
    """
    查找图像和对应的标签文件
    """
    img_dir = os.path.join(data_dir, 'images_IRRG')
    lbl_dir = os.path.join(data_dir, 'labels')

    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        return []

    pairs = []
    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue

        base_name = os.path.splitext(img_name)[0]

        # 移除可能的后缀如 _RGB, _IRRG 等
        clean_base = base_name
        for suffix in ['_RGB', '_IRRG', '_IRG', '_RGBIR']:
            if clean_base.endswith(suffix):
                clean_base = clean_base[:-len(suffix)]
                break

        # 尝试多种标签文件名格式
        possible_lbl_names = [
            f"{base_name}_label.png",
            f"{base_name}_label.tif",
            f"{base_name}_label.jpg",
            f"{clean_base}_label.png",
            f"{clean_base}_label.tif",
            f"{clean_base}.png",
            f"{clean_base}.tif",
            f"{base_name}.png",
            f"{base_name}.tif",
        ]

        lbl_name = None
        for lbl_candidate in possible_lbl_names:
            if os.path.exists(os.path.join(lbl_dir, lbl_candidate)):
                lbl_name = lbl_candidate
                break

        if lbl_name:
            # 使用相对路径（相对于 data 目录）
            dataset_name = os.path.basename(data_dir)
            rel_img_path = os.path.join(dataset_name, 'images_IRRG', img_name).replace('\\', '/')
            rel_lbl_path = os.path.join(dataset_name, 'labels', lbl_name).replace('\\', '/')
            pairs.append((rel_img_path, rel_lbl_path))
        else:
            print(f"Warning: No label found for {img_name}")

    return pairs


def split_dataset(pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    划分数据集为训练集、验证集、测试集
    """
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_idx = int(total * train_ratio)
    val_idx = int(total * (train_ratio + val_ratio))

    train_pairs = pairs[:train_idx]
    val_pairs = pairs[train_idx:val_idx]
    test_pairs = pairs[val_idx:]

    return train_pairs, val_pairs, test_pairs


def generate_filelist_for_dataset(dataset_name, data_dir, output_dir, train_ratio, val_ratio, test_ratio):
    """
    为单个数据集生成训练集、验证集、测试集文件
    """
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"Data directory: {data_dir}")
    print(f"{'='*50}")

    # 查找图像-标签对
    pairs = find_image_label_pairs(data_dir)
    if not pairs:
        print(f"Error: No valid image-label pairs found in {data_dir}")
        return False

    print(f"Found {len(pairs)} valid image-label pairs")

    # 划分数据集
    train_pairs, val_pairs, test_pairs = split_dataset(pairs, train_ratio, val_ratio, test_ratio)

    print(f"Training set: {len(train_pairs)} images_IRRG ({train_ratio*100:.0f}%)")
    print(f"Validation set: {len(val_pairs)} images_IRRG ({val_ratio*100:.0f}%)")
    print(f"Test set: {len(test_pairs)} images_IRRG ({test_ratio*100:.0f}%)")

    # 生成文件
    train_file = os.path.join(output_dir, f'train_{dataset_name}.txt')
    val_file = os.path.join(output_dir, f'val_{dataset_name}.txt')
    test_file = os.path.join(output_dir, f'test_{dataset_name}.txt')

    with open(train_file, 'w') as f:
        for img_path, lbl_path in train_pairs:
            f.write(f"{img_path} {lbl_path}\n")
    print(f"Created: {train_file}")

    with open(val_file, 'w') as f:
        for img_path, lbl_path in val_pairs:
            f.write(f"{img_path} {lbl_path}\n")
    print(f"Created: {val_file}")

    with open(test_file, 'w') as f:
        for img_path, lbl_path in test_pairs:
            f.write(f"{img_path} {lbl_path}\n")
    print(f"Created: {test_file}")

    return True


def generate_filelist_for_loveda(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    为 LoveDA 数据集生成文件列表
    """
    print(f"\n{'='*50}")
    print("Processing LoveDA dataset")
    print(f"{'='*50}")

    for subset in ['urban', 'rural']:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"Warning: {subset_dir} not found, skipping")
            continue

        # 检查是否已经有预定义的 train/val/test 划分
        has_predefined = all(os.path.exists(os.path.join(subset_dir, s)) for s in ['train', 'val', 'test'])

        if has_predefined:
            print(f"\nProcessing {subset} subset (using predefined splits)")
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(subset_dir, split)
                img_dir = os.path.join(split_dir, 'images_IRRG')
                lbl_dir = os.path.join(split_dir, 'labels')

                if not os.path.exists(img_dir):
                    continue

                pairs = []
                for img_name in sorted(os.listdir(img_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    base_name = os.path.splitext(img_name)[0]
                    lbl_name = f"{base_name}_label.png"

                    if os.path.exists(os.path.join(lbl_dir, lbl_name)):
                        rel_img_path = os.path.join(subset, split, 'images_IRRG', img_name).replace('\\', '/')
                        rel_lbl_path = os.path.join(subset, split, 'labels', lbl_name).replace('\\', '/')
                        pairs.append((rel_img_path, rel_lbl_path))

                if pairs:
                    output_file = os.path.join(output_dir, f'{split}_loveda_{subset}.txt')
                    with open(output_file, 'w') as f:
                        for img_path, lbl_path in pairs:
                            f.write(f"{img_path} {lbl_path}\n")
                    print(f"Created: {output_file} ({len(pairs)} entries)")
        else:
            print(f"\nProcessing {subset} subset (generating splits)")
            subset_data_dir = os.path.join(data_dir, subset)
            generate_filelist_for_dataset(f'loveda_{subset}', subset_data_dir, output_dir,
                                         train_ratio, val_ratio, test_ratio)


def fix_existing_filelist(filepath):
    """
    修复已存在的文件列表中的路径
    """
    if not os.path.exists(filepath):
        return False

    print(f"Fixing {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    fixed_count = 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        # 将反斜杠替换为正斜杠
        img_path = parts[0].replace('\\', '/')

        if len(parts) > 1:
            lbl_path = parts[1].replace('\\', '/')
            new_line = f"{img_path} {lbl_path}\n"
        else:
            new_line = f"{img_path}\n"

        new_lines.append(new_line)
        fixed_count += 1

    # 备份原文件
    backup = filepath + '.bak'
    if not os.path.exists(backup):
        os.rename(filepath, backup)
        print(f"  Backup saved to {backup}")

    # 写入新文件
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"  Fixed {fixed_count} entries")

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test file lists for remote sensing datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['potsdam', 'vaihingen', 'loveda', 'all'],
                        help='Dataset name (or "all" for all datasets)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory (contains images_IRRG/labels subdirectories)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for txt files (default: data)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--fix', action='store_true',
                        help='Fix existing file lists (replace backslashes with forward slashes)')

    args = parser.parse_args()

    # 验证比例
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-5:
        print(f"Error: train_ratio + val_ratio + test_ratio must equal 1.0 (current: {total})")
        return

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 如果只是修复现有文件
    if args.fix:
        print("Fixing existing file lists...")
        for ds in ['potsdam', 'vaihingen']:
            for split in ['train', 'val', 'test']:
                filepath = os.path.join(args.output_dir, f'{split}_{ds}.txt')
                fix_existing_filelist(filepath)
        print("Done!")
        return

    # 生成新的文件列表
    if args.dataset == 'all':
        datasets = ['potsdam', 'vaihingen', 'loveda']
        for ds in datasets:
            ds_dir = os.path.join(args.data_dir, ds) if ds != 'loveda' else args.data_dir
            if os.path.exists(ds_dir):
                if ds == 'loveda':
                    generate_filelist_for_loveda(ds_dir, args.output_dir,
                                                args.train_ratio, args.val_ratio, args.test_ratio)
                else:
                    generate_filelist_for_dataset(ds, ds_dir, args.output_dir,
                                                 args.train_ratio, args.val_ratio, args.test_ratio)
            else:
                print(f"Warning: {ds_dir} not found, skipping")

    elif args.dataset == 'loveda':
        generate_filelist_for_loveda(args.data_dir, args.output_dir,
                                     args.train_ratio, args.val_ratio, args.test_ratio)

    else:
        generate_filelist_for_dataset(args.dataset, args.data_dir, args.output_dir,
                                     args.train_ratio, args.val_ratio, args.test_ratio)

    print("\n" + "="*50)
    print("All file lists generated successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
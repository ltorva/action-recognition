import os
from dataset import UCF101Dataset
from torch.utils.data import DataLoader
import torch

def verify_labels():
    # 加载数据集
    dataset = UCF101Dataset(
        root_dir="data/videos",
        split_file="data/train.txt",
        transform=None,
        num_frames=16
    )
    
    # 检查类别分布
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # 打印结果
    print("\n=== 类别分布 ===")
    for label, count in class_counts.items():
        class_name = dataset.classes[label]
        print(f"类别 {label} ({class_name}): {count} 个样本")
    
    # 检查标签范围
    print(f"\n标签范围: {min(class_counts.keys())} - {max(class_counts.keys())}")
    print(f"总类别数: {len(class_counts)}")
    
if __name__ == "__main__":
    verify_labels()
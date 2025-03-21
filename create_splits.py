import os
from pathlib import Path
import random
import shutil

def create_data_splits():
    """创建训练集和测试集划分文件"""
    data_root = Path('data')
    videos_dir = data_root / 'videos'
    train_ratio = 0.8
    
    # 读取类别映射
    classes = {}
    with open(data_root / 'classInd.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                idx, name = line.strip().split(' ', 1)[0:2]
                classes[name.split('#')[0].strip()] = int(idx)
    
    # 创建训练和测试集列表
    train_samples = []
    test_samples = []
    
    # 遍历每个类别目录
    for class_name in classes.keys():
        class_dir = videos_dir / class_name
        if not class_dir.exists():
            print(f"警告: {class_name} 目录不存在")
            continue
            
        # 获取该类别下的所有视频文件
        videos = list(class_dir.glob('*.mp4'))
        if not videos:
            print(f"警告: {class_name} 目录下没有视频文件")
            continue
            
        # 随机打乱视频列表
        random.shuffle(videos)
        
        # 划分训练集和测试集
        split_idx = int(len(videos) * train_ratio)
        train_vids = videos[:split_idx]
        test_vids = videos[split_idx:]
        
        # 添加到对应列表
        for video in train_vids:
            train_samples.append(f"{class_name}/{video.name} {classes[class_name]}")
        for video in test_vids:
            test_samples.append(f"{class_name}/{video.name} {classes[class_name]}")
    
    # 保存训练集文件
    with open(data_root / 'train.txt', 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(f"{sample}\n")
    
    # 保存测试集文件
    with open(data_root / 'test.txt', 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(f"{sample}\n")
    
    print(f"总样本数: {len(train_samples) + len(test_samples)}")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")

if __name__ == '__main__':
    random.seed(42)
    create_data_splits()
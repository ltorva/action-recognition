import torch
from dataset import UCF101Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_dataset_output():
    # 创建数据集实例
    dataset = UCF101Dataset(
        root_dir="data/videos",
        split_file="data/train.txt",
        num_frames=16
    )
    
    # 获取第一个样本
    frames, label = dataset[0]
    print(f"帧张量形状: {frames.shape}")
    print(f"视频标签: {label}")
    
    # 使用固定的正方形显示比例
    base_size = 2.0  # 基础单位大小
    fig = plt.figure(figsize=(base_size * 8, base_size * 2))
    
    # 创建网格布局
    rows, cols = 2, 8
    
    for i in range(min(16, frames.shape[1])):
        # 获取当前帧 [C, T, H, W] -> [C, H, W]
        frame = frames[:, i, :, :]
        
        # 转换维度顺序并反归一化
        frame = frame.numpy().transpose(1, 2, 0)  # [H, W, C]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        # 添加子图
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(frame)
        plt.title(f'Frame {i}')
        plt.axis('off')
        
        # 设置正方形显示比例
        ax.set_aspect('equal')  # 使用相等的宽高比
    
    plt.suptitle('Dataset Output Frames', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_frames.png', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_multiple_samples(num_samples=5):
    # 创建数据集实例
    dataset = UCF101Dataset(
        root_dir="data/videos",
        split_file="data/train.txt",
        num_frames=16
    )
    
    # 对每个样本生成可视化
    for sample_idx in range(num_samples):
        # 获取随机样本
        frames, label = dataset[torch.randint(len(dataset), (1,)).item()]
        print(f"\n样本 {sample_idx + 1}:")
        print(f"帧张量形状: {frames.shape}")
        print(f"动作类别: {label}")
        
        # 创建图形
        base_size = 2.0
        fig = plt.figure(figsize=(base_size * 8, base_size * 2))
        
        # 创建网格布局
        rows, cols = 2, 8
        
        for i in range(min(16, frames.shape[1])):
            # 获取当前帧
            frame = frames[:, i, :, :]
            
            # 转换维度顺序并反归一化
            frame = frame.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            frame = frame * std + mean
            frame = np.clip(frame, 0, 1)
            
            # 添加子图
            ax = plt.subplot(rows, cols, i + 1)
            plt.imshow(frame)
            plt.title(f'Frame {i}')
            plt.axis('off')
            ax.set_aspect('equal')
        
        plt.suptitle(f'Sample {sample_idx + 1} - Action: {label}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'dataset_frames_sample_{sample_idx + 1}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    visualize_dataset_output()
    print("可视化结果已保存到 'dataset_frames.png'")
    visualize_multiple_samples()
    print("多个样本的可视化结果已保存")
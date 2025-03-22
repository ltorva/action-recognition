import torch
import matplotlib.pyplot as plt
from dataset import UCF101Dataset
from model import HARViT
from torchvision import transforms
import numpy as np
import os

def verify_preprocessing_and_dimensions():
    # 1. 数据预处理验证
    print("\n=== 数据预处理验证 ===")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),              # 首先调整大小到略大尺寸
        transforms.CenterCrop(224),          # 使用中心裁剪替代随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset = UCF101Dataset(
        root_dir="data/videos",
        split_file="data/train.txt",
        transform=transform,
        num_frames=16
    )
    
    # 获取一个样本
    frames, label = dataset[0]
    
    print(f"\n数据形状:")
    print(f"frames shape: {frames.shape}")
    print(f"label: {label}")
    
    # 修改可视化部分
    plt.figure(figsize=(15, 5))
    for i in range(16):  # 显示前4帧
        plt.subplot(2, 8, i+1)
        # 正确处理维度顺序
        frame = frames[:, i, :, :].numpy()  # 获取第i帧，维度为[C, H, W]
        frame = frame.transpose(1, 2, 0)    # 转换为[H, W, C]
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        plt.imshow(frame)
        plt.title(f'Frame {i}')
        plt.axis('off')  # 隐藏坐标轴
    
    plt.tight_layout()
    plt.savefig('preprocessed_frames.png')
    print(f"\n预处理后的帧已保存到 'preprocessed_frames.png'")
    
    # 添加详细的维度检查
    print("\n=== 数据维度变化追踪 ===")
    
    # 1. 修复视频路径，使用完整路径
    video_path = os.path.join(
        "D:/Action/action-recognition/data/videos",
        dataset.samples[0][0]
    )
    print(f"尝试加载视频: {video_path}")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        print("请检查以下内容:")
        print("1. 视频文件夹结构是否正确")
        print("2. 文件名是否匹配")
        print("3. 路径分隔符是否正确")
        return
    
    # 加载视频帧
    video_frames = dataset._load_frames_efficiently(video_path)
    if video_frames is None:
        print(f"错误: 无法加载视频: {video_path}")
        return
        
    print(f"原始视频帧形状: {video_frames.shape}")  # [T, H, W, C]
    
    # 2. 采样后的帧
    sampled_frames = video_frames[::len(video_frames)//16]
    print(f"采样后形状: {sampled_frames.shape}")    # [16, H, W, C]
    
    # 3. 预处理后的帧
    processed_frames = frames
    print(f"预处理后形状: {processed_frames.shape}") # [C, T, H, W]
    
    # 添加每一步变换的可视化
    plt.figure(figsize=(15, 10))
    
    # 原始帧 - 需要调整维度顺序
    plt.subplot(3, 1, 1)
    frame = video_frames[0].permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    plt.imshow(frame)
    plt.title('原始帧')
    
    # 采样帧
    plt.subplot(3, 1, 2)
    frame = sampled_frames[0].permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    plt.imshow(frame)
    plt.title('采样后帧')
    
    # 预处理后帧（需要反归一化）
    plt.subplot(3, 1, 3)
    frame = processed_frames[:, 0].permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    frame = frame * std + mean
    frame = np.clip(frame, 0, 1)
    plt.imshow(frame)
    plt.title('预处理后帧')
    
    plt.tight_layout()
    plt.savefig('preprocessing_steps.png')
    print("\n预处理步骤可视化已保存到 'preprocessing_steps.png'")
    
    # 2. 模型维度验证
    print("\n=== 模型维度验证 ===")
    
    # 创建模型
    model = HARViT(
        img_size=224,
        patch_size=32,
        in_channels=3,
        num_classes=12,
        embed_dim=128,
        depth=2,
        num_heads=4,
        num_frames=16
    )
    model.eval()
    
    # 准备批次数据
    batch_size = 2
    dummy_input = frames.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    print(f"\n输入维度:")
    print(f"Batch shape: {dummy_input.shape}")  # 应该是 [B, C, T, H, W]
    
    # 跟踪中间层输出
    with torch.no_grad():
        try:
            # Backbone 特征
            backbone_features = model.backbone(dummy_input)
            print(f"\nBackbone 输出: {backbone_features.shape}")
            
            # Patch 嵌入
            patch_embed = model.patch_embed(dummy_input)
            print(f"Patch 嵌入输出: {patch_embed.shape}")
            
            # 完整前向传播
            output = model(dummy_input)
            print(f"\n最终输出: {output.shape}")
            print(f"期望输出: [batch_size, num_classes] = [{batch_size}, 12]")
            
            print("\n✓ 模型维度检查通过")
            
        except Exception as e:
            print(f"\n✗ 模型维度检查失败:")
            print(f"错误: {str(e)}")

if __name__ == "__main__":
    verify_preprocessing_and_dimensions()
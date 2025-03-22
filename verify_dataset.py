import os
from dataset import UCF101Dataset
from torch.utils.data import DataLoader
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

def quick_video_check(video_path):
    """快速检查视频文件"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "无法打开视频文件"
        
        # 只读取基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return True, {
            'frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}"
        }
    except Exception as e:
        return False, str(e)

def verify_labels():
    # 设置正确的文件路径
    root_dir = "D:/建筑工人动作识别/action-recognition/data/videos"
    split_file = "D:/建筑工人动作识别/action-recognition/data/train.txt"
    
    print("\n=== 检查文件路径 ===")
    print(f"数据根目录: {root_dir}")
    print(f"训练集文件: {split_file}")
    
    # 加载数据集元数据
    try:
        dataset = UCF101Dataset(
            root_dir=root_dir,
            split_file=split_file,
            transform=None,
            num_frames=16
        )
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return

    # 首先验证类别映射
    print("\n=== 类别映射 ===")
    if hasattr(dataset, 'classes'):
        print(f"找到 {len(dataset.classes)} 个类别:")
        for class_name, idx in dataset.classes.items():
            print(f"类别名: {class_name}, 索引: {idx}")
    else:
        print("警告: 数据集未定义类别映射!")
    
    # 快速统计类别分布
    class_counts = {}
    video_paths = []
    
    print("\n=== 检查数据集结构 ===")
    try:
        for video_path, label in dataset.samples:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
            full_path = os.path.join(root_dir, video_path)
            video_paths.append(full_path)
            
            # 打印前几个样本以验证
            if len(video_paths) <= 5:
                print(f"示例: {video_path} -> 标签 {label}")
    
        # 打印类别统计
        print("\n=== 类别分布 ===")
        for label, count in sorted(class_counts.items()):
            try:
                # 反向查找类别名
                class_name = next(name for name, idx in dataset.classes.items() if idx == label)
                print(f"类别 {label} ({class_name}): {count} 个样本")
            except StopIteration:
                print(f"类别 {label} (未知类别名): {count} 个样本")
    
        # 使用多线程验证视频文件
        print("\n=== 验证视频文件 ===")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(quick_video_check, path) for path in video_paths]
            
            valid_videos = 0
            invalid_videos = []
            
            for path, future in tqdm(zip(video_paths, futures), total=len(video_paths)):
                success, info = future.result()
                if success:
                    valid_videos += 1
                    # 随机打印一些有效视频的信息
                    if valid_videos % 10 == 0:
                        print(f"\n视频信息 - {os.path.basename(path)}:")
                        print(f"  帧数: {info['frames']}")
                        print(f"  FPS: {info['fps']}")
                        print(f"  分辨率: {info['resolution']}")
                else:
                    invalid_videos.append((path, info))
        
        # 打印验证结果
        print(f"\n=== 验证结果 ===")
        print(f"总视频数: {len(video_paths)}")
        print(f"有效视频: {valid_videos}")
        print(f"无效视频: {len(invalid_videos)}")
        
        if invalid_videos:
            print("\n问题视频文件:")
            for path, error in invalid_videos:
                print(f"- {path}: {error}")
                
    except Exception as e:
        print(f"\n验证过程出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    verify_labels()
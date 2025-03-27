import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image

class RandomTemporalCrop:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frames):
        # frames shape: (T, C, H, W)
        t = frames.shape[0]
        if t <= self.size:
            return frames
        start = random.randint(0, t - self.size)
        return frames[start:start+self.size]

class TemporalMask:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size
    
    def __call__(self, frames):
        # frames shape: (T, C, H, W)
        t = frames.shape[0]
        if t <= self.mask_size:
            return frames
        start = random.randint(0, t - self.mask_size)
        frames[start:start+self.mask_size] = 0
        return frames

class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std
        
    def __call__(self, frames):
        return frames + torch.randn_like(frames) * self.std

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, num_frames=8, temporal_transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),  # 先将短边缩放到256
                transforms.CenterCrop(224),  # 中心裁剪得到224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Default temporal transforms if none provided
        if temporal_transform is None:
            self.temporal_transform = transforms.Compose([
                RandomTemporalCrop(size=num_frames),
                TemporalMask(mask_size=2),
                GaussianNoise(std=0.1)
            ])
        else:
            self.temporal_transform = temporal_transform
        
        # Load class mapping
        self.classes = {}
        classInd_path = os.path.join(os.path.dirname(split_file), "classInd.txt")
        with open(classInd_path, "r") as f:
            for line in f:
                idx, name = line.strip().split(" ")
                self.classes[name] = int(idx) - 1
        
        # Load samples
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                video_path = parts[0]
                if len(parts) > 1:
                    label = int(parts[1]) - 1
                else:
                    class_name = video_path.split('/')[0]
                    label = self.classes[class_name]
                self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, video_path)
        
        # 读取视频帧
        frames = self._load_frames_efficiently(full_path)
        
        if frames is None or frames.shape[1] < self.num_frames:
            # 返回空白视频
            frames = torch.zeros((3, self.num_frames, 224, 224))
            return frames, label
        
        # frames 此时的形状是 [C, T, H, W]
        # 需要转换为 [T, C, H, W] 用于时序变换
        frames = frames.permute(1, 0, 2, 3)
        
        # 应用时序变换
        if self.temporal_transform:
            frames = self.temporal_transform(frames)
        
        # 转换回 [C, T, H, W] 用于返回
        frames = frames.permute(1, 0, 2, 3)
        
        return frames, label
    
    def _detect_person(self, frame):
        """使用HOG检测器检测人物"""
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 检测人物
        boxes, _ = hog.detectMultiScale(frame, 
                                      winStride=(8, 8),
                                      padding=(16, 16),
                                      scale=1.05)
        
        if len(boxes) > 0:
            # 选择最大的检测框
            box = max(boxes, key=lambda x: x[2] * x[3])
            x, y, w, h = box
            
            # 扩大检测框以包含更多上下文
            margin = int(max(w, h) * 0.3)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)
            
            return (x, y, w, h)
        return None

    def _load_frames_efficiently(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_path}")
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            bbox = None
            last_valid_bbox = None
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 更新检测框
                if bbox is None or frame_idx % 5 == 0:
                    bbox = self._detect_person(frame)
                    if bbox is not None:
                        last_valid_bbox = bbox
                    else:
                        bbox = last_valid_bbox  # 使用最后一个有效的检测框
                
                if bbox is not None:
                    x, y, w, h = bbox
                    # 计算正方形裁剪区域
                    size = max(w, h)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 确保正方形区域在图像范围内
                    x1 = max(0, center_x - size // 2)
                    y1 = max(0, center_y - size // 2)
                    x2 = min(frame.shape[1], x1 + size)
                    y2 = min(frame.shape[0], y1 + size)
                    
                    # 调整起始点，确保裁剪区域大小一致
                    if x2 - x1 != size:
                        x1 = max(0, x2 - size)
                    if y2 - y1 != size:
                        y1 = max(0, y2 - size)
                    
                    # 裁剪正方形区域
                    frame = frame[y1:y2, x1:x2]
                
                # 应用变换
                if self.transform:
                    frame = self.transform(frame)
                
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                return None
                
            frames = torch.stack(frames)  # [T, C, H, W]
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            return None
    
    def _resize_frames(self, frames):
        b, c, h, w = frames.shape
        
        # 保持宽高比的调整
        if b != self.num_frames:
            frames = F.interpolate(frames.unsqueeze(0), 
                                 size=(self.num_frames, h, w),
                                 mode='trilinear', 
                                 align_corners=False).squeeze(0)
        
        # 计算保持宽高比的新尺寸
        target_h, target_w = 288, 432  # 或其他适合的长宽比
        if h != target_h or w != target_w:
            frames = F.interpolate(frames, 
                                 size=(target_h, target_w),
                                 mode='bilinear',
                                 align_corners=False)
        
        return frames
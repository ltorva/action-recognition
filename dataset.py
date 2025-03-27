import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

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

class SkeletonExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def extract_keypoints(self, frame):
        results = self.pose.process(frame)
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                keypoints.append((x, y))
            print(f"提取到骨骼点: {keypoints}")  # 调试信息
            return keypoints
        print("未检测到骨骼点")  # 调试信息
        return None

class PersonDetector:
    def __init__(self):
        # 加载 Faster R-CNN 模型
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # 设置为评估模式
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def detect_person(self, frame, expand_ratio=0.4):
        """使用 Faster R-CNN 检测人体"""
        # 转换帧为张量
        img_tensor = F.to_tensor(frame).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # 进行推理
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]

        # 筛选出人体检测框
        person_boxes = []
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if label == 1 and score > 0.7:  # 1 是人体类别，置信度阈值为 0.7
                person_boxes.append(box.cpu().numpy())

        if len(person_boxes) > 0:
            # 返回最大的检测框
            x1, y1, x2, y2 = max(person_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            margin_x = int((x2 - x1) * expand_ratio)
            margin_y = int((y2 - y1) * expand_ratio)
            x1 = max(0, int(x1 - margin_x))
            y1 = max(0, int(y1 - margin_y))
            x2 = min(frame.shape[1], int(x2 + margin_x))
            y2 = min(frame.shape[0], int(y2 + margin_y))
            return x1, y1, x2, y2
        return None

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, temporal_transform=None, num_frames=16, use_skeleton=False):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.temporal_transform = temporal_transform if temporal_transform else RandomTemporalCrop(size=num_frames)
        
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
            
        self.person_detector = PersonDetector()  # 使用 Faster R-CNN 检测器
        self.use_skeleton = use_skeleton
        if use_skeleton:
            self.skeleton_extractor = SkeletonExtractor()
        
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
    
    def _detect_person(self, frame, prev_box=None):
        """使用HOG检测器检测人物，并动态调整检测框"""
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # 检测人物
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

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

            # 如果有上一帧的检测框，进行平滑
            if prev_box:
                x = int(0.8 * prev_box[0] + 0.2 * x)
                y = int(0.8 * prev_box[1] + 0.2 * y)
                w = int(0.8 * prev_box[2] + 0.2 * w)
                h = int(0.8 * prev_box[3] + 0.2 * h)

            return (x, y, w, h)
        return None

    def _load_frames_efficiently(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

            frames = []
            skeleton_points = []
            prev_box = None  # 用于跟踪上一帧的检测框

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # 检测人体并裁剪
                person_box = self.person_detector.detect_person(frame, expand_ratio=0.4)
                if person_box is not None:
                    x1, y1, x2, y2 = map(int, person_box)  # 确保坐标为整数
                    frame = frame[y1:y2, x1:x2]  # 裁剪出人体区域
                    prev_box = person_box  # 更新上一帧的检测框
                elif prev_box is not None:
                    # 如果当前帧未检测到人体，使用上一帧的检测框
                    x1, y1, x2, y2 = map(int, prev_box)  # 确保坐标为整数
                    frame = frame[y1:y2, x1:x2]

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 提取骨骼关键点
                if self.use_skeleton:
                    keypoints = self.skeleton_extractor.extract_keypoints(frame)
                    if keypoints:
                        skeleton_points.append(keypoints)

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

def mask_skeleton(frames, skeleton_points, mask_value=0):
    """基于骨骼点的 Mask 增强"""
    for frame, points in zip(frames, skeleton_points):
        for x, y in points:
            cv2.circle(frame, (x, y), radius=10, color=(mask_value, mask_value, mask_value), thickness=-1)
    return frames

def pose_cutmix(frames1, frames2, skeleton1, skeleton2, alpha=0.5):
    """基于骨骼点的 CutMix 增强"""
    mixed_frames = []
    for f1, f2, s1, s2 in zip(frames1, frames2, skeleton1, skeleton2):
        mask = np.zeros_like(f1)
        for x, y in s1:
            cv2.circle(mask, (x, y), radius=20, color=1, thickness=-1)
        mixed_frame = alpha * f1 * mask + (1 - alpha) * f2 * (1 - mask)
        mixed_frames.append(mixed_frame)
    return np.array(mixed_frames)
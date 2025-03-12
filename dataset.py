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
        start = random.randint(0, t - self.size)
        return frames[start:start+self.size]

class TemporalMask:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size
    
    def __call__(self, frames):
        # frames shape: (T, C, H, W)
        t = frames.shape[0]
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
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomRotation(10),
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
        
        # Read video frames more efficiently
        frames = self._load_frames_efficiently(full_path)
        
        if frames is None or len(frames) == 0:
            # Return a blank video if loading fails
            frames = torch.zeros((self.num_frames, 3, 224, 224))
            return frames, label
        
        # Apply temporal transforms
        if self.temporal_transform:
            frames = self.temporal_transform(frames)
        
        # Verify tensor shape
        if frames.shape != (self.num_frames, 3, 224, 224):
            frames = self._resize_frames(frames)
        
        # Reshape to (C, T, H, W) for the model
        frames = frames.permute(1, 0, 2, 3)
        return frames, label
    
    def _load_frames_efficiently(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise RuntimeError(f"No frames in video: {video_path}")
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Apply spatial transforms
                    if self.transform:
                        frame = self.transform(frame)
                    
                    frames.append(frame)
                else:
                    # If frame reading fails, add a blank frame
                    frames.append(torch.zeros(3, 224, 224))
            
            cap.release()
            return torch.stack(frames)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            return None
    
    def _resize_frames(self, frames):
        # Ensure frames have the correct shape
        b, c, h, w = frames.shape
        if b != self.num_frames:
            # Temporally resize
            frames = F.interpolate(frames.unsqueeze(0), size=(self.num_frames, h, w), 
                                 mode='trilinear', align_corners=False).squeeze(0)
        if h != 224 or w != 224:
            # Spatially resize
            frames = F.interpolate(frames, size=(224, 224), 
                                 mode='bilinear', align_corners=False)
        return frames
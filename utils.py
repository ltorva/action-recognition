import torch
import cv2
import numpy as np
from typing import List, Tuple
import os

def load_video_frames(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Load video frames from a video file and preprocess them.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        target_size: Target size for each frame (height, width)
    
    Returns:
        Tensor of shape (3, num_frames, height, width)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at index {idx}")
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    # Stack frames and convert to tensor
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(2, 0, 1, 3)
    
    return frames

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy given model outputs and targets.
    
    Args:
        outputs: Model outputs (N, C) where C is the number of classes
        targets: Ground truth labels (N,)
    
    Returns:
        Accuracy as a percentage
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100. * correct / total

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, best_acc: float, filename: str):
    """
    Save a checkpoint of the model and optimizer state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        best_acc: Best accuracy achieved so far
        filename: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   filename: str) -> Tuple[int, float]:
    """
    Load a checkpoint of the model and optimizer state.
    
    Args:
        model: The model to load the state into
        optimizer: The optimizer to load the state into
        filename: Path to the checkpoint file
    
    Returns:
        Tuple of (epoch, best_acc)
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_acc'] 
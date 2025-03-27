print("Script starting...")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
import time
from datetime import datetime
import shutil
import random
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your custom classes
from model import HARViT
from dataset import UCF101Dataset, RandomTemporalCrop, TemporalMask, GaussianNoise

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_previous_runs():
    """Clean up files from previous runs"""
    print("\n=== CLEANING UP PREVIOUS FILES ===")
    cleanup_targets = ["training_logs", "best_model.pth", "training_curves.png", 
                      "confusion_matrix.png", "nohup.out"]
    # Don't clean up checkpoints and saved models
    for target in cleanup_targets:
        try:
            if os.path.isfile(target):
                os.remove(target)
                print(f"Removed file: {target}")
            elif os.path.isdir(target) and "checkpoint" not in target and "saved_models" not in target:
                shutil.rmtree(target)
                print(f"Removed directory: {target}")
        except Exception as e:
            print(f"Could not remove {target}: {str(e)}")
    print("Cleanup completed")

def setup_logger():
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint.pth.tar"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
        shutil.copyfile(checkpoint_path, best_path)
        logging.info(f"Saved best model to {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch, best_val_acc, train_losses, val_losses, train_accs, val_accs
    else:
        logging.info("No checkpoint found, starting from scratch")
        return 0, 0.0, [], [], [], []

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, 
                scheduler=None, early_stopping_patience=5, wandb_logging=True, resume_from=None):
    # Set up mixed precision training based on device
    use_mixed_precision = device.type == 'cuda'  # Only use mixed precision on GPU
    scaler = GradScaler() if use_mixed_precision else None
    
    # Initialize or load checkpoint
    if resume_from:
        start_epoch, best_val_acc, train_losses, val_losses, train_accs, val_accs = load_checkpoint(
            resume_from, model, optimizer, scheduler)
        patience_counter = 0  # Reset patience counter on resume
    else:
        start_epoch = 0
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        patience_counter = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if use_mixed_precision:
                # GPU training with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU training without mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with device-specific info
            current_lr = optimizer.param_groups[0]['lr']
            progress_info = {
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*train_correct/train_total:.2f}%",
                'LR': f"{current_lr:.6f}",
                'Device': device.type
            }
            if device.type == 'cuda':
                progress_info['Memory'] = f"{torch.cuda.memory_allocated() / 1024**2:.0f}MB"
            progress_bar.set_postfix(progress_info)
            
            if wandb_logging:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_acc': 100.*train_correct/train_total,
                    'learning_rate': current_lr,
                    'gpu_memory_used': torch.cuda.memory_allocated() / 1024**2 if device.type == 'cuda' else 0
                })
            
            # Save checkpoint every 1000 batches
            if (batch_idx + 1) % 1000 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                }
                save_checkpoint(checkpoint, False, filename=f"checkpoint_epoch{epoch}_batch{batch_idx}.pth.tar")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log epoch results
        logging.info(
            f"\nEpoch {epoch+1}/{num_epochs} Summary:\n"
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%\n"
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
        )
        
        if wandb_logging:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
        
        # Save checkpoint at the end of each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
        }
        save_checkpoint(checkpoint, val_acc > best_val_acc, filename=f"checkpoint_epoch{epoch}.pth.tar")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model in a separate directory
            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")
            model_save_path = os.path.join("saved_models", f"best_model_acc{val_acc:.2f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'config': {
                    'img_size': model.img_size if hasattr(model, 'img_size') else 224,
                    'patch_size': model.patch_size if hasattr(model, 'patch_size') else 32,
                    'embed_dim': model.embed_dim,
                    'depth': len(model.blocks),
                    'num_heads': model.blocks[0].attn.num_heads,
                    'num_frames': model.num_frames,
                    'num_classes': model.num_classes
                }
            }, model_save_path)
            logging.info(f"Saved best model to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return train_losses, val_losses, train_accs, val_accs

def main():
    try:
        # 设置统一的配置参数
        config = {
            # 数据相关
            'num_frames': 16,
            'batch_size': 16,  # 减小batch_size以提高稳定性
            'num_classes': 12,
            'num_workers': 4,  # 根据CPU核心数调整
            
            # 模型结构 - 增加模型容量
            'img_size': 224,
            'patch_size': 16,  # 减小patch_size以获取更细粒度的特征
            'embed_dim': 256,  # 增加嵌入维度
            'depth': 4,       # 增加Transformer深度
            'num_heads': 8,   # 增加注意力头数
            
            # 训练参数 - 调整学习策略
            'learning_rate': 1e-4,  # 降低学习率
            'initial_learning_rate': 1e-4,
            'weight_decay': 0.05,   # 增加正则化
            'num_epochs': 100,      # 增加训练轮数
            'warmup_epochs': 5,     # 增加预热轮数
            'early_stopping': 15,   # 增加早停轮数
            'early_stopping_patience': 10,  # 增加早停耐心值
            
            # 新增优化参数
            'dropout_rate': 0.2,    # 添加dropout
            'gradient_clip': 1.0,   # 梯度裁剪 
            'label_smoothing': 0.1, # 标签平滑
            
            # 硬件优化
            'device': 'cuda',
            'mixed_precision': True,  # 使用混合精度训练
            'gradient_accumulation_steps': 2  # 梯度累积
        }

        # 初始化 wandb
        wandb.init(
            project="harvit-training",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )

        # Set random seed for reproducibility
        seed_everything(42)
        
        cleanup_previous_runs()
        log_file = setup_logger()
        logging.info("Training started")
        
        # Check for existing checkpoints
        checkpoint_dir = "checkpoints"
        resume_from = None
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch")])
            if checkpoints:
                resume_from = os.path.join(checkpoint_dir, checkpoints[-1])
                logging.info(f"Found checkpoint: {resume_from}")
        
        # Set device and log hardware info
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name}")
            logging.info(f"GPU Memory: {gpu_memory:.2f} GB")
            # Set GPU-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            logging.info("GPU not available, using CPU")
            # Set CPU-specific optimizations
            torch.set_num_threads(os.cpu_count())
            logging.info(f"Using {os.cpu_count()} CPU threads")
        
        # Create datasets with device-specific settings
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        temporal_transform = transforms.Compose([
            RandomTemporalCrop(size=config['num_frames']),
            TemporalMask(mask_size=2),
            GaussianNoise(std=0.1)
        ])
        
        # 修改数据集路径
        train_dataset = UCF101Dataset(
            root_dir="data/videos",          # 视频根目录
            split_file="data/train.txt",     # 训练集文件
            transform=transform,
            temporal_transform=temporal_transform,
            num_frames=config['num_frames']
        )
        
        val_dataset = UCF101Dataset(
            root_dir="data/videos",         # 视频根目录
            split_file="data/test.txt",     # 测试集文件
            transform=transform,
            temporal_transform=temporal_transform,
            num_frames=config['num_frames']
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True
        )
        
        # Initialize model with device-specific settings
        model = HARViT(
            img_size=224,
            patch_size=32,
            in_channels=3,
            num_classes=len(train_dataset.classes),
            embed_dim=128,
            depth=2,
            num_heads=4,
            num_frames=config['num_frames']
        ).to(device)
        
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['initial_learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['warmup_epochs'],
            T_mult=2
        )
        
        # Train model with resume capability
        train_losses, val_losses, train_accs, val_accs = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['num_epochs'],
            device=device,
            scheduler=scheduler,
            early_stopping_patience=config['early_stopping_patience'],
            wandb_logging=True,
            resume_from=resume_from
        )
        
        # Plot and save training curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        plt.title("Training and Validation Accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_curves.png")
        wandb.log({"training_curves": wandb.Image("training_curves.png")})
        
        # Finish wandb run
        wandb.finish()
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        import traceback
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        if wandb.run is not None:
            wandb.finish(exit_code=1)

if __name__ == "__main__":
    main()

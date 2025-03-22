# 创建测试脚本 check_gpu.py
import torch

def check_gpu():
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

if __name__ == '__main__':
    check_gpu()
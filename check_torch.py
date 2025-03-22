import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"PyTorch CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
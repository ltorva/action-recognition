import torch

def verify_cuda_setup():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

if __name__ == "__main__":
    verify_cuda_setup()
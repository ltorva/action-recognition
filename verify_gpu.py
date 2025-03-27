import torch
import sys
import os
import subprocess

def verify_gpu_setup():
    print("=== 系统环境信息 ===")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"是否为 CUDA 版本: {'cpu' not in torch.__version__}")
    print(f"PyTorch 安装位置: {torch.__file__}")
    
    print("\n=== CUDA 环境 ===")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if hasattr(torch.version, 'cuda'):
        print(f"PyTorch CUDA 版本: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"\n=== GPU 信息 ===")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        
        # 测试 GPU 功能
        print("\n=== GPU 功能测试 ===")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        torch.cuda.synchronize()
        print(f"矩阵乘法耗时: {start_time.elapsed_time(end_time):.2f} ms")
    else:
        print("\n=== 故障排查 ===")
        print("1. 检查 PyTorch 安装:")
        try:
            import pkg_resources
            installed_packages = [dist.key for dist in pkg_resources.working_set]
            torch_packages = [p for p in installed_packages if p in ['torch', 'torchvision', 'torchaudio']]
            for package in torch_packages:
                version = pkg_resources.get_distribution(package).version
                print(f"{package}: {version}")
        except Exception as e:
            print(f"包信息获取失败: {e}")

        print("\n2. 检查 NVIDIA 驱动:")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"无法运行 nvidia-smi: {e}")
        
        print("\n3. 检查 CUDA 环境变量:")
        cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k.upper()}
        if cuda_vars:
            for k, v in cuda_vars.items():
                print(f"{k}: {v}")
        else:
            print("未找到 CUDA 相关环境变量")

if __name__ == "__main__":
    verify_gpu_setup()
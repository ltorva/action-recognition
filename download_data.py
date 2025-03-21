import os
import requests
import zipfile
from tqdm import tqdm
import urllib3
import subprocess
import hashlib

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def verify_file(file_path, expected_size):
    """验证文件是否完整"""
    if not os.path.exists(file_path):
        return False
    return os.path.getsize(file_path) == expected_size

def download_file(url, filename):
    """下载文件并验证"""
    try:
        response = requests.get(url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        
        if verify_file(filename, total_size):
            print(f"{filename} 已存在且完整，跳过下载")
            return True
            
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        # 验证下载是否完整
        if not verify_file(filename, total_size):
            print(f"{filename} 下载不完整，将删除并重试")
            os.remove(filename)
            return False
        return True
    except Exception as e:
        print(f"下载出错: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def main():
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 数据集URLs和预期大小（字节）
    files_info = {
        'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar': {
            'size': 6916731831,
            'filename': 'UCF101.rar'
        },
        'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip': {
            'size': 1437338,
            'filename': 'UCF101TrainTestSplits-RecognitionTask.zip'
        }
    }
    
    # 下载文件
    for url, info in files_info.items():
        filename = os.path.join('data', info['filename'])
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            print(f"尝试下载 {filename} (第 {attempt + 1} 次)")
            if download_file(url, filename):
                break
            attempt += 1
            
        if attempt == max_attempts:
            print(f"下载 {filename} 失败，请手动下载：")
            print(url)
            continue
    
    # 解压文件
    print("开始解压文件...")
    
    # 解压 RAR 文件
    rar_file = os.path.join('data', 'UCF101.rar')
    if os.path.exists(rar_file):
        print("解压 RAR 文件...")
        if not extract_rar(rar_file, 'data/'):
            print("请手动使用 WinRAR 解压文件")
    
    # 解压 ZIP 文件
    zip_file = os.path.join('data', 'UCF101TrainTestSplits-RecognitionTask.zip')
    if os.path.exists(zip_file):
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('data')
            print("ZIP 文件解压完成")
        except zipfile.BadZipFile:
            print(f"ZIP 文件损坏，请重新下载")
            os.remove(zip_file)
            print("请从以下链接手动下载：")
            print("https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
    
    print("下载和解压操作完成！")

if __name__ == '__main__':
    main()
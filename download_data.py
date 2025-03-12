import os
import requests
import zipfile
from tqdm import tqdm
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, filename):
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True, verify=False)  # Disable SSL verification
    total_size = int(response.headers.get('content-length', 0))
    
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

def main():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # UCF101 dataset URLs (alternative sources)
    urls = [
        'https://storage.googleapis.com/deep-learning-data/UCF101.rar',
        'https://storage.googleapis.com/deep-learning-data/UCF101TrainTestSplits-RecognitionTask.zip'
    ]
    
    # Download files
    for url in urls:
        filename = os.path.join('data', url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                download_file(url, filename)
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                print("Please download the files manually from:")
                print("1. UCF101 dataset: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar")
                print("2. Train/Test splits: https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
                return
    
    # Extract files
    print("Extracting files...")
    
    # Extract UCF101.rar
    rar_file = os.path.join('data', 'UCF101.rar')
    if os.path.exists(rar_file):
        os.system(f'unar -o data/ {rar_file}')
    
    # Extract UCF101TrainTestSplits-RecognitionTask.zip
    zip_file = os.path.join('data', 'UCF101TrainTestSplits-RecognitionTask.zip')
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data')
    
    print("Download and extraction complete!")
    print("Dataset location: data/UCF101")
    print("Train/Test splits location: data/ucfTrainTestlist")

if __name__ == '__main__':
    main() 
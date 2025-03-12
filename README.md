# Action Recognition with Transformers

This project implements a video-based action recognition system using a combination of 3D ResNet and Transformer architectures. The system is designed to process video sequences and classify human actions in real-time.

## Features

- Video-based action recognition using R3D_18 backbone
- Transformer layers for temporal modeling
- Support for both GPU and CPU training
- Mixed precision training for improved performance
- Weights & Biases integration for experiment tracking
- Docker support for cloud deployment

## Project Structure

```
action-recognition/
├── data/               # Dataset directory
├── model.py           # Model architecture definition
├── train.py          # Training script
├── test.py           # Evaluation script
├── dataset.py        # Dataset and data loading utilities
├── utils.py          # Helper functions
├── download_data.py  # Script to download and prepare dataset
├── requirements.txt  # Python dependencies for local development
└── docker/          # Docker configuration for cloud deployment
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare the dataset:
```bash
python download_data.py
```

3. Configure your environment:
- Create a `.env` file with your Weights & Biases API key
- Ensure CUDA is installed for GPU support (optional)

## Training

To train the model:
```bash
python train.py
```

The script automatically detects available hardware and optimizes for either GPU or CPU training.

## Cloud Deployment

For cloud deployment:
1. Build the Docker image:
```bash
docker build -t action-recognition .
```

2. Use the provided `startup.sh` script for cloud instance setup

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
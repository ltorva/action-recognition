FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements_cloud.txt .
RUN pip3 install -r requirements_cloud.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV WANDB_API_KEY=b3720787c766a79531ea0b89e6bf2860363bc7e5

# Command to run the training
CMD ["python3", "train.py"] 
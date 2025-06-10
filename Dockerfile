FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install apt dependencies
RUN apt update --fix-missing && apt install -y \
    build-essential \
    vim \
    git \
    curl

# Clean up apt cache for smaller image size
RUN apt clean && rm -rf /var/lib/apt/lists/*

# Install latest version of pip
RUN pip install --upgrade pip

# Copy requirements.txt to the root of the folder
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
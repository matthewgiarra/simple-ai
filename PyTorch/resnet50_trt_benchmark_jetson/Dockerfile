# Dockerfile for demonstrating simple tensorcore benchmark
# Matthew Giarra <matthew.giarra@jhuapl.edu>

# Nvidia machine learning image for Jetson
FROM nvcr.io/nvidia/l4t-ml:r32.4.2-py3

# Clone and install NVIDIA torch2trt for 
# converting PyTorch models to TensorRT
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /torch2trt
RUN python3 setup.py install

WORKDIR /workspace

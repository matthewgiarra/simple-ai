# Dockerfile for demonstrating simple tensorcore benchmark
# Matthew Giarra <matthew.giarra@jhuapl.edu>

# Nvidia machine learning image for Jetson
FROM nvcr.io/nvidia/l4t-ml:r32.4.2-py3

# Uncomment below (and comment above) to build for normal machine (i.e., not a Jetson)
# FROM nvcr.io/nvidia/pytorch:19.06-py3

RUN pip3 install --upgrade --no-cache-dir pip

# Clone and install apex
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /apex
RUN pip install -v --no-cache-dir ./

# Uncomment this (and comment the above line) to perform the full apex build 
# (warning: this takes a long time) 
# RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace

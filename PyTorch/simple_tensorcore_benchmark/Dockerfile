# Dockerfile for demonstrating simple tensorcore benchmark
# Matthew Giarra <matthew.giarra@jhuapl.edu>

# Nvidia machine learning image for PyTorch
FROM nvcr.io/nvidia/pytorch:19.06-py3
WORKDIR /workspace
CMD (jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root --no-browser &) && /bin/bash
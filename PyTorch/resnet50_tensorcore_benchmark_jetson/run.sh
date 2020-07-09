#!/bin/bash
docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace resnet50_tensorcore_bechmark_jetson python3 resnet50_tensorcore_bechmark_jetson.py

# Uncomment below (and comment above) to run on normal machine (i.e., not a Jetson)
# docker run -it --rm --gpus all --network host --ipc=host -v $(pwd):/workspace resnet50_tensorcore_bechmark_jetson python3 resnet50_tensorcore_bechmark_jetson.py
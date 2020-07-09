#!/bin/bash
docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace resnet50_tensorcore_bechmark_jetson python3 resnet50_tensorcore_bechmark_jetson.py


#!/bin/bash
docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace resnet50_trt_benchmark_jetson python3 resnet50_trt_benchmark_jetson.py

exit
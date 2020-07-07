#!/bin/bash
docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace simple_tensorcore_benchmark_l4t-ml python3 simple_tensorcore_benchmark_jetson.py

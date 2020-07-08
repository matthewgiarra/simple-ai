#!/bin/bash
docker run -it --rm --gpus all --network host --ipc=host -v $(pwd):/workspace simple_tensorcore_benchmark python3 simple_tensorcore_benchmark.py

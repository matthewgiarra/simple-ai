#!/bin/bash
docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace simple_tensorcore_benchmark_jetson


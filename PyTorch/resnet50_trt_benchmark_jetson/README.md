# Resnet50 (image classifier) TensorRT Benchmark on Jetson

# Overview
This folder contains code to perform a speed comparison between deep neural networks using a native PyTorch model versus a model converted from PyTorch to TensorRT
We provide two files containing the same source code:

1. `resnet50_tensorcore_benchmark_jetson.py`: Python source code (quickest way to get something running)
2. `resnet50_tensorcore_benchmark_jetson.ipynb`: Jupyter notebook containing the same code (more flexible for experimenting).

Depending on your workflow, you may find one or the other easier to use. We provide instructions for both methods below.

# Assumptions
These instructions assume that you are operating on an NVIDIA Jetson with tensor cores (Volta architecture or newer, e.g., AGX Xavier), loaded with [Jetpack 4.4](https://developer.nvidia.com/embedded/jetpack) (release L4T R32.4.2) or newer, either directly or via `ssh`. 
 
# Prerequisites 
Upgrade docker to version > `19.03`:

```bash
sudo apt-get --only-upgrade install docker.io
```

# Instructions

1. Clone this repository

	```bash
	git clone git@gitlab.jhuapl.edu:giarrmn1/simpleai.git
	```
	
	If the device doesn't have access to the repository, you'll need to clone it on a machine that does have access and then copy it to the Jetson (e.g., using `scp`).
	
2. Build the docker image

	```bash
	cd simpleai/PyTorch/resnet50_trt_benchmark_jetson
	docker build --rm -t resnet50_trt_benchmark_jetson .
	```
	
3.  Run `run.sh` to launch the container and run the contents of 
`resnet50_trt_benchmark_jetson.py`

	```bash
	./run.sh
	```

	If you see an output similar to the following, it worked!
	
	```python
	Results summary (500 images)
	===============
	PyTorch  (FP32): 13.06 seconds  (38 FPS),  1.00x PyTorch FP32 speed
	PyTorch  (FP16): 7.16 seconds  (69 FPS),  1.82x PyTorch FP32 speed
	TensorRT (FP32): 4.24 seconds  (117 FPS),  3.08x PyTorch FP32 speed
	TensorRT (FP16): 1.46 seconds  (342 FPS),  8.95x PyTorch FP32 speed
	
	```

	## Variation: using the Jupyter notebook

	The instructions for using the Jupyter notebook are the same as above up to step 2.

4. Launch a docker container from the image we built:

	 ```bash
	 docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace resnet50_trt_benchmark_jetson 
	 ```
		
	You should see a prompt that says something like:
	
	```bash
	allow 10 sec for JupyterLab to start @ http://localhost:8888 (password nvidia)
	JupterLab logging location:  /var/log/jupyter.log  (inside the container)
	root@lowswap-xavier-dev3:/workspace# 
	``` 

5. Once you've launched the docker container, navigate a browser window to the following address:
`http://<hostname>:<port number>`

	- Replace `<hostname>` with `localhost` if your browser is running on the jetson, or with the name of the device if you've SSH'd into it (you can get this by typing `hostname` in the terminal window of the device you're SSH'd into)
	- `<port number>` is the number following the word `localhost` in the prompt from step 3 (default is `8888`)
	- For example, if the device is named `lowswap-xavier-dev3`, then navigate your browser to:
	`http://lowswap-xavier-dev3:8888`
	- If prompted for a password, enter `nvidia`

6. In the browser window, open the file `resnet50_trt_benchmark_jetson.ipynb` to view and run the code.

# Expected* Results

<I>*Note: This code is giving low-performance results right now! The purpose of posting the code this way is to solicit feedback about how to improve it.</I>

## Current results using Jetson AGX Xavier

### Configuration
- Platform: NVIDIA Jetson AGX Xavier
- Jetpack SDK: 4.4 ([L4T R32.4.3](https://developer.nvidia.com/embedded/jetpack))
- Power mode: [MAXN](https://www.jetsonhacks.com/2018/10/07/nvpmodel-nvidia-jetson-agx-xavier-developer-kit/) (`$ sudo nvpmodel -m 0`) 
- Docker source image: `nvcr.io/nvidia/l4t-ml:r32.4.3-py3`  ([link](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml))
- NVIDIA `torch2trt` for converting PyTorch models to TensorRT ([link](https://github.com/NVIDIA-AI-IOT/torch2trt))
- Model: `torchvision.models.resnet50()`
- Images: 224x224 pixel RGB

### Results
| Framework | Precision |  Execution time (sec) | Throughput (FPS) | Speed-up (vs. PyTorch FP32) |
|:----------:|:-------:|:-------------------:|:----------:|:--------:|
|   PyTorch  |  `Float32`|      13.06        |   38   |     1.00 |
|   PyTorch   |  `Float16`|      7.16        |   69   |     1.82 |
|   TensorRT |  `Float32`|      4.24        |   117   |     3.08 |
|   TensorRT |  `Float16`|      1.46        |   342   |     8.95 |

## Comparison with published results
Our results indicate that inference using `Float16` precision on Resnet50 yields about a 2x speedup compared to inference using `Float32` precision in PyTorch. Moreover, converting the model to TensorRT results in speed-ups of 3x and 9x for `Float32` and `Float16`, respectively, compared to `Float32` in PyTorch. This is significant, but still falls far short of the results posted on the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/jetson-xavier-nx-the-worlds-smallest-ai-supercomputer/) (shown below), which are over 5x faster than our best performance here (1941 FPS vs. our 342 FPS for 224x224 images).  

NVIDIA's results use `int8` precision for inference versus `FP16` used by our fastest model. This discrepancy probably accounts for a large fraction of the difference in performance. I need to read more to figure out how to convert models to `int8`.

![Image](https://developer.download.nvidia.com/devblogs/inferencing-performance.png)

## Caveats
The TensorRT models appear not to run with batch sizes greater than 1. The code doesn't crash, but the inference lines execute immediately for any batch size > 1. I verified via `tegrastats` that the device is not running out of memory. I currently don't know the cause of this issue, and it would be interesting to compare inference performance using larger batches.  

# Troubleshooting
If you can't upgrade docker or pull the docker image, try updating the system clock
```bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```
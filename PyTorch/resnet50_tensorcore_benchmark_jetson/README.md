# Resnet50 (image classifier) tensor core benchmark for NVIDIA Jetson

# Overview
This folder contains code to perform a speed comparison between deep neural networks with and without using tensor cores. 
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
	git clone http://github.com/matthewgiarra/simple-ai
	```
	
	If the device doesn't have access to the repository, you'll need to clone it on a machine that does have access and then copy it to the Jetson (e.g., using `scp`).
	
2. Build the docker image

	```bash
	cd simpleai/PyTorch/resnet50_tensorcore_benchmark_jetson
	docker build --rm -t resnet50_tensorcore_benchmark_jetson .
	```
	
3.  Run `run.sh` to launch the container and run the contents of 
`resnet50_tensorcore_benchmark_jetson.py`

	```bash
	./run.sh
	```

	If you see an output similar to the following, it worked!
	
	```python
	Results summary (3000 images)
	===============
	AMP O3: 9.62 seconds  (1.01x full precision speed)
	AMP O2: 9.67 seconds  (1.00x full precision speed)
	AMP O1: 9.66 seconds  (1.00x full precision speed)
	AMP O0: 9.73 seconds  (1.00x full precision speed)
	Float32: 9.70 seconds  (1.00x full precision speed)
	
	```

	## Variation: using the Jupyter notebook

	The instructions for using the Jupyter notebook are the same as above up to step 2.

4. Launch a docker container from the image we built:

	 ```bash
	 docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace resnet50_tensorcore_benchmark_jetson 
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

6. In the browser window, open the file `resnet50_tensorcore_benchmark_jetson.ipynb` to view and run the code.

# Expected* Results

<I>*Note: This code is giving low-performance results right now! The purpose of posting the code this way is to solicit feedback about how to improve it.</I>

### Configuration
- Platform: NVIDIA Jetson AGX Xavier
- Jetpack SDK: 4.4 ([L4T R32.4.3](https://developer.nvidia.com/embedded/jetpack))
- Power mode: [MAXN](https://www.jetsonhacks.com/2018/10/07/nvpmodel-nvidia-jetson-agx-xavier-developer-kit/) (`$ sudo nvpmodel -m 0`) 
- Docker source image: `nvcr.io/nvidia/l4t-ml:r32.4.3-py3`  ([link](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml))
- Apex build: [full build](https://github.com/NVIDIA/apex#quick-start)

### Results
| Precision| Execution time (sec) | Throughput (FPS) | Speed-up |
|:----------:|:----------------------:|:----------:|:--------:|
|   AMP O3 |        9.72        |   308   |     1.00 | 
|   AMP O2 |        9.67        |   310   |     1.01 |
|   AMP O1 |        9.65        |   310   |     1.01 |
|   AMP O0 |        9.72        |   308   |     1.00 |
|   Float32 |        9.70        |   309   |     1.00 |

## Comparison with published results
These results are much worse than what's posted on the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/jetson-xavier-nx-the-worlds-smallest-ai-supercomputer/). Specifically, the throughput is around 16% of their reported throughput using Resnet50 on images of the same size (1941 FPS for 224x224 images). 

![Image](https://developer.download.nvidia.com/devblogs/inferencing-performance.png)

## Why I think it should be faster

Our results indicate that inference using mixed precision (or even pure `Float16`) on `Resnet50` yields no speedup compared to inference using `Float32` precision. I've experimented with various batch sizes, and the results are not exceptionally different from what's above. This is contrary to my expectation. I think we should see a speed-up because the tensor cores should be invoked under the following circumstances, which I believe I've met: 

1. Device has tensor cores ([NVIDIA Jetson AGX has Volta architecture](http://info.nvidia.com/rs/156-OFN-742/images/Jetson_AGX_Xavier_New_Era_Autonomous_Machines.pdf))
2. Much of the work in the feed-forward process consists of convolutional layers, which [should invoke tensor cores for FP16 operations](https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use).

3. I'm using cuDNN 8.0, and [for "cudnn 7.3 and later, convolutions should use TensorCores for FP16 inputs"](https://discuss.pytorch.org/t/cnn-fp16-slower-than-fp32-on-tesla-p100/12146/4):

    ```python
    >>> import torch
    >>> print(torch.backends.cudnn.version())
    8000
    ```      


4. The number of input and output channels in each `Conv2d` layer is a multiple of 8 (except the 3-channel input to the first layer), which is a [requirement for tensor cores](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf). So are the dimensions of the fully connected `linear` layers. You can verify this by inspecting the output of the following commands:

    ```python
    >>> import torchvision
    >>> print(torchvision.models.resnet50())

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
          .
          .
          .

      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )
    ```
etc. 




# Troubleshooting
If you can't upgrade docker or pull the docker image, try updating the system clock
```bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```
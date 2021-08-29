# Simple Tensor Core Benchmark using NVIDIA `apex`

# Overview
This folder contains code to demonstrate running a simple neural network using tensor cores.
We provide two files containing the same source code:

1. `simple_tensorcore_benchmark_jetson.py`: Python source code (quickest way to get something running)
2. `simple_tensorcore_benchmark_jetson.ipynb`: Jupyter notebook containing the same code (more flexible for experimenting).

Depending on your workflow, you may find one or the other easier to use. We provide instructions for both methods below.

The example was adapted from one provided in [an NVIDIA presentation](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9998-automatic-mixed-precision-in-pytorch.pdf).

# Assumptions
These instructions assume that you are operating on an NVIDIA Jetson with tensor cores (Volta architecture or newer, e.g., Xavier NX or AGX Xavier), with [Jetpack 4.4](https://developer.nvidia.com/embedded/jetpack) (release L4T R32.4.2) or newer, either directly or via `ssh`. 

# Prerequisites 
Upgrade docker

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
	cd simpleai/PyTorch/simple_tensorcore_benchmark_jetson
	docker build --rm -t simple_tensorcore_benchmark_jetson .
	```
	<I> Note: Dockerfile defaults to python-only </I> `apex` <I> build. Follow instructions in the Dockerfile to perform the full </I>`apex` <I>build.</I>
	
3. Run `run.sh` to launch the container and run the contents of 
`simple_tensorcore_benchmark_jetson.py`
	
	```bash
	./run.sh
	```

	You should see a prompt similar to the following:
	
	```bash
	Results summary (100 iterations)
	===============
	Float32: 5.43 seconds  (1.00x full precision speed)
	AMP O0: 5.50 seconds  (0.99x full precision speed)
	AMP O1: 2.61 seconds  (2.08x full precision speed)
	AMP O2: 2.15 seconds  (2.53x full precision speed)
	AMP O3: 1.15 seconds  (4.73x full precision speed)
	```

	## Variation: using the Jupyter notebook
The instructions for using the Jupyter notebook are the same as above up to step 2. 

4. Launch a docker container from the image that you built in step 3 above:
	
	```bash
	docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace simple_tensorcore_benchmark_jetson
	``` 
	
	You should see a prompt similar to the following:
	
	```bash
	allow 10 sec for JupyterLab to start @ http://localhost:8888 (password nvidia)
	JupterLab logging location:  /var/log/jupyter.log  (inside the container)
	```
	
	Once you've launched the docker container, navigate a browser window to the following address:
	`http://<hostname>:<port number>`
	
	- Replace `<hostname>` with `localhost` if your browser is running on the jetson, or with the name of the device if you've SSH'd into it (you can get this by typing `hostname` in the terminal window of the device you're SSH'd into)
	- Replace `<port number>` with the number following the word `localhost` in the prompt above (default is `8888`). 
	- For example, if the device is named `lowswap-xavier-dev3`, then navigate your browser to:
	`http://lowswap-xavier-dev3:8888`
	- If prompted for a password, enter `nvidia`

5. In the browser window, open the file `simple_tensorcore_benchmark_jetson.ipynb` to view and run the code. 

# Expected Performance

## Jetson AGX Xavier (Python-only `apex` build)
| Precision| Execution time (sec) | Speed-up |
|:----------:|:----------------------:|:----------:|
|   Float32 |        5.43        |   1.00   |
|   AMP O0 |        5.50        |   0.99   |
|   AMP O1 |        3.22        |   1.68   |
|   AMP O2 |        2.89        |   1.88   |
|   AMP O3 |        1.16        |   4.69   |

## Jetson AGX Xavier (Full `apex` build)
| Precision| Execution time (sec) | Speed-up |
|:----------:|:----------------------:|:----------:|
|   Float32 |        5.43        |   1.00   |
|   AMP O0 |        5.50        |   0.99   |
|   AMP O1 |        2.61        |   2.08   |
|   AMP O2 |        2.16        |   2.52   |
|   AMP O3 |        1.15        |   4.72   |

# Troubleshooting
## Can't upgrade Docker or pull the Docker image
If you can't upgrade docker or pull the Docker image, try updating the system clock
```bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```

## Can't build the Docker image with full `apex` build
Building the Docker image with the [full apex build](https://github.com/NVIDIA/apex#quick-start) (versus the Python-only build) appears to require access to Cuda libraries at build-time. If the build fails, you may need to [change your default Docker runtime](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html#using-nv-container-runtime) to `nvidia`. This allows access to `cuda` libraries while building the Docker image (I think). 

To set nvidia as the default Docker runtime:

1. Create or modify the file `/etc/docker/daemon.json` to contain the following content:

	```bash
	{
	    "runtimes": {
	        "nvidia": {
	            "path": "/usr/bin/nvidia-container-runtime",
	            "runtimeArgs": []
	        }
	    },
	    "default-runtime": "nvidia"
	}
	
	```

1. Restart Docker daemon
 
	```bash
	sudo systemctl restart docker
	```
 
3. Build the Docker container as usual

	```bash
	docker build --rm -t simple_tensorcore_benchmark_jetson . 
	```
 
# Simple Tensor Core Benchmark using NVIDIA `apex`

# Overview
This folder contains code to demonstrate running a simple neural network using tensor cores.
We provide two files containing the same source code:

1. `simple_tensorcore_benchmark.py`: Python source code (quickest way to get something running)
2. `simple_tensorcore_benchmark.ipynb`: Jupyter notebook containing the same code (more flexible for experimenting).

Depending on your workflow, you may find one or the other easier to use. We provide instructions for both methods below.

The example was adapted from one provided in [an NVIDIA presentation](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9998-automatic-mixed-precision-in-pytorch.pdf).

# Assumptions
These instructions assume that you are logged in to a machine equipped with an NVIDIA GPU with tensor cores (Volta architecture or newer, e.g., RTX8000), either directly or via `ssh`. 

# Prerequisites 
Upgrade docker to version > `19.03`

```bash
sudo apt-get --only-upgrade install docker.io
```

# Instructions
1. Clone this repository
 
	```bash
	git clone git@gitlab.jhuapl.edu:giarrmn1/simpleai.git
	```
	
2. Build the docker image

	```bash
	cd simpleai/PyTorch/simple_tensorcore_benchmark
	docker build --rm -t simple_tensorcore_benchmark .
	```
		
3. Run `run.sh` to launch the container and run the contents of 
`simple_tensorcore_benchmark.py`
	
	```bash
	./run.sh
	```

	You should see a prompt similar to the following:
	
	```bash
	Results summary (500 iterations)
	===============
	Full precision: 12.57 seconds
	AMP O0: 12.78 seconds
	AMP O1: 4.21 seconds
	AMP O2: 3.65 seconds
	AMP O3: 2.84 seconds
	```

	## Variation: using the Jupyter notebook
The instructions for using the Jupyter notebook are the same as above up to step 2. 

4. Launch a docker container from the image that you built in step 2 above:
	
	```bash
	docker run -it --rm --gpus all --network host --ipc=host -v $(pwd):/workspace simple_tensorcore_benchmark
	``` 
	
	You should see a prompt similar to the following:
	
	```bash
	To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-52-open.html
    Or copy and paste one of these URLs:
        http://(arg-morty or 127.0.0.1):8888/?token=788ffb2f34c31f6cf0adb81b34346b0a3d8567b3b583924a
	```
	
5. Copy the line from the prompt that resembles the following text: 
	
	```bash
	http://(arg-morty or 127.0.0.1):8888/?token=788ffb2f34c31f6cf0adb81b34346b0a3d8567b3b583924a
	```
6. Paste the copied text into a browser window, replacing the text that resembles `http://(arg-morty or 127.0.0.1):8888` with `http://<host name>:8888`, where `<host name>` is the name of the machine you're SSH'd into. For example, if the name of your machine is `arg-morty`, then the line you paste into the browser should resemble the following:

	```bash
	http://arg-morty:8888/?token=788ffb2f34c31f6cf0adb81b34346b0a3d8567b3b583924a
	```
	<I>Notes:
	
	- The text following </I> `token=` <I> will be unique to your instance, and different from what's above. 
	- To determine your computer's name, enter `hostname` into the command prompt, e.g., </I>

	```bash
	root@arg-morty:/workspace# hostname
	arg-morty
	
	``` 

7. In the browser window, open the file `simple_tensorcore_benchmark.ipynb` to view and run the code. 

# Expected Performance
## Virtual machine with RTX8000 GPU

| Precision| Execution time (sec) | Speed-up |
|:----------:|:----------------------:|:----------:|
|   Full precision |        12.65        |   1.00   |
|   AMP O0 |        12.79        |   0.99   |
|   AMP O1 |        4.09        |   3.09   |
|   AMP O2 |        3.55        |   3.56   |
|   AMP O3 |        2.73        |   4.63   |


# Troubleshooting
## Can't upgrade Docker or pull the Docker image
If you can't upgrade docker or pull the Docker image, try updating the system clock

```bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```
 
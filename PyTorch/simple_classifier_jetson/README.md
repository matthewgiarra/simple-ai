# Simple Image Classifier (Alexnet) on Jetson

# Overview
This folder contains code to demonstrate using a simple image classification neural network ("Alexnet") to classify an image of a cat.
We provide two files containing the same source code:
- `simple_classifier_jetson.py`: Python source code (quickest way to get something running)
- `simple_classifier_jetson.ipynb`: Jupyter notebook containing the same code as `simple_classifier_jetson.py` (more flexible for modifying, experimenting, etc.)

Depending on your workflow, you may find one or the other easier to use. We provide instructions for both methods below.

# Assumptions
These instructions assume that you are operating on the Jetson, either directly or via `ssh`. 
 
# Prerequisites 
Upgrade docker
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
	cd simpleai/PyTorch/simple_classifier_jetson
	docker build --rm -t simple_classifier_jetson .
	```
	
3.  Run `run.sh` to launch the container and run the contents of 
`simple_classifier_jetson`

	```bash
	./run.sh
	```

	If you see a message that says `Persian cat`, it worked!

	## Variation: using the Jupyter notebook

	The instructions for using the Jupyter notebook are the same as above up to step 2.

4. Launch a docker container from the image we built:

	 ```bash
	 docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace simple_classifier_jetson 
	 ```
		
	You should see a prompt that says something like:
	
	```bash
	allow 10 sec for JupyterLab to start @ http://localhost:8888 (password nvidia)
	JupterLab logging location:  /var/log/jupyter.log  (inside the container)
	root@lowswap-tx2:/workspace# 
	``` 

5. Once you've launched the docker container, navigate a browser window to the following address:
`http://<hostname>:<port number>`

	- Replace `<hostname>` with `localhost` if your browser is running on the jetson, or with the name of the device if you've SSH'd into it (you can get this by typing `hostname` in the terminal window of the device you're SSH'd into)
	- `<port number>` is the number following the word `localhost` in the prompt from step 3 (default is `8888`)
	- For example, if the device is named `lowswap-tx2`, then navigate your browser to:
	`http://lowswap-tx2:8888`
	- If prompted for a password, enter `nvidia`

6. In the browser window, open the file `simple_classifier_jetson.ipynb` to view and run the code. If you see a message that says `Persian cat`, it worked!

# Troubleshooting
If you can't upgrade docker or pull the docker image, try updating the system clock
```bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```
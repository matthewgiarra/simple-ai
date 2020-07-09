# simpleai

Most online tutorials are needlessly complicated. The purpose of this website is to get you up and running with popular AI algorithms without needing a Ph.D. in computer science. 

# How to use this repository
Navigate to the subdirectory of the example that you want to explore. Each subdirectory contains:

- `README.md` with instructions for running that example.
- Dockerfile
- Source code with `.py` extension
- Same source code as a Jupyter notebook (`.ipynb` extension)
- `ruh.sh`, which launches a Docker container and runs the `.py` file

# Philosophy
Many online [examples](https://github.com/NVIDIA/DeepLearningExamples/) include custom functions to load their datasets, launch processes, calculate performance benchmarks, and so on.
This often obfuscates parts of the code that would otherwise help you understand how to implement the functionality in your own projects.  
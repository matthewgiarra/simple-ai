# Matthew Giarra <matthew.giarra@jhuapl.edu>

import torch
import torchvision
from torch2trt import torch2trt
import time # for timing execution

# This function times inference for a model
def time_inference(model, data, niter):
    
    # Warm-up inference loop.
    # Inference is often slowest on first iteration.
    # "Warm up" takes care of that.
    with torch.no_grad(): 
        for t in range(3):
            output = model(data)
    
    # Timed inference loop
    tic = time.perf_counter() # Start a timer
    with torch.no_grad():  # torch.no_grad() turns off gradient calculations for faster performance
        for t in range(niter): # Loop niter times
            output = model(data) # RUN THE INFERENCE
    toc = time.perf_counter() # Stop the timer
    exe_sec = toc-tic # Seconds elapsed
    FPS = niter / (exe_sec) # Frames per second
    return exe_sec, FPS

# Make input data
input_tensor_cpu = torch.rand([1,3,224,224])
input_tensor_gpu_full  = input_tensor_cpu.to('cuda')
input_tensor_gpu_half  = input_tensor_gpu_full.half()
# # # # # Get models  # # # # #  

# Pytorch, Float32
model_pytorch_fp32 = torchvision.models.resnet50(pretrained=False).eval().to('cuda')

# Pytorch, Float16
model_pytorch_fp16 = torchvision.models.resnet50(pretrained=False).eval().to('cuda').half()

# # # # # Convert models to TensorRT # # # # #

# TensorRT, Float32
model_trt_fp32 = torch2trt(model_pytorch_fp32, [input_tensor_gpu_full], fp16_mode=False)

# TensorRT, Float16
model_trt_fp16 = torch2trt(model_pytorch_fp16, [input_tensor_gpu_half], fp16_mode=True)

# Vectors to hold results
times_list = []
fps_list = []
names_list = []

# number of iterations per inference trial
niter = 500

# Do all the inferences
sec, fps = time_inference(model_pytorch_fp32, input_tensor_gpu_full, niter); times_list.append(sec); fps_list.append(fps); names_list.append("PyTorch  (FP32)")
sec, fps = time_inference(model_pytorch_fp16, input_tensor_gpu_half, niter); times_list.append(sec); fps_list.append(fps); names_list.append("PyTorch  (FP16)") 
sec, fps = time_inference(model_trt_fp32, input_tensor_gpu_full, niter); times_list.append(sec); fps_list.append(fps); names_list.append("TensorRT (FP32)")
sec, fps = time_inference(model_trt_fp16, input_tensor_gpu_half, niter); times_list.append(sec); fps_list.append(fps); names_list.append("TensorRT (FP16)")

# Print results
print("\nResults summary (%d images)\n===============" % (niter) )
for name, exe_sec, fps in zip(names_list, times_list, fps_list):
    print("%s: %0.2f seconds  (%d FPS),  %0.2fx PyTorch FP32 speed" % (name, exe_sec, fps, fps/fps_list[0]))
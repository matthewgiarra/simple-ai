# Matthew Giarra <matthew.giarra@jhuapl.edu>

import torch
import torchvision
import apex.amp as amp
import time # for timing execution

# Number of images per batch
batch_size = 100

# Number of iterations
niter = 30

# Results vectors
results_list = []
results_names = []

# Make data
input_batch_cpu = torch.rand([batch_size, 3, 224, 224], dtype=torch.float32)
input_batch_gpu_full  = input_batch_cpu.to('cuda')
input_batch_gpu_half  = input_batch_gpu_full.half()

# # # # # Inference with automatic mixed precision (AMP) via APEX  # # # # #  

# Run each of the APEX AMP optimization levels
for opt_level in ["O3", "O2", "O1", "O0"]:
    model = torchvision.models.resnet50(pretrained=False).eval().to('cuda')
    model_amp = amp.initialize(model, opt_level=opt_level)
    
    # Warm up
    with torch.no_grad():
        for t in range(3):
            output_gpu = model_amp(input_batch_gpu_half)

    # Run inference on the batch of images
    # torch.no_grad() turns off gradient calculations for faster performance
    tic = time.perf_counter()
    with torch.no_grad():
        for t in range(niter):
            output_gpu = model_amp(input_batch_gpu_half)
    # Execution time
    toc = time.perf_counter()
    print("AMP (opt level %s): %0.2f seconds" % (opt_level, toc-tic))
    
    # Results
    results_list.append(toc-tic)
    results_names.append('AMP ' + opt_level)
        
    
    
# # # # # Inference with full precision (Float32) # # # # #        

# Load the model
model = torchvision.models.resnet50(pretrained=False).eval().to('cuda')

# Warm up
with torch.no_grad():
    for t in range(3):
        output_gpu = model(input_batch_gpu_full)
      
# Run inference on the batch of images
# torch.no_grad() turns off gradient calculations for faster performance
tic = time.perf_counter()
with torch.no_grad():
    for t in range(niter):
        output_gpu = model(input_batch_gpu_full)
toc = time.perf_counter()
print("Float32: %0.2f seconds" % (toc-tic))

# Results
results_list.append(toc-tic)
results_names.append('Float32')

print("\nResults summary (%d images)\n===============" % (batch_size * niter) )
for name, result in zip(results_names, results_list):
    print("%s: %0.2f seconds  (%0.2fx full precision speed)" % (name, result, results_list[-1]/result))
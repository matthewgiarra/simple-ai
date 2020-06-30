# Adapted from: https://pytorch.org/hub/pytorch_vision_alexnet/
# Matthew Giarra <matthew.giarra@jhuapl.edu>
# Created: 30 June 2020
# Modified: 30 June 2020

from PIL import Image # For loading images
import torch
import torchvision
import numpy as np
import ast # For reading labels file

# Load the model
model = torchvision.models.alexnet(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load the image that we'll classify. 
input_image = Image.open('cat.jpg')

# Define some image preprocessing. 
# This is required for Alexnet trained on Imagenet.
# (i.e., this block is reusable)
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Normalize the image so it's consistent with training data
input_tensor = preprocess(input_image)

# Resize the image
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Turns off gradient calculation for better inference performance
with torch.no_grad():
    output = model(input_batch)

# If the model ran on the GPU, you have to copy
# the results back to the CPU to inspect them
output_cpu = output.cpu().detach().numpy()

# Find the max
idx = np.argmax(output_cpu)

# Read the labels file and print the entry 
# corresponding to the max value in the class vector
with open('imagenet_labels.txt', 'r') as file:
    contents = file.read()
    labels = ast.literal_eval(contents)
print(labels[idx])
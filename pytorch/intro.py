## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set_theme()

## Progress bar
from tqdm.notebook import tqdm

## Check Torch version 
import torch
print("Using torch", torch.__version__)

## By default, all tensors you create are stored on the CPU. 
## We can push a tensor to the GPU by using the function .to(...),
## or .cuda(). However, it is often a good practice to define a 
## device object in your code which points to the GPU if you have one, and otherwise to the CPU. 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

## Setup your code to be reproducible with the exact same random numbers
torch.manual_seed(42) # Setting the seed

x = torch.Tensor(2, 3, 4) # Allocates memory for a 2x3x4 tensor
print("x is", x)


## Let us try a matrix multiplication with torch.matmul
print("\nMatrix multiplication with torch.matmul\n")
x = torch.arange(6)
x = x.view(2, 3)
print("X", x)

W = torch.arange(9).view(3, 3)
print("W", W)

h = torch.matmul(x, W)
print("h", h)

## Dynamic Computation Graph and Backpropagation
## One of the main reasons for using PyTorch in Deep Learning 
## projects is that we can automatically get gradients/derivatives 
## of functions that we define.
print("\nDynamic Computation Graph and Backpropagation\n")

# By default when we create a tensor it does not require gradients
x = torch.ones((3,))
print(x.requires_grad)

x.requires_grad_(True) # Underscore indicates that it is an in-place operation
print(x.requires_grad)

x = torch.arange(3, dtype=torch.float32, requires_grad=True)
print("X", x)

a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
print("Y", y)

y.backward()  # This computes the gradients

print(x.grad)

## Check GPU support availability
gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")

## We can also compare the runtime of a large
## matrix multiplication on the CPU with a operation on the GPU
x = torch.randn(5000, 5000)

## CPU version
start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print(f"CPU time: {(end_time - start_time):6.5f}s")

## GPU version
x = x.to(device)
_ = torch.matmul(x, x)  # First operation to 'burn in' GPU
# CUDA is asynchronous, so we need to use different timing functions
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
_ = torch.matmul(x, x)
end.record()
torch.cuda.synchronize()  # Waits for everything to finish running on the GPU
print(f"GPU time: {0.001 * start.elapsed_time(end):6.5f}s")  # Milliseconds to seconds


# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
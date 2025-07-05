"""
Walk through of a lot of different useful Tensor Operations, where we
go through what I think are four main parts in:

1. Initialization of a Tensor
2. Tensor Mathematical Operations and Comparison
3. Tensor Indexing
4. Tensor Reshaping

But also other things such as setting the device (GPU/CPU) and converting
between different types (int, float etc) and how to convert a tensor to an
numpy array and vice-versa.

Programmed by Aladdin Persson
* 2020-06-27: Initial coding
* 2022-12-19: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import numpy as np

# ================================================================= #
#                        Initializing Tensor                        #
# ================================================================= #

device = "cuda" if torch.cuda.is_available() else "cpu"  # Cuda to run on GPU!

# Initializing a Tensor in this case of shape 2x3 (2 rows, 3 columns)
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

# A few tensor attributes
# print(
#     f"Information about tensor: {my_tensor}"
# )  # Prints data of the tensor, device and grad info
# print(
#     "Type of Tensor {my_tensor.dtype}"
# )  # Prints dtype of the tensor (torch.float32, etc)
# print(
#     f"Device Tensor is on {my_tensor.device}"
# )  # Prints cpu/cuda (followed by gpu number)
# print(f"Shape of tensor {my_tensor.shape}")  # Prints shape, in this case 2x3
# print(f"Requires gradient: {my_tensor.requires_grad}")  # Prints true/false

# Other common initialization methods (there exists a ton more)
x = torch.empty(size=(3, 3))  # Tensor of shape 3x3 with uninitialized data
x = torch.zeros((3, 3))  # Tensor of shape 3x3 with values of 0
x = torch.rand(
    (3, 3)
)  # Tensor of shape 3x3 with values from uniform distribution in interval [0,1)
x = torch.ones((3, 3))  # Tensor of shape 3x3 with values of 1
x = torch.eye(5, 5)  # Returns Identity Matrix I, (I <-> Eye), matrix of shape 2x3
x = torch.arange(
    start=0, end=5, step=1
)  # Tensor [0, 1, 2, 3, 4], note, can also do: torch.arange(11)
x = torch.linspace(start=0.1, end=1, steps=10)  # x = [0.1, 0.2, ..., 1]
x = torch.empty(size=(1, 5)).normal_(
    mean=0, std=1
)  # Normally distributed with mean=0, std=1
x = torch.empty(size=(1, 5)).uniform_(
    0, 1
)  # Values from a uniform distribution low=0, high=1
x = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3

# How to make initialized tensors to other types (int, float, double)
# These will work even if you're on CPU or CUDA!
tensor = torch.arange(4)  # [0, 1, 2, 3] Initialized as int64 by default

# print(f"Converted Boolean: {tensor.bool()}")  # Converted to Boolean: 1 if nonzero
# print(f"Converted int16 {tensor.short()}")  # Converted to int16
# print(
#     f"Converted int64 {tensor.long()}"
# )  # Converted to int64 (This one is very important, used super often)
# print(f"Converted float16 {tensor.half()}")  # Converted to float16
# print(
#     f"Converted float32 {tensor.float()}"
# )  # Converted to float32 (This one is very important, used super often)
# print(f"Converted float64 {tensor.double()}")  # Converted to float64

# Array to Tensor conversion and vice-versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_again = (
    tensor.numpy()
)  # np_array_again will be same as np_array (perhaps with numerical round offs)

# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# -- Addition --
z1 = torch.empty(3)
torch.add(x, y, out=z1)  # This is one way
z2 = torch.add(x, y)  # This is another way
z = x + y  # This is my preferred way, simple and clean.

# -- Subtraction --
z = x - y  # We can do similarly as the preferred way of addition

# -- Division (A bit clunky) --
z = torch.true_divide(x, y)  # Will do element wise division if of equal shape

# -- Inplace Operations --
t = torch.zeros(3)

t.add_(x)  # Whenever we have operation followed by _ it will mutate the tensor in place
t += x  # Also inplace: t = t + x is not inplace, bit confusing.

# -- Exponentiation (Element wise if vector or matrices) --
z = x.pow(2)  # z = [1, 4, 9]
z = x**2  # z = [1, 4, 9]


#matrix exponentiation
matrix_exp = torch.rand(size=(3,3))


# matrix multiplication

x1 = torch.randn((3,1))
print(x1)
x2 = torch.ones((1,3))
print(x2)
print(x1.mm(x2))

# batch matrix multiplication
batch= 32
n= 10
m= 20
p= 30

tensor = torch.rand((batch, n, m))
tensor2= torch.rand((batch,m,p))

print(tensor.bmm(tensor2).shape)

#example of broadcasting

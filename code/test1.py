import numpy
import torch

a = torch.ones(5)
b = a.numpy()
a += 1
print(a, b)
b += 1
print(a, b)

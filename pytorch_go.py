import torch

x = torch.rand(3, 5)
y = torch.empty(4, 3)
print(y)
z = torch.zeros(5, 2, dtype=torch.long)
x1 = torch.tensor([5.2, 3])
y = y.new_ones(5, 3, dtype=torch.float64)
print(y)

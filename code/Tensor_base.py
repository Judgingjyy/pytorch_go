import numpy as np
import torch

x = torch.rand(3, 5)
y = torch.empty(4, 3)
print(y)
z = torch.zeros(5, 2, dtype=torch.long)
x1 = torch.tensor([5.2, 3])
y = y.new_ones(5, 3, dtype=torch.float64)
print(y)
print(x.size())
print(x.shape)
print(x + y)
print(torch.add(x, y))
y.add_(x)
z = x.view(-1, 5)  # view()改变Tensor形状，-1所指的维度可以根据其他维度的值退出来
#               ,依旧共享内存
y = x[0, :]  # 这是索引，共享内存
x_cp = x.clone().view(15)  # 这就是新的了
x = torch.randn(1)  # it is strange
print(x.item())  # important tensor to num
x = torch.arange(1, 3).view(1, 2)
y = torch.arange(1, 4).view(3, 1)
print(x + y)  # 广播
# Tensor && numpy
a = torch.ones(5)
b = a.numpy()  # tensor to numpy

a = np.ones(5)
b = torch.from_numpy(a)  # 都是直接修改两个

c = torch.tensor(a)  # 拷贝了

# tensor on gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    y=torch.ones_like(x,device=device)
    x=x.to(device) # .to("cuda")
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))   

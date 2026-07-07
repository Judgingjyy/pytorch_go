import torch
x=torch.ones(2,2,requires_grad=True)

print(x.grad_fn)
y=x
y=y+1
y.requires_grad_(True)
z=y*y*3
out=z.mean()

print(z,out)
#learn so bad
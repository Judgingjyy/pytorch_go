import torch
X=torch.tensor([[1,2,3],[4,5,6]])
print(X.sum(dim=0,keepdim=True))
print(X.sum(dim=1,keepdim=True))
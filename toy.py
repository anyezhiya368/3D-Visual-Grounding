import torch

a = torch.tensor([0, 1, 2, 4])
b = torch.arange(4)
mask = a == b
print(torch.sum(mask))

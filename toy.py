import torch

a = torch.arange(9).reshape(3, 3)
b = torch.tensor([0, 1, 2]).unsqueeze(-1)
c = torch.arange(3).unsqueeze(-1)
print('A:', a)

import torch

a = torch.arange(9).reshape(3, 3)
a = a.unsqueeze(0)
b = a.repeat(2, 1, 1)
print(b)
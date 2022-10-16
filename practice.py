import torch

a = torch.tensor([1., 2., 3], dtype=torch.float16)

b = torch.tensor([1., 2., 3.], dtype=torch.float32)

a = a.float()

c = torch.subtract(a, b)

print(c)
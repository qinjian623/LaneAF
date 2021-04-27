import torch
import torchvision as tv
import torchvision.transforms as T


t = T.ColorJitter(.1, .1, .1)
t = T.RandomErasing(p=0.99)
data = torch.rand(3, 255, 255)
print((t(data) - data).mean().item())

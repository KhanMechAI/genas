from __future__ import print_function
import torch
import torchvision

from torch import nn

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)


print(torch.__version__)
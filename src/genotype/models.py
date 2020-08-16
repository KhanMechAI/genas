import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)

class InputBlock(nn.Module):
    def forward(self, x):
        return x


class OutBlock(nn.Module):
    def __init__(self, dropout_rate, out_features, classes):
        super(OutBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.out_features = out_features
        self.classes = classes

    def forward(self, value_dict):
        v = list(value_dict.values())[0]
        model = nn.Sequential(
            Flatten(),
            nn.Linear(
                in_features=v.view(v.shape[0],-1).shape[1],#v.shape[1],
                out_features=self.out_features
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(
                p=self.dropout_rate
            ),
            nn.Linear(
                in_features=self.out_features,
                out_features=self.classes
            ),
        )

        super(OutBlock, self).add_module(str(np.random.randint(0, 10000, size=1)), model)

        return model(v)


class ConvBlock(nn.Module):
    def __init__(self, out_channels, padding, pad_mode, dilation, stride, kernel):
        super(ConvBlock, self).__init__()
        self.out_channels = out_channels
        self.padding = padding
        self.pad_mode = pad_mode
        self.dilation = dilation
        self.stride = stride
        self.kernel = kernel

    def forward(self, value_dict):
        x = list(value_dict.values())[0]
        kernel = self.kernel
        if kernel > x.shape[2]:
            if x.shape[2] % 2:
                kernel = x.shape[2] - 1
            else:
                kernel = x.shape[2] - 2
            if kernel < 1:
                kernel = 1

        model = nn.Sequential(
            nn.Conv2d(
                in_channels=x.shape[1],
                out_channels=self.out_channels,
                kernel_size=kernel,
                padding=self.padding,
                padding_mode=self.pad_mode,
                dilation=self.dilation,
                stride=self.stride
            ),
            nn.ReLU(),
            nn.BatchNorm2d(
                num_features=self.out_channels
            ),
        )
        super(ConvBlock, self).add_module(str(np.random.randint(0, 10000, size=1)), model)

        return model(x)


class PoolBlock(nn.Module):

    def __init__(self, pool_func, kwargs):
        super(PoolBlock, self).__init__()
        self.model = nn.Sequential(
            pool_func(
                **kwargs
            )
        )

    def forward(self, x: dict):
        v = [i for i in x.values()][0]
        return self.model(v)

class BinaryBlock(nn.Module):
    def __init__(self, consolidated_processing_stack, num_inputs):
        super(BinaryBlock, self).__init__()
        self.consolidated_processing_stack = consolidated_processing_stack

        self.num_inputs = num_inputs

class SumBlock(BinaryBlock):
    def forward(self, kwargs):
        if self.num_inputs > 1:
            tensor = [self.consolidated_processing_stack[k](v) for k, v in kwargs.items()]
            return sum(tensor)
        else:
            return [v for v in kwargs.values()][0]


class ConcatBlock(BinaryBlock):
    def forward(self, kwargs):
        if self.num_inputs > 1:
            tensor = [self.consolidated_processing_stack[k](v) for k, v in kwargs.items()]
            return torch.cat(tensor, dim=1)
        else:
            return [v for v in kwargs.values()][0]
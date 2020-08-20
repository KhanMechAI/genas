import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)

class InputBlock(nn.Module):
    def __init__(self):
        super(InputBlock, self).__init__()
        # self.trace = []
        self.model = nn.Identity()

    def forward(self, x):
        # self.trace.append(x)
        x = self.model(x)
        return x

# class DynModule(nn.Module):
#

class OutBlock(nn.Module):
    def __init__(self, dropout_rate, in_features, out_features, classes):
        super(OutBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.in_features = in_features
        self.out_features = out_features
        self.classes = classes
        # self.registered = False
        self.model = self.model = nn.Sequential(
                Flatten(),
                nn.Linear(
                    in_features=self.in_features,  # v.shape[1],
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
        # self.trace = []

    def forward(self, value_dict):
        v = list(value_dict.values())[0]
        # self.trace.append(v)
        v = self.model(v)
        v = F.softmax(v)
        return v


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_width, padding, pad_mode, dilation, stride, kernel, id_name):
        super(ConvBlock, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = padding
        self.pad_mode = pad_mode
        self.dilation = dilation
        self.stride = stride

        self.model = None
        self.id_name = id_name

        self.kernel = kernel
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel,
                padding=self.padding,
                padding_mode=self.pad_mode,
                dilation=self.dilation,
                stride=self.stride
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(
                num_features=self.out_channels
            ),
        )
        # self.trace = []
        # self.registered = False

    def forward(self, value_dict):
        x = list(value_dict.values())[0]
        # self.trace.append(x)

            # self.registered = True
        return self.model(x)


class PoolBlock(nn.Module):

    def __init__(self, pool_func, kwargs):
        super(PoolBlock, self).__init__()
        self.model = nn.Sequential(
            pool_func(
                **kwargs
            )
        )
        # self.trace = []

    def forward(self, x: dict):
        v = [i for i in x.values()][0]
        # self.trace.append(v)
        return self.model(v)

class BinaryBlock(nn.Module):
    def __init__(self, consolidated_processing_stack, num_inputs):
        super(BinaryBlock, self).__init__()
        self.consolidated_processing_stack = nn.ModuleDict(consolidated_processing_stack)
        self.requires_grad_(True)
        self.num_inputs = num_inputs
        # self.trace = []

class SumBlock(BinaryBlock):
    def forward(self, kwargs):
        if self.num_inputs > 1:
            tensor = [self.consolidated_processing_stack[k](v) for k, v in kwargs.items()]
            return torch.add(*tensor)
        else:
            return [v for v in kwargs.values()][0]


class ConcatBlock(BinaryBlock):

    def forward(self, kwargs):
        if self.num_inputs > 1:
            tensor = [self.consolidated_processing_stack[k](v) for k, v in kwargs.items()]
            return torch.cat(tensor, dim=1)
        else:
            return [v for v in kwargs.values()][0]
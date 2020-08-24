# from __future__ import annotations

from collections import defaultdict
from functools import reduce

import numpy as np
from typing import Union, List, Callable, Dict, Type, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from networkx import DiGraph

from .models import InputBlock, OutBlock, ConvBlock, PoolBlock, SumBlock, ConcatBlock


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)


class Node(nn.Module):

    def __init__(self, node_id: int, predecessors: List = [], successors: List = [], ):
        super().__init__()
        self.requires_grad = True
        self.predecessors: List[Node] = predecessors  # where the results get pushed to
        self.successors: List[Node] = successors  # recieving inputs from

        self.inputs: Dict[str, torch.Tensor] = dict()  # list to store actual input values
        self.processed_inputs: Dict[str, torch.Tensor] = dict()  # list to store actual input values

        # need to adjust how lists are handled for sum and concat node
        self.in_channels: int = None
        self.out_channels: int = None

        self.in_height: int = None
        self.out_height: int = None

        self.in_width: int = None
        self.out_width: int = None

        self.model: nn.Module = None  # I pass a dictionary
        # of values that most functions only need the first element of, but some functions need the whole dict.

        self.node_id: int = node_id

        self.arity: int = None  # Max number of inputs
        self.num_inputs: int = None  # set this with the degree of the node from the graph

        self.initialised: bool = False

        self.name = 'Base Class Node'

        self.terminal = self.node_id == -1

        self.device: str = None  # gets updated by controller
        self.short_name = 'Base'

    @property
    def short_id(self):
        return f'{self.node_id}:{self.short_name}'

    @property
    def id_name(self):
        return f'{self.node_id}:{self.name.replace(" ", "_")}'

    def __str__(self):
        k = 4
        representation = (
            'General: ',
            f'{" " * k}ID: {self.node_id}',
            f'{" " * k}Initialised: {self.initialised}',
            f'{" " * k}Type: {self.name}',
            'Input Info: ',
            f'{" " * k}Channels: {self.in_channels}',
            f'{" " * k}Height: {self.in_height}',
            f'{" " * k}Width: {self.in_width}',
            'Output Info: ',
            f'{" " * k}Channels: {self.out_channels}',
            f'{" " * k}Height: {self.out_height}',
            f'{" " * k}Width: {self.out_width}',
            'Connectivity: ',
            f'{" " * k}Successors ID: {self.get_successor_id()}',
            f'{" " * k}Predecessors ID: {self.get_predecessor_id()}',
            f'{" " * k}Inputs: {self.num_inputs}',
        )
        return '\n'.join(representation, )

    def __repr__(self):
        rep = (
            f'{type(self).__name__}('
            f'node_id={self.node_id}, '
            f')'
        )
        return rep

    def count_parameters(self, requires_grad=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == requires_grad)

    def get_model(self):
        pass

    def node_name(self):
        return f'{self.name}:{self.node_id}'

    def successors_initialised(self):
        for suc in self.successors:
            if not suc.initialised:
                return False
        return True

    def inputs_full(self):
        return len(self.inputs.keys()) == self.num_inputs

    def forward(self, tensor):
        v = self.model(tensor)
        return v, self.get_predecessor_nid()

    def random_initialisation(self):
        # Allows overriding by necessary base classes, else does nothing
        pass

    def _initialise_predecessors(self):
        for pred in self.predecessors:
            pred.initialise()

    def _initialise(self):
        pass

    def initialise(self):
        # Some code to initialise the node
        if self.successors_initialised():
            self._initialise()  # Some code to initialise the node. method to be overwritten by children
            self.initialised = True

        if self.initialised:
            self._initialise_predecessors()

    def update_connectivity(self, graph: DiGraph, node_reference: Dict[int, object]):
        self.num_inputs = graph.out_degree(self.node_id)
        self.predecessors = [node_reference[x] for x in graph.predecessors(self.node_id)]
        self.successors = [node_reference[x] for x in graph.successors(self.node_id)]

    def get_successor_id(self):
        return [suc.node_id for suc in self.successors]

    def get_predecessor_id(self):
        return [pred.node_id for pred in self.predecessors]

    def get_successor_nid(self):
        return [suc.id_name for suc in self.successors]

    def get_predecessor_nid(self):
        return [pred.id_name for pred in self.predecessors]

    def out_features(self):
        return self.out_width * self.out_height * self.out_channels

    def local_params(self):
        params = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                if isinstance(v, int):
                    params[k] = v

        return params

    def verify_parameters(self):
        for k, v in self.local_params().items():
            if v <= 0:
                print(f'Warning: Negative or zero parameter. {k}: {v}')

    def update_device(self, device):
        self.device = device


class InputNode(Node):

    def __init__(self, node_id: int, channels: int, height: int, width: int):
        super().__init__(node_id)
        self.in_channels: int = channels
        self.out_channels: int = self.in_channels

        self.in_height: int = height
        self.out_height: int = self.in_height

        self.in_width: int = width
        self.out_width: int = self.in_width
        self.arity = 0
        self._initialise()
        self.model = nn.Identity()

        self.name = 'Input Node'
        self.short_name = 'In'

    def _initialise(self):
        self.initialised = True

    def __repr__(self) -> str:
        rep = (
            f'{type(self).__name__}('
            f'node_id={self.node_id}, '
            f'channels={self.in_channels}, '
            f'height={self.in_height}, '
            f'width={self.in_width}, '
            f')'
        )
        return rep


class OutputNode(Node):
    FEATURE_LIMIT = 64

    def __init__(self, classes: int):
        node_id = -1
        super().__init__(node_id)
        self.in_features: int = None
        self.out_features: int = None
        self.classes = classes
        self.arity = 0
        # self._initialise()

        self.name = 'Output Node'
        self.short_name = 'Out'

    def __repr__(self) -> str:
        rep = (
            f'{type(self).__name__}('
            f'node_id={self.node_id}, '
            f'out_features={self.out_features}, '
            f'classes={self.classes}, '
            f'width={self.in_width}, '
            f'[successors={self.get_successor_id()}] '
            f')'
        )
        return rep

    def _initialise(self):
        # arbitrary choice on these
        self.out_features = np.random.randint(self.classes, self.FEATURE_LIMIT)
        self.dropout_rate = np.random.uniform(high=0.5)

        self.in_features = self.successors[0].out_features()

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(
                in_features=self.in_features,  # v.shape[1],
                out_features=self.out_features
            ),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(
                p=self.dropout_rate
            ),
            nn.Linear(
                in_features=self.out_features,
                out_features=self.classes
            ),
            nn.Softmax()
        )
        self.initialised = True


class KernelNode(Node):
    # ARITY = 1
    KERNEL_CHOICES = (3, 3, 5, 7)

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.padding = 1
        self.stride = 1
        self.kernel = None
        self.dilation = 1
        self.arity = 1

        self.name = 'Kernel Class Node'

    def _out_dim(self, dim):
        return int((dim + 2 * self.padding - self.dilation * (self.kernel - 1) - 1) / self.stride) + 1

    def _out_width(self):
        return self._out_dim(self.in_width)

    def _out_height(self):
        return self._out_dim(self.in_height)

    def _calculate_out_params(self):
        self.out_width = self._out_width()
        self.out_height = self._out_height()

    def _random_initialisation(self):
        self.kernel = np.random.choice(KernelNode.KERNEL_CHOICES).item()
        self.padding = 1  # Can randomize later with np.random.randint()

    def _pre_initialise(self):
        pass

    def _initialise(self):
        successor = self.successors[0]  # only one successor for conv node
        self.in_channels = successor.out_channels
        self.in_height = successor.out_height
        self.in_width = successor.out_width

        if self.kernel > self.in_width:
            if self.in_width % 2:
                self.kernel = self.in_width - 1
            else:
                self.kernel = self.in_width - 2
            if self.kernel < 1:
                self.kernel = 1

        self.verify_parameters()

        self._pre_initialise()

        self._calculate_out_params()

        self.get_model()


class ConvNode(KernelNode):
    ARITY = 1
    PAD_MODES = ('zeros', 'replicate', 'circular')
    CHANNEL_CHOICES = (32, 64, 128, 256, 512)

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.pad_mode = 'zeros'
        self.name = 'Convolutional Node'
        self.short_name = 'Conv'

    def random_initialisation(self):
        super()._random_initialisation()
        self.out_channels = np.random.choice(ConvNode.CHANNEL_CHOICES).item()
        self.pad_mode = np.random.choice(ConvNode.PAD_MODES).item()

    def get_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
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
        return


class PoolNode(KernelNode):
    KERNEL_CHOICES = (2, 2, 2, 2, *KernelNode.KERNEL_CHOICES)  # potentially just override with 2

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.stride = 2
        self.name = 'Pool Class Node'

    def random_initialisation(self):
        super()._random_initialisation()
        self.kernel = np.random.choice(PoolNode.KERNEL_CHOICES).item()

    @staticmethod
    def _get_model(pool_func: Union[Type[nn.MaxPool2d], Type[nn.AvgPool2d]], kwargs: Dict = None):
        if kwargs is None:
            kwargs = {}
        return PoolBlock(pool_func, kwargs)

    def _pre_initialise(self):
        self.out_channels = self.in_channels


class MaxPoolNode(PoolNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'MaxPool Node'
        self.short_name = 'MaxPool'

    def get_model(self):
        self.model = nn.MaxPool2d(
            dilation=self.dilation,
            kernel_size=self.kernel,
            padding=self.padding,
            stride=self.stride,
        )
        return


class AvgPoolNode(PoolNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'AveragePool Node'
        self.short_name = 'AvgPool'

    def get_model(self):
        self.model = nn.AvgPool2d(
            kernel_size=self.kernel,
            padding=self.padding,
            stride=self.stride,
        )
        return


class BinaryNode(Node):
    # ARITY = 2

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.output_shape: tuple = ()
        self.arity = 2
        self.in_channels: Dict[str, int] = dict()
        self.in_height: Dict[str, int] = dict()
        self.in_width: Dict[str, int] = dict()
        self.in_sizes: Dict[str, int] = dict()

        self.max_channels = None

        self.processing_stack: Dict[str, nn.ModuleList] = defaultdict(nn.ModuleList)

        self.name = 'Binary Class Node'

        self.identity = nn.Identity()
        self.preprocessors: Dict[str, PreProcessingNode] = dict()

    def set_max_channels(self):
        self.max_channels = max(self.in_channels.values())

    def equal_size_input(self) -> bool:
        if self.num_inputs < 2:
            return True
        shapes = [[v for v in self.in_width.values()], \
                  [v for v in self.in_height.values()]]
        for d1, d2 in shapes:
            if d1 != d2:
                return False
        return True

    def shape_ids(self) -> Tuple[str, str]:
        # smallest input is defined as min(w*h) of all inputs
        return min(self.in_sizes, key=self.in_sizes.get), max(self.in_sizes, key=self.in_sizes.get)

    def resize_func(self, smaller_id) -> nn.Module:
        resize_shape = (self.in_height[smaller_id], self.in_width[smaller_id])

        return nn.Upsample(resize_shape, mode='bilinear')

    def add_resize_to_larger_input_stack(self):
        smaller_id, larger_id = self.shape_ids()

        self.processing_stack[larger_id].append(self.resize_func(smaller_id))

    def equal_channel_inputs(self) -> bool:
        return max(self.in_channels.values()) == min(self.in_channels.values())

    def add_channel_to_smaller_input_stack(self):
        smaller_id, larger_id = self.channel_ids()
        num_channels = max(self.in_channels.values()) - min(self.in_channels.values())

        self.processing_stack[smaller_id].append(PadModule(num_channels))

    def channel_ids(self) -> Tuple[str, str]:
        return min(self.in_channels, key=self.in_channels.get), max(self.in_channels, key=self.in_channels.get)

    def set_output_shape(self):
        small_id, _ = self.shape_ids()
        output_shape = (None, self.max_channels, self.in_height[small_id], self.in_width[small_id])
        self.out_channels = output_shape[1]
        self.out_height = output_shape[2]
        self.out_width = output_shape[3]
        self.output_shape = output_shape

    def make_preprocessors(self):
        for suc in self.successors:
            node_id = suc.id_name
            if node_id in self.processing_stack:
                pre_proc = self.processing_stack[node_id]
            else:
                pre_proc = nn.ModuleList([nn.Identity()])
            self.preprocessors[node_id] = PreProcessingNode(pre_proc)

    def _pre_initialise(self):
        for suc in self.successors:
            self.in_channels[suc.id_name] = suc.out_channels
            self.in_height[suc.id_name] = suc.out_height
            self.in_width[suc.id_name] = suc.out_width
            self.in_sizes[suc.id_name] = suc.out_width * suc.out_height

        self.set_max_channels()
        self.set_output_shape()
        self.get_processing_stack()
        self.make_preprocessors()


class SumNode(BinaryNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'Sum Node'
        self.short_name = 'Sum'

    def get_processing_stack(self):
        for suc in self.successors:
            self.processing_stack[suc.id_name] = nn.ModuleList()

        if not self.equal_size_input():
            self.add_resize_to_larger_input_stack()

        if not self.equal_channel_inputs():
            self.add_channel_to_smaller_input_stack()

        for k, l in self.processing_stack.items():
            if len(l) < 1:
                self.processing_stack[k].append(nn.Identity())

    def _initialise(self):
        self._pre_initialise()

    def forward(self, *inputs):
        if len(inputs) > 1:
            return_tensor = torch.add(*inputs)
        else:
            return_tensor = self.identity(*inputs)

        return return_tensor, self.get_predecessor_nid()


class ConcatNode(BinaryNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'Concat Node'
        self.short_name = 'Concat'

    # this needs to be child node dependent
    def get_processing_stack(self):

        for suc in self.successors:
            self.processing_stack[suc.id_name] = nn.ModuleList()

        if not self.equal_size_input():
            self.add_resize_to_larger_input_stack()

        for k, l in self.processing_stack.items():
            if len(l) < 1:
                self.processing_stack[k].append(nn.Identity())

    def _initialise(self):
        self._pre_initialise()

        self.out_channels = sum(self.in_channels.values())

    def forward(self, *inputs):
        if len(inputs) > 1:
            return_tensor = torch.cat(inputs, dim=1, )
        else:
            return_tensor = self.identity(*inputs)

        return return_tensor, self.get_predecessor_nid()


class PadModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.device = None

    def forward(self, tensor: torch.Tensor):
        shape = list(tensor.shape)
        shape[1] = self.channels  # TODO: Figure out how to make this before forward pass and move to __init__
        return torch.cat((tensor, torch.zeros(shape).to(self.device)), 1)


class PreProcessingNode(Node):
    def __init__(self, module_list: nn.ModuleList):
        super().__init__(node_id=None)
        self.module_list = module_list
        self.device = None
        self.name = 'Preprocessing Node'
        self.short_name = 'Prep'

    def forward(self, x):
        for l in self.module_list:
            l.device = self.device
            x = l(x)
        return x, self.get_predecessor_nid()

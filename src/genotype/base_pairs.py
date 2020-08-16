from __future__ import annotations

from collections import defaultdict
from functools import reduce

import numpy as np
from typing import Union, List, Callable, Dict, Type, Tuple, TypeVar
import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import Resize

from networkx import DiGraph


class Node(nn.Module):

    def __init__(self, node_id: int, predecessors: List[Node] = [], successors: List[Node] = [], ):
        super(Node, self).__init__()
        self.predecessors: List[Node] = predecessors  # where the results get pushed to
        self.successors: List[Node] = successors  # recieving inputs from

        self.inputs: Dict[int, torch.Tensor] = dict()  # list to store actual input values
        self.processed_inputs: Dict[str, torch.Tensor] = dict()  # list to store actual input values

        # need to adjust how lists are handled for sum and concat node
        self.in_channels: int = None
        self.out_channels: int = None

        self.in_height: int = None
        self.out_height: int = None

        self.in_width: int = None
        self.out_width: int = None

        self.model: Callable = lambda value_dict: (list(value_dict.values())[0])  # I pass a dictionary
        # of values that most functions only need the first element of, but some functions need the whole dict.

        self.node_id: int = node_id

        self.arity: int = None  # Max number of inputs
        self.num_inputs: int = None  # set this with the degree of the node from the graph

        self.initialised: bool = False

        self.name = 'Base Class Node'

        self.terminal = self.node_id == -1

    def __str__(self):
        representation = (
            'General: \n'
            '\t ID: {node_id}\n'
            '\t Initialised: {initialised}\n'
            '\t Type: {name}\n'
            '\n Input Info: \n'
            '\t Channels: {in_channels}\n'
            '\t Height: {in_height}\n'
            '\t Width: {in_width}\n'
            '\n Output Info: \n'
            '\t Channels: {out_channels}\n'
            '\t Height: {out_height}\n'
            '\t Width: {out_width}\n'
            '\n Connectivity: \n'
            f'\t Successors ID: {self.get_successor_id()}\n'
            f'\t Predecessors ID: {self.get_predecessor_id()}\n'
            '\t Inputs: {num_inputs}\n'
        ).format(**self.__dict__)
        return representation

    def __repr__(self):
        rep = (
            f'{type(self).__name__}(\n'
            f'\t node_id={self.node_id},\n'
            # f'\t [predecessors=[{self.predecessors}]],\n'
            # f'\t [successors=[{self.successors}]]\n'
            f'\t )'
        )
        return rep

    def get_function(self):
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

    def forward(self, node: Node, tensor: torch.Tensor):
        self.inputs[node.node_id] = tensor

        if self.inputs_full():
            x = self.model(self.inputs)
            if self.terminal:
                print(self)
                return x, self.node_id
            else:
                for pred in self.predecessors:
                    out = pred.forward(self, x)
                    if out is not None:
                        return out[0], out[1]

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

    def update_connectivity(self, graph: DiGraph, node_reference: Dict[int, Node]):
        self.num_inputs = graph.out_degree(self.node_id)
        self.predecessors = [node_reference[x] for x in graph.predecessors(self.node_id)]
        self.successors = [node_reference[x] for x in graph.successors(self.node_id)]

    def get_successor_id(self):
        return [suc.node_id for suc in self.successors]

    def get_predecessor_id(self):
        return [pred.node_id for pred in self.predecessors]


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

        self.name = 'Input Node'

    def _initialise(self):
        self.initialised = True

    def __repr__(self) -> str:
        rep = (
            f'{type(self).__name__}(\n'
            f'\t node_id={self.node_id},\n'
            f'\t channels={self.in_channels},\n'
            f'\t height={self.in_height},\n'
            f'\t width={self.in_width},\n'
            # f'\t [predecessors=[{self.predecessors}]],\n'
            # f'\t [successors=[{self.successors}]]\n'
            f'\t )'
        )
        return rep

    def forward(self, tensor: torch.Tensor):
        for pred in self.predecessors:
            x = pred.forward(self, tensor)
            if x is not None:
                return x


class OutputNode(Node):
    FEATURE_LIMIT = 10000
    def __init__(self, classes: int):
        node_id = -1
        super().__init__(node_id)

        self.out_features: int = None
        self.classes = classes
        self.arity = 0
        self._initialise()

        self.name = 'Output Node'

    def __repr__(self) -> str:
        rep = (
            f'{type(self).__name__}(\n'
            f'\t node_id={self.node_id},\n'
            f'\t out_features={self.out_features},\n'
            f'\t classes={self.classes},\n'
            f'\t width={self.in_width},\n'
            f'\t [successors={self.get_successor_id()}]\n'
            f'\t )'
        )
        return rep

    def _initialise(self):
        #arbitrary choice on these
        self.out_features = np.random.randint(self.classes**2, self.FEATURE_LIMIT)
        self.dropout_rate = np.random.uniform(high=0.5)

        self.model = self.get_function()
        self.initialised = True

    def get_function(self):
        def output_block(value_dict):
            x = list(value_dict.values())[0]
            x = torch.flatten(x, 1)
            x = nn.Linear(
                in_features=x.shape[1],
                out_features=self.out_features
            )(x)
            x = F.relu(x)
            x = nn.Dropout(
                p=self.dropout_rate
            )(x)
            x = nn.Linear(
                in_features=self.out_features,
                out_features=self.classes
            )(x)
            return x

        return output_block


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
        self._pre_initialise()

        self._calculate_out_params()

        self.model = self.get_function()


class ConvNode(KernelNode):
    ARITY = 1
    PAD_MODES = ('zeros', 'reflect', 'replicate', 'circular')
    CHANNEL_CHOICES = (32, 64, 128, 256, 512)

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.pad_mode = 'zeros'
        self.name = 'Conv Node'

    def random_initialisation(self):
        super()._random_initialisation()
        self.out_channels = np.random.choice(ConvNode.CHANNEL_CHOICES).item()
        self.pad_mode = np.random.choice(ConvNode.PAD_MODES).item()

    def get_function(self):
        def conv_block(value_dict):
            x = list(value_dict.values())[0]
            kernel = self.kernel
            if kernel > x.shape[2]:
                if x.shape.item() % 2:
                    kernel = x.shape - 1
                else:
                    kernel = x.shape - 2
                if not kernel:
                    kernel = 1

            x = nn.Conv2d(
                in_channels=x.shape[1],
                out_channels=self.out_channels,
                kernel_size=kernel,
                padding=self.padding,
                padding_mode=self.pad_mode,
                dilation=self.dilation,
                stride=self.stride
            )(x)
            x = F.relu(x)
            x = nn.BatchNorm2d(
                num_features=self.out_channels
            )(x)
            return x

        return conv_block


class PoolNode(KernelNode):
    KERNEL_CHOICES = (2, 2, 2, 2, *KernelNode.KERNEL_CHOICES)  # potentially just override with 2

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.stride = 2
        self.name = 'Pool Class Node'

    def random_initialisation(self):
        super()._random_initialisation()
        self.kernel = np.random.choice(PoolNode.KERNEL_CHOICES).item()

    def _get_function(self, pool_func: Union[Type[nn.MaxPool2d], Type[nn.AvgPool2d]], kwargs: Dict = {}):
        def pool(value_dict):
            x = pool_func(
                kernel_size=self.kernel,
                padding=self.padding,
                stride=self.stride,
                **kwargs
            )(list(value_dict.values())[0])
            return x

        return pool

    def _pre_initialise(self):
        self.out_channels = self.in_channels


class MaxPoolNode(PoolNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'MaxPool Node'

    def get_function(self):
        params = dict(
            dilation=self.dilation,
        )
        return self._get_function(pool_func=nn.MaxPool2d, kwargs=params)


class AvgPoolNode(PoolNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'Average Pool Node'

    def get_function(self):
        return self._get_function(pool_func=nn.AvgPool2d)


class BinaryNode(Node):
    # ARITY = 2

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.output_shape: tuple = ()
        self.arity = 2
        self.in_channels: Dict[int, int] = dict()
        self.in_height: Dict[int, int] = dict()
        self.in_width: Dict[int, int] = dict()
        self.in_sizes: Dict[int, int] = dict()

        self.processing_stack: Dict[int, List[Callable]] = defaultdict(list)
        self.consolidated_processing_stack: Dict[int, Callable] = {}

        self.name = 'Binary Class Node'

    def set_max_channels(self):
        self.max_channels = max(self.in_channels.values())

    def equal_size_input(self) -> bool:
        if self.num_inputs < 2:
            return True
        shapes = [[v for v in self.in_width.values()], [v for v in self.in_height.values()]]
        for d1, d2 in shapes:
            if d1 != d2:
                return False
        return True

    def shape_ids(self) -> Tuple[int, int]:
        # smallest input is defined as min(w*h) of all inputs
        return min(self.in_sizes, key=self.in_sizes.get), max(self.in_sizes, key=self.in_sizes.get)

    def resize_func(self, smaller_id) -> Callable:
        resize_shape = (self.in_height[smaller_id], self.in_width[smaller_id])

        def resize(x):
            return F.interpolate(x, resize_shape)

        return resize

    def add_resize_to_larger_input_stack(self):
        smaller_id, larger_id = self.shape_ids()
        self.processing_stack[larger_id].append(self.resize_func(smaller_id))

    def equal_channel_inputs(self) -> bool:
        return max(self.in_channels.values()) == min(self.in_channels.values())

    def add_channel_to_smaller_input_stack(self):
        smaller_id, larger_id = self.channel_ids()
        num_channels = max(self.in_channels.values()) - min(self.in_channels.values())

        def pad(tensor: torch.Tensor):
            shape = (tensor.shape[0], num_channels, tensor.shape[2], tensor.shape[3])
            # concatentate zero_layers to smaller tensor
            return torch.cat((tensor, torch.zeros(shape)), 1)

        # Pad the smaller tensor with zeros
        self.processing_stack[smaller_id].append(pad)

    def channel_ids(self) -> Tuple[int, int]:
        return min(self.in_channels, key=self.in_channels.get), max(self.in_channels, key=self.in_channels.get)

    def set_output_shape(self):
        small_id, _ = self.shape_ids()
        output_shape = (None, self.max_channels, self.in_height[small_id], self.in_width[small_id])
        self.out_channels = output_shape[1]
        self.out_height = output_shape[2]
        self.out_width = output_shape[3]
        self.output_shape = output_shape

    # from: https://stackoverflow.com/questions/16739290/composing-functions-in-python
    @staticmethod
    def _compose(f, g):
        return lambda arg: f(g(arg))

    # from: https://stackoverflow.com/questions/16739290/composing-functions-in-python
    @staticmethod
    def reduce_compose(*fs):
        return reduce(BinaryNode._compose, fs)

    def consolidate_processing_stack(self):
        for input_node_id in self.get_successor_id():
            if self.processing_stack[input_node_id]:
                self.consolidated_processing_stack[input_node_id] \
                    = self.reduce_compose(*self.processing_stack[input_node_id])
            else:
                self.consolidated_processing_stack[input_node_id] = lambda x: x

    def _pre_initialise(self):
        for suc in self.successors:
            self.in_channels[suc.node_id] = suc.out_channels
            self.in_height[suc.node_id] = suc.out_height
            self.in_width[suc.node_id] = suc.out_width
            self.in_sizes[suc.node_id] = suc.out_width * suc.out_height

        self.set_max_channels()
        self.set_output_shape()

    def _initialise(self):
        self._pre_initialise()


class SumNode(BinaryNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'Sum Node'

    # this needs to be child node dependent
    def get_processing_stack(self):

        if not self.equal_size_input():
            self.add_resize_to_larger_input_stack()

        if not self.equal_channel_inputs():
            self.add_channel_to_smaller_input_stack()

    def _initialise(self):
        self._pre_initialise()
        self.get_processing_stack()
        self.consolidate_processing_stack()
        self.model = self.get_function()

    def get_function(self):
        if self.num_inputs > 1:
            def sum_block(kwargs):
                tensor = [self.consolidated_processing_stack[k](v) for k, v in kwargs.items()]
                return sum(tensor)
        else:
            def sum_block(x):
                return x

        return sum_block


class ConcatNode(BinaryNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'Concat Node'

    # this needs to be child node dependent
    def get_processing_stack(self):
        if not self.equal_size_input():
            self.add_resize_to_larger_input_stack()

    def _initialise(self):
        self._pre_initialise()
        self.get_processing_stack()
        self.consolidate_processing_stack()
        self.model = self.get_function()

    def get_function(self):
        if self.num_inputs > 1:
            def concat_block(kwargs):
                tensor = [self.consolidated_processing_stack[k](v) for k, v in kwargs.items()]
                # a = self.consolidated_processing_stack[0](a)
                # b = self.consolidated_processing_stack[1](b)
                return torch.cat(tensor, dim=1)
        else:
            def concat_block(x):
                return x
        return concat_block

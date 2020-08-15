from collections import defaultdict
from functools import reduce

import numpy as np
from typing import Union, List, Callable, Dict, Type, Tuple
import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import Resize

import networkx as nx


class Node:

    def __init__(self, node_id: int, predecessors: List = [], successors: List = [], ):
        self.predecessors: List[Node] = predecessors  # where the results get pushed to
        self.successors: List[Node] = successors  # recieving inputs from

        self.inputs = Dict[int, torch.Tensor]  # list to store actual input values
        self.processed_inputs = Dict[str, torch.Tensor]  # list to store actual input values

        # need to adjust how lists are handled for sum and concat node
        self.in_channels: int = None
        self.out_channels: int = None

        self.in_height: int = None
        self.out_height: int = None

        self.in_width: int = None
        self.out_width: int = None

        self.function: Callable = None

        self.node_id: int = node_id

        self.arity: int = None  # Max number of inputs
        self.num_inputs: int = None  # set this with the degree of the node from the graph

        self.initialised: bool = False

        self.name = 'Base Class Node'

    def node_name(self):
        return f'{self.name}:{self.node_id}'

    def successors_initialised(self):
        for suc in self.successors:
            if not suc.initialised:
                return False
        return True

    def inputs_full(self):
        return len(self.inputs.keys()) == self.num_inputs

    # TODO: get the function call and pushing of inputs working
    def add_input(self, node, tensor: torch.Tensor):
        self.inputs[node.id] = tensor
        if self.inputs_full():
            x = self.function(**self.processed_inputs)
            for pred in self.predecessors:
                pred.add_input(self, x)


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

    def update_connectivity(self, graph: nx.DiGraph, node_reference: Dict[int, object]):
        self.num_inputs = graph.out_degree(self.node_id)
        self.predecessors = [node_reference[x] for x in graph.predecessors(self.node_id)]
        self.predecessors = [node_reference[x] for x in graph.successors(self.node_id)]

    def __repr__(self):
        return repr(self.__dict__)


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


class KernelNode(Node):
    # ARITY = 1
    KERNEL_CHOICES = (3, 3, 5, 7)

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.padding = None
        self.stride = 1
        self.kernel = None
        self.dilation = 1
        # self.min_width = min(self.in_width)
        # self.min_height = min(self.in_height)
        self.arity = 1

        self.name = 'Kernel Class Node'

    def _out_dim(self, dim):
        return int((self.dim + 2 * self.padding - self.dilation * (self.kernel - 1) - 1) / self.stride) + 1

    def _out_width(self):
        return self._out_dim(self.in_width)

    def _out_height(self):
        return self._out_dim(self.in_height)

    def _calculate_out_params(self):
        self.out_width = self._out_width()
        self.out_height = self._out_height()

    def random_initialisation(self):
        self.kernel = np.random.choice(KernelNode.KERNEL_CHOICES).item()
        self.padding = 1  # Can randomize later with np.random.randint()

    def get_function(self):
        pass

    def _initialise(self):
        successor = self.successors[0]  # only one successor for conv node
        self.in_channels = successor.out_channels
        self.in_height = successor.out_height
        self.in_width = successor.out_width

        self._calculate_out_params()

        self.function = self.get_function()


class ConvNode(KernelNode):
    ARITY = 1
    PAD_MODES = {'zeros', 'reflect', 'replicate', 'circular'}
    CHANNEL_CHOICES = (32, 64, 128, 256, 512)

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.pad_mode = None
        self.name = 'Conv Node'

    def random_initialisation(self):
        super().random_initialisation()
        self.out_channels = np.random.choice(ConvNode.CHANNEL_CHOICES).item()

    def get_function(self):
        def conv_block(x):
            x = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
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
        super().random_initialisation()
        self.kernel = np.random.choice(PoolNode.KERNEL_CHOICES).item()

    def _get_function(self, pool_func: Union[Type[nn.MaxPool2d], Type[nn.AvgPool2d]]):
        def pool(x):
            x = pool_func(
                kernel_size=self.kernel,
                padding=self.padding,
                dilation=self.dilation,
                stride=self.stride
            )(x)
            return x

        return pool


class MaxPoolNode(PoolNode):

    def __init__(self, node_id: int, ):
        super().__init__(node_id)
        self.name = 'MaxPool Node'

    def get_function(self):
        return self._get_function(pool_func=nn.MaxPool2d)


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

        self.processing_stack: Dict[int, List[Callable]] = defaultdict(List)
        self.consolidated_processing_stack: Dict[int, Callable] = {}

        self.name = 'Binary Class Node'

    def set_max_channels(self):
        self.max_channels = max(self.in_channels.values())

    def equal_size_input(self) -> bool:
        shapes = [[v for v in self.in_width.values()], [v for v in self.in_height.values()]]
        for d1, d2 in shapes:
            if d1 != d2:
                return False
        return True

    def shape_ids(self) -> Tuple[int, int]:
        # smallest input is defined as min(w*h) of all inputs
        return min(self.in_sizes, key=self.in_sizes.get), min(self.in_sizes, key=self.in_sizes.get)

    def resize_func(self, id) -> Resize:
        resize_shape = (None, self.max_channels, self.in_height[id], self.in_width[id])
        return Resize(size=resize_shape)

    def add_resize_to_larger_input_stack(self):
        smaller_id, larger_id = self.shape_ids()
        self.processing_stack[larger_id].append(self.resize_func(smaller_id))

    def equal_channel_inputs(self) -> bool:
        return max(self.in_channels.values()) == min(self.in_channels.values())

    def add_channel_to_smaller_input_stack(self):
        smaller_id, larger_id = self.channel_ids()
        num_channels = max(self.in_channels.values()) - min(self.in_channels.values())
        shape = (None, num_channels, self.output_shape[3], self.output_shape[4])

        def pad(tensor):
            # concatentate zero_layers to smaller tensor
            return torch.cat((tensor, torch.zeros(shape)), 1)

        # Pad the smaller tensor with zeros
        self.processing_stack[smaller_id].append(pad)

    def channel_ids(self) -> Tuple[int, int]:
        return min(self.in_channels, key=self.in_channels.get), max(self.in_channels, key=self.in_channels.get)

    def set_output_shape(self):
        small_id, _ = self.shape_ids()
        self.output_shape = (None, self.max_channels, self.in_height[small_id], self.in_width[small_id])

    # from: https://stackoverflow.com/questions/16739290/composing-functions-in-python
    @staticmethod
    def _compose(f, g):
        return lambda arg: f(g(arg))

    # from: https://stackoverflow.com/questions/16739290/composing-functions-in-python
    @staticmethod
    def reduce_compose(*fs):
        return reduce(BinaryNode._compose, fs)

    def consolidate_processing_stack(self):
        for input_node_id in self.inputs.keys():
            self.consolidated_processing_stack[input_node_id] \
                = self.reduce_compose(*self.processing_stack[input_node_id])

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

    def get_function(self):
        def sum_block(a, b):
            a = self.consolidated_processing_stack[0](a)
            b = self.consolidated_processing_stack[0](b)
            x = a + b
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

    def get_function(self):
        def concat_block(a, b):
            a = self.consolidated_processing_stack[0](a)
            b = self.consolidated_processing_stack[0](b)
            return torch.cat((a, b), dim=1)

        return concat_block

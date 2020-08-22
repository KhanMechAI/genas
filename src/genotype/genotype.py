from collections import defaultdict
from queue import Queue
from typing import Union, List, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from networkx import DiGraph

from .nodes import ConvNode, MaxPoolNode, AvgPoolNode, SumNode, ConcatNode, InputNode, Node, \
    BinaryNode, PoolNode, OutputNode


def check_full_inputs(input_dict):
    candidates = []
    for candidate, in_d in input_dict.items():
        if not any([x is None for x in in_d.values()]):
            candidates.append(candidate)

    return candidates


class Controller(nn.Module):
    def __init__(self, module_dict, network_map, entry_point, network_dir, device: str = 'cpu'):
        super(Controller, self).__init__()
        self.module_dict = module_dict  # to register parameters
        self.network_map = network_map  # register of all nodes
        self.entry_point: str = entry_point
        self.input_node = network_map[entry_point]  # to enter the model
        self.device: str = device
        self.network_dir = network_dir

        self.update_device(device)

    def forward(self, x):
        next_node = [self.entry_point]
        current_out = defaultdict(dict)
        current_out[self.entry_point][self.entry_point] = x
        computed_nodes = set()

        while next_node != []:
            next_id = next_node.pop(0)
            inputs = list(current_out[next_id].values())
            output, to_nodes = self.network_map[next_id](*inputs)
            # output.retain_grad()
            for n in to_nodes:
                if n not in current_out:
                    current_out[n] = {k: None for k in self.network_dir[n]['in']}

                current_out[n][next_id] = output

            if next_id in current_out:
                del current_out[next_id] # prevents storing too much info. Shouldnt need the key any more
            computed_nodes.add(next_id)
            candidates = check_full_inputs(current_out)
            [next_node.append(c) for c in candidates if c not in next_node and c not in computed_nodes]

        return output

    def update_device(self, device):
        self.device = device
        for name, mod in self.network_map.items():
            mod.update_device(device)


class RandomArchitectureGenerator:
    MAX_DEPTH = 100
    MIN_DEPTH = 5
    MIN_NODES = 5
    MAX_ITER = 100
    NODE_TYPES = {'BINARY', 'CONV', 'POOL', 'INPUT', 'REFERENCE'}

    IMAGE_CHANNELS = 3
    DEFAULT_IMAGE_SIZE = 128  # 128x128 assuming square

    def __init__(self, prediction_classes: int, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_nodes: int = MIN_NODES,
                 image_size: Union[int, tuple, list] = DEFAULT_IMAGE_SIZE, input_channels: int = IMAGE_CHANNELS):

        super(RandomArchitectureGenerator, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_nodes = min_nodes
        if isinstance(image_size, int):
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_height = image_size[0]
            self.image_width = image_size[1]

        self.prediction_classes = prediction_classes

        self.image_size = image_size
        self.input_channels = input_channels

        self.target_depth = np.random.randint(self.min_depth, self.max_depth)

        self.level = 0
        self.nonleaf = False

        self.root_id = 0
        self.next_node_id = self.new_node_id(start_node=self.root_id)

        self.queue = Queue()
        self.queue.put((self.root_id, self.level))

        self.leaf_nodes = set()

        self.max_pool = np.floor(np.log2(self.image_width))  # as per paper
        self.pool_nodes = []

        self.binary_nodes = []

        self.input_nodes: Union[List, int] = []  # will implement a consolidation of input nodes to a single node

        self.graph = DiGraph()

        self.node_reference: Dict[int, Node] = {}

        root_node = self.random_new_node({'REFERENCE', 'INPUT', })
        output_node = OutputNode(prediction_classes)

        self.add_new_node(output_node)
        self.add_new_node(root_node, predecessor_id=output_node.node_id)

        self._module_dict = nn.ModuleDict()
        self.network_map = nn.ModuleDict()
        self.network_directory = {}
        self.state  = False

    @staticmethod
    def new_node_id(start_node=1, step=1):
        n = start_node
        while True:
            yield n
            n += step

    @property
    def pool_count(self):
        return len(self.pool_nodes)

    def random_node_from_graph(self) -> int:
        num_nodes = self.graph.number_of_nodes()
        if num_nodes == 1:
            return 1
        else:
            # Random node from 1 to num nodes, so that the output node isnt selected
            return np.random.choice(range(1, num_nodes - 1))

    def connected_to_input(self, node) -> bool:
        connections = self.graph.neighbors(node)
        for neighbor in connections:
            if isinstance(self.node_reference[neighbor], InputNode):
                return True
        return False

    def disallowed_types(self, node_id, level) -> set:
        restricted_types = set()
        connected_to_input = self.connected_to_input(node_id)

        if connected_to_input or not self.nonleaf:
            restricted_types.add('INPUT')

        if (level + 2 == self.target_depth) or isinstance(self.node_reference[node_id], BinaryNode):
            restricted_types.add('BINARY')

        if (level < self.min_depth):
            restricted_types.add('REFERENCE')

        return restricted_types

    def create_new_node(self, node_type, node_id=None):
        if node_id is None:
            # allows for reassignment of nodes, otherwise, yield new node number
            node_id = next(self.next_node_id)
        if node_type == 'SUM':
            new_node = SumNode(node_id, )
            self.binary_nodes.append(node_id)
        elif node_type == 'CONCAT':
            new_node = ConcatNode(node_id)
            self.binary_nodes.append(node_id)
        elif node_type == 'CONV':
            new_node = ConvNode(node_id)
        elif node_type == 'MAX':
            new_node = MaxPoolNode(node_id)
            self.pool_nodes.append(node_id)
        elif node_type == 'AVERAGE':
            new_node = AvgPoolNode(node_id)
            self.pool_nodes.append(node_id)
        elif node_type == 'INPUT':
            new_node = InputNode(
                node_id=node_id,
                channels=self.input_channels,
                height=self.image_height,
                width=self.image_width
            )
            self.input_nodes.append(node_id)

        new_node.random_initialisation()

        return new_node

    def random_new_node(self, restricted_types: set) -> Node:
        valid_types: Tuple[str, ...] = tuple(self.NODE_TYPES - restricted_types)

        new_type = np.random.choice(valid_types, size=1).item()

        if new_type == 'BINARY':
            q = np.random.uniform()
            if q < 0.5:
                new_type = 'SUM'
            else:
                new_type = 'CONCAT'
        elif new_type == 'POOL':
            q = np.random.uniform()
            if q < 0.5:
                new_type = 'MAX'
            else:
                new_type = 'AVERAGE'
        elif new_type in {'CONV', 'INPUT'}:
            pass
        else:
            new_type = None
            new_node = None

        if new_type is not None:
            new_node = self.create_new_node(new_type)

        return new_node

    def add_new_node(self, node_type: Union[str, Node], predecessor_id: int = None):

        # Check if the node is already created. Therefore would already ahve a node ID
        if isinstance(node_type, Node):
            node = node_type
        # Else we yield a new ID and create the node
        else:
            node = self.create_new_node(node_type=node_type)

        self.graph.add_node(node.node_id)
        if predecessor_id is not None:
            self.graph.add_edge(predecessor_id, node.node_id)

        self.node_reference[node.node_id] = node

        return node

    def get_architecture(self, reset_on_finish=False) -> Union[int, nn.Module]:  # Tuple[DiGraph, Dict[int, Node], Node]:
        current_level = 0
        if self.state:
            self.reset()
        while not self.queue.empty():
            node_id, current_level = self.queue.get()
            arity = self.node_reference[node_id].arity

            if current_level != self.level:
                self.nonleaf = False
                self.level = current_level
            i = 0
            while i < arity:
                if current_level + 1 == self.target_depth:
                    self.add_new_node(node_type='INPUT', predecessor_id=node_id)
                else:
                    restricted_node_types = self.disallowed_types(node_id, current_level)
                    new_node = self.random_new_node(restricted_types=restricted_node_types)
                    if new_node is None:
                        self.leaf_nodes.add(node_id)
                    else:
                        self.add_new_node(node_type=new_node, predecessor_id=node_id)

                        if not isinstance(new_node, InputNode):
                            self.queue.put((new_node.node_id, current_level + 1))
                            self.nonleaf = True

                i += 1

        print(f'Final depth:{self.level}')
        print(f'Number of nodes:{len(self.graph.nodes)}')
        if current_level < self.min_depth and len(self.graph.nodes) < self.min_nodes:
            print('Degenerate Graph or less than min depth. Resetting')
            self.reset()
            return None

        self.check_input_nodes()
        self.add_missing_edges()
        self.prune_pool_nodes()
        self.contract_input_nodes()
        self.update_nodes()

        # Now that the graph is complete, the nodes can be initialised
        self.initialise_nodes()
        self.compile_model()
        self.decompose_binary_nodes()
        self.update_nodes() #Need to re-wire the assignments from the decomposition

        self.compile_model()

        if reset_on_finish:
            self.reset()
            return -1

        return self.controller()

    def entry_point_name(self):
        return self.node_reference[self.input_nodes].id_name

    def initialise_nodes(self):
        self.node_reference[self.input_nodes].initialise()

    def controller(self):

        return Controller(self._module_dict, self.network_map, self.entry_point_name(),
                          network_dir=self.network_directory)

    def compile_model(self):
        model_list = []
        network_map = []
        network_directory = {}
        for k, v in self.node_reference.items():
            network_directory[v.id_name] = {
                'in': [n.id_name for n in v.successors],
                'out': [n.id_name for n in v.predecessors]
            }
            if isinstance(v, InputNode):
                input_name, input_node = v.id_name, v.model
            else:
                model_list.append([v.id_name, v.model])

            network_map.append([v.id_name, v])
        model_list.append([input_name, input_node])
        model_list = reversed(model_list)
        self._module_dict.update(model_list)
        self.network_map.update(network_map)
        self.network_directory = network_directory

    def check_input_nodes(self):
        if not self.input_nodes:
            node = self.add_new_node(
                node_type='INPUT',
            )
            self.leaf_nodes.add(node.node_id)

    def add_missing_edges(self, max_iter=MAX_ITER):
        for node_id in self.leaf_nodes:
            k = 0
            existing_node = None
            valid = False

            while k < max_iter:
                existing_node = self.random_node_from_graph()

                b_1 = nx.has_path(self.graph, existing_node, node_id)
                b_2 = existing_node in nx.all_neighbors(self.graph, node_id)

                if isinstance(self.node_reference[existing_node], InputNode) and (not b_1) and (not b_2):
                    valid = True
                    break
                k += 1

            if valid:
                self.graph.add_edge(node_id, existing_node)
            else:
                self.add_new_node(node_type='INPUT', predecessor_id=node_id)
        return

    def contract_input_nodes(self):
        if isinstance(self.input_nodes, int):
            return

        if len(self.input_nodes) < 2:
            self.input_nodes = self.input_nodes[0]
            return

        first_input = self.input_nodes[0]
        contracted_graph = self.graph.copy()
        for node in self.input_nodes[1:]:
            del self.node_reference[node]
            contracted_graph = nx.contracted_nodes(contracted_graph, first_input, node, self_loops=False)

        self.graph = contracted_graph
        self.input_nodes = first_input

    def _pool_predecessors(self, node):
        return [x for x in self.graph.predecessors(node) if isinstance(self.node_reference[x], PoolNode)]

    def _prune_pool(self, query_node):
        connected_pool_nodes = self._pool_predecessors(query_node)
        break_flag = False
        if connected_pool_nodes:  # If there is any connected nodes, iterate over them
            for pool_node in connected_pool_nodes:
                # get a new conv type to replace the pool node
                self.node_reference[pool_node] = self.create_new_node(node_type='CONV', node_id=pool_node)

                # remove from list of pool nodes as it is now a conv node
                self.pool_nodes.remove(pool_node)

                break_flag = self.pool_count < self.max_pool
                if break_flag:
                    break

        return break_flag

    def prune_pool_nodes(self):
        break_flag = False
        # Prune from the input nodes first
        if isinstance(self.input_nodes, list):
            for input_node in self.input_nodes:
                break_flag = self._prune_pool(input_node)
                if break_flag:
                    break
        else:
            self._prune_pool(self.input_nodes)

    def update_nodes(self):
        for node in self.graph.nodes():
            self.node_reference[node].update_connectivity(self.graph, self.node_reference)


    def decompose_binary_nodes(self):
        for bin in self.binary_nodes:
            bin_node: BinaryNode = self.node_reference[bin]
            replacement_map = {}
            for suc, pre_proc_n in bin_node.preprocessors.items():
                new_id = next(self.next_node_id)
                replacement_map[new_id] = self.network_map[suc].node_id
                pre_proc_n.node_id = new_id #Doesnt get assigned on creation, so need to add here
                self.add_new_node(pre_proc_n, bin)

            for new, old in replacement_map.items():
                self.graph.remove_edge(bin, old)
                self.graph.add_edge(new, old)


    def reset(self, min_depth: int = None, max_depth: int = None, image_size: int = None,
              input_channels: int = None, min_nodes: int=None):
        if min_depth is None:
            min_depth = self.min_depth

        if max_depth is None:
            max_depth = self.max_depth

        if image_size is None:
            image_size = self.image_size

        if input_channels is None:
            input_channels = self.input_channels

        if min_nodes is None:
            min_nodes = self.min_nodes

        self.__init__(
            prediction_classes=self.prediction_classes,
            min_depth=min_depth,
            max_depth=max_depth,
            input_channels=input_channels,
            image_size=image_size,
            min_nodes=min_nodes
        )

    def show(self, labels='type'):
        if labels == 'both' or labels == 'type':
            relabel_mapping = {k: v.node_name() for k, v in self.node_reference.items()}
            computational_graph = nx.relabel_nodes(self.graph, relabel_mapping, )
            options = {'node_size': 2000, 'alpha': 0.7}
            if labels == 'both':
                plt.figure(figsize=(15, 8))
                plt.subplot(121)
                pos = nx.spring_layout(computational_graph, iterations=50)
                nx.draw(computational_graph, pos, with_labels=True, **options)

                plt.subplot(122)
                pos = nx.spiral_layout(self.graph, )
                nx.draw(self.graph, with_labels=True)

            else:
                plt.figure(figsize=(15, 8))
                plt.subplot()
                nx.draw(computational_graph, with_labels=True)

        else:
            plt.figure(figsize=(15, 8))
            plt.subplot()
            nx.draw(self.graph, with_labels=True)

        plt.show()


if __name__ == '__main__':
    image_size = (128, 128)
    rag = RandomArchitectureGenerator(prediction_classes=10, min_depth=3, max_depth=5, image_size=28, input_channels=1,
                                      min_nodes=3)
    cont = -1
    while cont == -1:
        cont = rag.get_architecture()

    rag.show(labels='both')

    # cont = rag.controller()

    in_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)
    out = cont(in_tensor)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cont.parameters(), lr=0.001, momentum=0.9)
    print(out)

    out

import numpy as np
from queue import Queue
from collections import defaultdict
import networkx as nx
from networkx import DiGraph
from typing import Union, List, Callable, Dict, Type, Tuple

from src.genotype.base_pairs import ConvNode, MaxPoolNode, AvgPoolNode, SumNode, ConcatNode, InputNode, Node, \
    BinaryNode, PoolNode

# from .base_pairs import ConvNode, MaxPoolNode, AvgPoolNode, SumNode, ConcatNode, InputNode, Node, \
#     BinaryNode, PoolNode

# import matplotlib
import matplotlib.pyplot as plt


class RandomArchitectureGenerator:
    MAX_DEPTH = 100
    MIN_DEPTH = 5
    MAX_ITER = 100
    NODE_TYPES = {'BINARY', 'CONV', 'POOL', 'INPUT', 'REFERENCE'}
    KERNEL_CLASSES = {'CONV', 'MAX', 'AVERAGE'}
    PADDING_MODE = 'replicate'
    KERNEL_CHOICES = (3, 3, 5, 7)
    IMAGE_CHANNELS = 3
    IMAGE_SIZE = 128  # 128x128 assuming square
    # CHANNEL_CHOICES = (32, 64, 128, 256, 512)
    # KERNEL_CHOICES = (3, 3, 5, 7)  # 3x3 kernel 50% likely, 25% for 5x5 and 7x7 respectively.
    DEFAULT_IMAGE_SIZE = 128
    TYPE_MAP = dict(
        CONV=dict(
            TYPE='CONV',
            ARITY=1,
        ),
        INPUT=dict(
            TYPE='INPUT',
            ARITY=0
        ),
        MAX=dict(
            TYPE='MAX',
            ARITY=1
        ),
        AVERAGE=dict(
            TYPE='AVERAGE',
            ARITY=1
        ),
        SUM=dict(
            TYPE='SUM',
            ARITY=2
        ),
        CONCAT=dict(
            TYPE='CONCAT',
            ARITY=2
        ),
    )
    POOL_TYPES = {'MAX', 'AVERAGE'}
    BINARY_TYPES = {'SUM', 'CONCAT'}

    def __init__(self, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, image_size: Union[int, tuple, list]=DEFAULT_IMAGE_SIZE, input_channels: int=IMAGE_CHANNELS):
        self.min_depth = min_depth
        self.max_depth = max_depth
        if isinstance(image_size, int):
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_height = image_size[0]
            self.image_width = image_size[1]
        self.image_size = image_size
        self.input_channels = input_channels

        self.target_depth = np.random.randint(self.min_depth, self.max_depth)

        self.level = 0
        self.nonleaf = False
        self.node_count = 1
        self.root_id = 0

        self.queue = Queue()
        self.queue.put((self.root_id, self.level))

        self.leaf_nodes = set()

        self.max_pool = np.floor(np.log2(self.image_width))  # as per paper
        self.pool_nodes = []

        self.input_nodes = []  # will implement a consolidation of input nodes to a single node

        self.graph = DiGraph()
        self.graph.add_node(self.root_id)

        self.node_reference: Dict[int, Node] = {}

        self.node_reference: Dict[int, Node] = {
            self.root_id: self.random_new_node(self.root_id, {'REFERENCE', 'INPUT', })
        }




    @property
    def pool_count(self):
        return len(self.pool_nodes)

    def random_node_from_graph(self) -> int:
        num_nodes = self.graph.number_of_nodes()
        if num_nodes == 1:
            return 1
        else:
            # Random node from 1 to num nodes, so that the output node isnt selected
            return np.random.choice(range(1, num_nodes))

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

        return restricted_types

    def create_new_node(self, node_type, node_id):
        if node_type == 'SUM':
            new_node = SumNode(node_id)
        elif node_type == 'CONCAT':
            new_node = ConcatNode(node_id)
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

        return new_node

    def random_new_node(self, node_id: int, restricted_types: set) -> Node:
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
            new_node = self.create_new_node(new_type, node_id)
            new_node.random_initialisation()

        return new_node

    def add_new_node(self, node_id: int, node_type: Union[str, Node], predecessor_id: int = None):
        self.graph.add_node(node_id)

        if isinstance(node_type, Node):
            node = node_type
        else:
            node = self.create_new_node(node_type=node_type, node_id=node_id)

        # self._increment_counters(node_id, node)

        if predecessor_id is not None:
            self.graph.add_edge(predecessor_id, node_id)

        self.node_reference[node_id] = node

    def check_input_nodes(self):
        if not self.input_nodes:
            self.add_new_node(
                node_id=self.node_count,
                node_type='INPUT',
            )
            self.leaf_nodes.add(self.node_count)
            self.node_count += 1

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
                new_node_id = self.graph.number_of_nodes()
                self.add_new_node(new_node_id, node_type='INPUT', predecessor_id=node_id)
        return

    def get_architecture(self, reset_on_finish=True) -> Tuple[DiGraph, Dict[int, Node]]:

        while not self.queue.empty():
            node_id, current_level = self.queue.get()
            arity = self.node_reference[node_id].arity

            if current_level != self.level:
                self.nonleaf = False
                self.level = current_level
            i = 0
            while i < arity:
                if current_level + 1 == self.target_depth:
                    self.add_new_node(self.node_count, node_type='INPUT', predecessor_id=node_id)
                    self.node_count += 1
                else:
                    restricted_node_types = self.disallowed_types(node_id, current_level)
                    new_node = self.random_new_node(node_id=self.node_count, restricted_types=restricted_node_types)
                    if new_node is None:
                        self.leaf_nodes.add(node_id)
                    else:
                        self.add_new_node(self.node_count, node_type=new_node, predecessor_id=node_id)

                        if not isinstance(new_node, InputNode):
                            self.queue.put((self.node_count, current_level + 1))
                            self.nonleaf = True

                        self.node_count += 1

                i += 1

        print(f'Final depth:{self.level}')
        print(f'Number of nodes:{self.node_count}')
        if self.graph.number_of_nodes() < 3:
            self.reset()
            return None, None

        self.check_input_nodes()
        self.add_missing_edges()
        self.prune_pool_nodes()
        self.contract_input_nodes()
        self.update_nodes()

        retval = (self.graph, self.node_reference)

        if reset_on_finish:
            self.reset()

        return retval

    def _pool_predecessors(self, node):
        return [x for x in self.graph.predecessors(node) if isinstance(self.node_reference[x], PoolNode)]

    def prune_pool_nodes(self):
        break_flag = False
        # Prune from the input nodes first
        for input_node in self.input_nodes:
            connected_pool_nodes = self._pool_predecessors(input_node)
            if break_flag:
                break
            if connected_pool_nodes:  # If there is any connected nodes, iterate over them
                for pool_node in connected_pool_nodes:
                    # get a new conv type to replace the pool node
                    self.node_reference[pool_node] = self.create_new_node(node_type='CONV', node_id=pool_node)

                    # remove from list of pool nodes as it is now a conv node
                    self.pool_nodes.remove(pool_node)

                    break_flag = self.pool_count < self.max_pool

                    if break_flag:
                        break

    def update_nodes(self):
        for node in self.graph.nodes():
            self.node_reference[node].update_connectivity(self.graph, self.node_reference)

    def reset(self, min_depth: int = None, max_depth: int = None):
        if min_depth is None:
            min_depth = self.min_depth

        if max_depth is None:
            max_depth = self.max_depth

        self.__init__(min_depth=min_depth, max_depth=max_depth)

    @staticmethod
    def show(graph, node_reference=None, labels='type'):
        if labels == 'both' or labels == 'type':
            assert node_reference is not None, 'labels=[\'both\' \'type\']  requires a non NoneType attribute map'

            relabel_mapping = {k:v.node_name() for k, v in node_reference.items()}
            computational_graph = nx.relabel_nodes(graph, relabel_mapping, )
            options = {'node_size': 2000, 'alpha': 0.7}
            if labels == 'both':
                plt.figure(figsize=(15, 8))
                plt.subplot(121)
                pos = nx.spring_layout(computational_graph, iterations=50)
                nx.draw(computational_graph, pos, with_labels=True, **options)

                plt.subplot(122)
                pos = nx.spiral_layout(graph, )
                nx.draw(graph, with_labels=True)

            else:
                plt.figure(figsize=(15, 8))
                plt.subplot()
                nx.draw(computational_graph, with_labels=True)

        else:
            plt.figure(figsize=(15, 8))
            plt.subplot()
            nx.draw(graph, with_labels=True)

    def contract_input_nodes(self):
        if len(self.input_nodes) < 2:
            self.input_nodes = self.input_nodes[0]
            return

        first_input = self.input_nodes[0]
        contracted_graph = self.graph.copy()
        # contracted_reference = self.node_reference.copy()
        for node in self.input_nodes[1:]:
            # Wrote this block to update the node_reference in case of graph re-numbering.
            # del self.node_reference[node] #removes redundant node from dict
            # del contracted_reference[node]
            # temp = contracted_reference.copy()
            # for k in range(node+1, len(self.node_reference)):
            #     if k > node:
            #         temp[k-1] = temp.pop(k)
            # contracted_reference = temp

            contracted_graph = nx.contracted_nodes(contracted_graph, first_input, node, self_loops=False)

        self.graph = contracted_graph
        # self.node_reference = contracted_reference
        self.input_nodes = first_input

if __name__ == '__main__':
    image_size = (128, 128)
    rag = RandomArchitectureGenerator(min_depth=10, max_depth=75, input_channels=3, image_size=image_size)
    g, a = rag.get_architecture()

    rag.show(g.reverse(), a, labels='both')


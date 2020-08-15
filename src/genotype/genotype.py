import numpy as np
from queue import Queue
from collections import defaultdict
import networkx as nx
from networkx import DiGraph
from typing import Union, Tuple

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
    CHANNEL_CHOICES = (32, 64, 128, 256, 512)
    KERNEL_CHOICES = (3, 3, 5, 7)  # 3x3 kernel 50% likely, 25% for 5x5 and 7x7 respectively.
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

    def __init__(self, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, image_size=DEFAULT_IMAGE_SIZE):
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.target_depth = np.random.randint(self.min_depth, self.max_depth)

        self.level = 0
        self.nonleaf = False
        self.node_count = 1

        self.queue = Queue()
        self.queue.put((0, self.level))

        self.graph = DiGraph()
        self.graph.add_node(0)

        self.attribute_map = defaultdict(dict)
        self.attribute_map[0] = self.random_new_node({'REFERENCE', 'INPUT', })

        self.leaf_nodes = set()

        self.max_pool = np.floor(np.log2(image_size))
        self.pool_nodes = []

        self.input_nodes = []

    @property
    def pool_count(self):
        return len(self.pool_nodes)

    @staticmethod
    def _type_map(node_type):
        return_type = RandomArchitectureGenerator.TYPE_MAP[node_type].copy()

        if node_type in RandomArchitectureGenerator.KERNEL_CLASSES:
            return_type['KERNEL'] = np.random.choice(RandomArchitectureGenerator.KERNEL_CHOICES).item()
            return_type['PADDING'] = np.random.randint(0, int(return_type['KERNEL'] - 1 / 2))
            return_type['PADDING_MODE'] = RandomArchitectureGenerator.PADDING_MODE

        if node_type == 'CONV':
            return_type['OUT_CHANNELS'] = np.random.choice(RandomArchitectureGenerator.CHANNEL_CHOICES).item()

        return return_type

    def random_node_from_graph(self) -> int:
        num_nodes = self.graph.number_of_nodes()
        if num_nodes == 1:
            return 1
        else:
            return np.random.choice(range(1, num_nodes))

    def connected_to_input(self, node) -> bool:
        connections = self.graph.neighbors(node)
        for neighbor in connections:
            if self.attribute_map[neighbor]['TYPE'] == 'INPUT':
                return True
        return False

    def disallowed_types(self, node, level) -> set:
        restricted_types = set()
        connected_to_input = self.connected_to_input(node)

        if connected_to_input or not self.nonleaf:
            restricted_types.add('INPUT')

        if (level + 2 == self.target_depth) or self.attribute_map[node]['TYPE'] in self.BINARY_TYPES:
            restricted_types.add('BINARY')

        return restricted_types

    def random_new_node(self, restricted_types: set) -> Union[dict]:
        valid_types: Tuple[str, ...] = tuple(self.NODE_TYPES - restricted_types)

        new_type = np.random.choice(valid_types, size=1).item()

        if new_type == 'BINARY':
            q = np.random.uniform()
            if q < 0.5:
                new_node = self._type_map('SUM')
            else:
                new_node = self._type_map('CONCAT')

        elif new_type == 'CONV':
            new_node = self._type_map('CONV')

        elif new_type == 'POOL':
            q = np.random.uniform()
            if q < 0.5:
                new_node = self._type_map('MAX')
            else:
                new_node = self._type_map('AVERAGE')

        elif new_type == 'INPUT':
            new_node = self._type_map('INPUT')

        else:
            new_node = None
        return new_node

    def _increment_counters(self, node, attributes):
        node_type = attributes['TYPE']
        if node_type == 'INPUT':
            self.input_nodes.append(node)

        elif node_type in {'MAX', 'AVERAGE'}:
            self.pool_nodes.append(node)

    def _update_inputs(self, node_attributes, predecessor):
        if node_attributes['ARITY'] > 2:
            if 'IN_CHANNELS' in node_attributes:
                node_attributes['IN_CHANNELS'].append(self.attribute_map[predecessor]['OUT_CHANNELS'])
            else:
                node_attributes['IN_CHANNELS'] = [self.attribute_map[predecessor]['OUT_CHANNELS']]
        else:
            node_attributes['IN_CHANNELS'] = self.attribute_map[predecessor]['OUT_CHANNELS']
        return node_attributes

    def add_new_node(self, new_node, node_type: Union[str, dict], predecessor=None):
        self.graph.add_node(new_node)

        if type(node_type) == dict:
            node_attributes = node_type
        else:
            node_attributes = self._type_map(node_type)

        self._increment_counters(new_node, node_attributes)

        if predecessor is not None:
            self.graph.add_edge(predecessor, new_node)
            # node_attributes = self._update_inputs(node_attributes, predecessor)

        self.attribute_map[new_node] = node_attributes

    def add_missing_edges(self, max_iter=MAX_ITER):
        for node in self.leaf_nodes:
            k = 0
            existing_node = None
            valid = False

            while k < max_iter:
                existing_node = self.random_node_from_graph()

                b_1 = nx.has_path(self.graph, existing_node, node)
                b_2 = existing_node in nx.all_neighbors(self.graph, node)

                if (self.attribute_map[existing_node]['TYPE'] != 'INPUT') and (not b_1) and (not b_2):
                    valid = True
                    break
                k += 1

            if valid:
                self.graph.add_edge(node, existing_node)
                # self.attribute_map[existing_node] = self._update_inputs(self.attribute_map[existing_node], node)
            else:
                new_node = self.graph.number_of_nodes()
                self.add_new_node(new_node, node_type='INPUT', predecessor=node)
        return

    def get_architecture(self, reset_on_finish=True) -> Tuple[DiGraph, defaultdict]:

        while not self.queue.empty():
            node, current_level = self.queue.get()
            arity = self.attribute_map[node]['ARITY']

            if current_level != self.level:
                self.nonleaf = False
                self.level = current_level
            i = 0
            while i < arity:
                if current_level + 1 == self.target_depth:
                    self.add_new_node(self.node_count, node_type='INPUT', predecessor=node)
                    self.node_count += 1
                else:
                    restricted_node_types = self.disallowed_types(node, current_level)
                    new_node = self.random_new_node(restricted_node_types)
                    if new_node is None:
                        self.leaf_nodes.add(node)
                    else:
                        self.add_new_node(self.node_count, node_type=new_node, predecessor=node)

                        if new_node['TYPE'] != 'INPUT':
                            self.queue.put((self.node_count, current_level + 1))
                            self.nonleaf = True

                        self.node_count += 1

                i += 1

        print(f'Final depth:{self.level}')
        print(f'Number of nodes:{self.node_count}')
        if self.graph.number_of_nodes() < 3:
            self.reset()
            return None, None

        self.add_missing_edges()
        self.prune_pool_nodes()

        retval = (self.graph, self.attribute_map)

        if reset_on_finish:
            self.reset()

        return retval

    def _pool_predecessors(self, node):
        return [x for x in self.graph.predecessors(node) if self.attribute_map[x]['TYPE'] in self.POOL_TYPES]

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
                    self.attribute_map[pool_node] = self._type_map('CONV')

                    # remove from list of pool nodes as it is now a conv node
                    self.pool_nodes.remove(pool_node)

                    break_flag = self.pool_count < self.max_pool

                    if break_flag:
                        break

    def reset(self, min_depth: int = None, max_depth: int = None):
        if min_depth is None:
            min_depth = self.min_depth

        if max_depth is None:
            max_depth = self.max_depth

        self.__init__(min_depth=min_depth, max_depth=max_depth)

    @staticmethod
    def show(graph, attribute_map=None, labels='type'):
        if labels == 'both' or labels == 'type':
            assert attribute_map is not None, 'labels=[\'both\' \'type\']  requires a non NoneType attribute map'

            relabel_mapping = {k: f'{v["TYPE"]}:{k}' for k, v in attribute_map.items()}
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


if __name__ == '__main__':
    rag = RandomArchitectureGenerator(min_depth=10, max_depth=75)
    rag.get_architecture()

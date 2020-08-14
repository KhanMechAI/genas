import numpy as np
from queue import Queue
from collections import defaultdict
import networkx as nx
from networkx import DiGraph
from typing import Union, List

# import matplotlib
import matplotlib.pyplot as plt


class RandomArchitectureGenerator():
    MAX_DEPTH = 100
    MIN_DEPTH = 5
    MAX_ITER = 100
    NODE_TYPES = set(('BINARY', 'CONV', 'POOL', 'INPUT', 'REFERENCE'))
    FEATURE_MAPS = (32, 64, 128, 256, 512)

    def __init__(self, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH):
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
        self.attribute_map[0] = random_node({'REFERENCE', 'INPUT', })

        self.leaf_nodes = set()

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
        connected_to_input = self.check_connected_to_input(node, )

        if connected_to_input or not self.nonleaf:
            restricted_types.add('INPUT')

        if (level + 2 == self.target_depth) or self.attribute_map[node]['TYPE'] in {'SUM', 'CONCAT'}:
            restricted_types.add('BINARY')

        return restricted_types

    def random_new_node(self, restricted_types: set) -> Union[dict]:
        valid_types = tuple(self.NODE_TYPES - restricted_types)

        new_type = np.random.choice(valid_types, size=1).item()

        new_node = dict()

        if new_type == 'BINARY':
            q = np.random.uniform()
            if q < 0.5:
                new_node['TYPE'] = 'SUM'
            else:
                new_node['TYPE'] = 'CONCAT'

            new_node['ARITY'] = 2

        elif new_type == 'CONV':
            new_node['TYPE'] = 'CONV'
            new_node['FEATURE_MAP'] = np.random.choice(self.FEATURE_MAPS)
            new_node['ARITY'] = 1

        elif new_type == 'POOL':
            q = np.random.uniform()
            if q < 0.5:
                new_node['TYPE'] = 'MAX'
            else:
                new_node['TYPE'] = 'AVERAGE'

            new_node['ARITY'] = 1

        elif new_type == 'INPUT':
            new_node['TYPE'] = 'INPUT'
            new_node['ARITY'] = 0

        else:
            new_node = None
        return new_node

    def add_missing_edges(self, max_iter=MAX_ITER):
        for node in self.leaf_nodes:
            k = 0
            existing_node = None
            valid = False

            while k < max_iter:
                existing_node = get_random_node(self.graph)

                b_1 = nx.has_path(self.graph, existing_node, node)
                b_2 = existing_node in nx.all_neighbors(self.graph, node)

                if (self.attribute_map[existing_node]['TYPE'] != 'INPUT') and (not b_1) and (not b_2):
                    valid = True
                    break
                k += 1

            if valid:
                self.graph.add_edge(node, existing_node)
            else:
                new_node = self.graph.number_of_nodes()
                self.graph.add_node(new_node)
                self.graph.add_edge(node, new_node)
                self.attribute_map[new_node]['TYPE'] = 'INPUT'
                self.attribute_map[new_node]['ARITY'] = 0

        return

    def get_architecture(self):
        # target_depth = np.random.randint(self.min_depth, self.max_depth)
        while not self.queue.empty():
            node, l = self.queue.get()
            arity = self.attribute_map[node]['ARITY']

            if l != self.level:
                self.nonleaf = False
                self.level = l
            i = 0
            while i < arity:
                if l + 1 == self.target_depth:
                    self.attribute_map[self.node_count] = dict(
                        TYPE='INPUT',
                        ARITY=0
                    )
                    self.graph.add_node(self.node_count)
                    self.graph.add_edge(node, self.node_count, )
                    self.node_count += 1
                else:
                    restricted_node_types = disallowed_nodes(node, l, )
                    new_node = random_node(restricted_node_types)
                    if new_node is None:
                        self.leaf_nodes.add(node)
                    else:
                        self.attribute_map[self.node_count] = new_node
                        self.graph.add_node(self.node_count)
                        self.graph.add_edge(node, self.node_count, )

                        if new_node['TYPE'] != 'INPUT':
                            self.queue.put((self.node_count, l + 1))
                            self.nonleaf = True

                        self.node_count += 1

                i += 1

        print(f'Final depth:{self.level}')
        print(f'Number of nodes:{self.node_count}')
        if self.graph.number_of_nodes() < 3:
            return None, None

        self.add_missing_edges()
        return self.graph, self.attribute_map

    def reset(self, min_depth: int = None, max_depth: int = None):
        if min_depth is None:
            min_depth = self.min_depth

        if max_depth is None:
            max_depth = self.max_depth

        self.__init__(min_depth=min_depth, max_depth=max_depth)


def check_connected_to_input(node, graph, attribute_map):
    connections = graph.neighbors(node)
    for neighbor in connections:
        if attribute_map[neighbor]['TYPE'] == 'INPUT':
            return True
    return False


# Algorithm 2
def disallowed_nodes(node, level, depth, nonleaf, attribute_map: dict, graph: DiGraph) -> set:
    restricted_types = set()
    connected_to_input = check_connected_to_input(node, graph, attribute_map)

    if connected_to_input or not nonleaf:
        restricted_types.add('INPUT')

    if (level + 2 == depth) or attribute_map[node]['TYPE'] in {'SUM', 'CONCAT'}:
        restricted_types.add('BINARY')

    return restricted_types


# redundant
def get_new_node():
    return defaultdict(
        TYPE=None,
        ARITY=None
    )


# Potentially use this for the graph https://networkx.github.io/documentation/stable/tutorial.html

# Done
# Algorithm 4
def random_node(restricted_nodes: set) -> dict:
    feature_map_choices = (32, 64, 128, 256, 512)
    all_types = set(('BINARY', 'CONV', 'POOL', 'INPUT', 'REFERENCE'))
    valid_types = all_types.difference(restricted_nodes)
    random_node_type = np.random.choice(tuple(valid_types), size=1).item()
    new_node = get_new_node()

    if random_node_type == 'BINARY':
        q = np.random.uniform()
        if q < 0.5:
            new_node['TYPE'] = 'SUM'
        else:
            new_node['TYPE'] = 'CONCAT'

        new_node['ARITY'] = 2

    elif random_node_type == 'CONV':
        new_node['TYPE'] = 'CONV'
        new_node['FEATURE_MAP'] = np.random.choice(feature_map_choices)
        new_node['ARITY'] = 1

    elif random_node_type == 'POOL':
        q = np.random.uniform()
        if q < 0.5:
            new_node['TYPE'] = 'MAX'
        else:
            new_node['TYPE'] = 'AVERAGE'

        new_node['ARITY'] = 1

    elif random_node_type == 'INPUT':
        new_node['TYPE'] = 'INPUT'
        new_node['ARITY'] = 0

    else:
        new_node = None
    return new_node


# Done
def get_random_node(graph):
    num_nodes = graph.number_of_nodes()
    if num_nodes == 1:
        return 1
    else:
        return np.random.choice(range(1, num_nodes))


# Algorithm 3
def add_missing_edges(S, graph, attribute_map, max_iter=100):
    for node in S:
        k = 0
        exisiting_node = None
        valid = False

        while k < max_iter:
            exisiting_node = get_random_node(graph)

            b_1 = nx.has_path(graph, exisiting_node, node)
            b_2 = exisiting_node in nx.all_neighbors(graph, node)

            if (attribute_map[exisiting_node]['TYPE'] != 'INPUT') and (not b_1) and (not b_2):
                valid = True
                break
            k += 1

        if valid:
            graph.add_edge(node, exisiting_node)
        else:
            new_node = graph.number_of_nodes()
            graph.add_node(new_node)
            graph.add_edge(node, new_node)
            attribute_map[new_node]['TYPE'] = 'INPUT'
            attribute_map[new_node]['ARITY'] = 0

    return graph, attribute_map


# Algorithm 1
def initialise_archs(min_depth=3, max_depth=100):
    init_q = Queue()
    depth = np.random.randint(min_depth, max_depth)
    level = 0
    root = 0
    nonleaf = False

    graph = DiGraph()
    graph.add_node(root)

    init_q.put((0, level))  # (node number, level). Root node = 0

    attribute_map = defaultdict(dict)
    attribute_map[root] = random_node({'REFERENCE', 'INPUT'})

    node_counter = 1
    S = set()

    print(f'Target depth: {depth}')
    while not init_q.empty():
        node, l = init_q.get()
        arity = attribute_map[node]['ARITY']

        if l != level:
            nonleaf = False
            level = l
        i = 0
        while i < arity:
            if l + 1 == depth:
                attribute_map[node_counter] = dict(
                    TYPE='INPUT',
                    ARITY=0
                )
                graph.add_node(
                    node_counter,
                )
                graph.add_edge(node, node_counter, )
                node_counter += 1
            else:
                restricted_node_types = disallowed_nodes(node, l, depth, nonleaf, attribute_map, graph)
                new_node = random_node(restricted_node_types)
                if new_node is None:
                    S.add(node)
                    # graph.add_node(node_counter)
                    # attribute_map[node_counter] = new_node
                else:
                    attribute_map[node_counter] = new_node
                    graph.add_node(node_counter)
                    graph.add_edge(node, node_counter, )

                    if new_node['TYPE'] != 'INPUT':
                        init_q.put((node_counter, l + 1))
                        nonleaf = True

                    node_counter += 1

            i += 1
    print(f'Final depth:{level}')
    print(f'Number of nodes:{node_counter}')
    if graph.number_of_nodes() < 3:
        return None, None

    graph, attribute_map = add_missing_edges(S, graph, attribute_map, max_iter=100)
    return graph, attribute_map


graph, attribute_map = initialise_archs(min_depth=10, max_depth=100)

if graph is None:
    print('Degenerate graph')
    exit()

relabel_mapping = {k: f'{v["TYPE"]}:{k}' for k, v in attribute_map.items()}
computational_graph = nx.relabel_nodes(graph, relabel_mapping, )
try:
    cycles = nx.find_cycle(
        computational_graph,
        source=0,
        orientation='original'
    )
    print(cycles)
except:
    print('no cycles')

print(nx.to_dict_of_lists(graph))

plt.subplot(121)
nx.draw(computational_graph, with_labels=True)

plt.subplot(122)
nx.draw(graph, with_labels=True)

plt.show()

print(graph)

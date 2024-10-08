import time
import numpy as np
import networkx as nx

from scipy.stats import rv_histogram

class Friction():

    def __init__(self, values, bins):

        self.histogram = rv_histogram((values, bins))

    def __call__(self, adjacency):

        return self.histogram.pdf(adjacency)

class Gravity():

    default_mass_function = lambda self, node: node.get('population', 0)
    default_friction_function = lambda self, adj: np.ones_like(adj)

    def __init__(self, graph, **kwargs):

        # self.graph = graph
        self.mass = kwargs.get('mass', self.default_mass_function)
        self.friction = kwargs.get('friction', self.default_friction_function)

        self.nodes = kwargs.get('nodes', list(graph.nodes))

        self.build(graph)

    def build(self, graph, **kwargs):

        self._node = graph._node # Pointer to _node structure
        self._adj = graph._adj # Pointer to _adj structure

        self.ratios = {s: {t: 0 for t in graph.nodes} for s in graph.nodes}

        self.denominator = 0

        total_gravity = sum([self.mass(node) for node in self._node.values()])

        adjacency = nx.to_numpy_array(graph, weight = kwargs.get('weight', 'distance'))

        friction = self.friction(adjacency)

        for idx_s, source in enumerate(self.nodes):

            source_node = self._node[source]

            source_portion = self.mass(source_node) / total_gravity

            denominator = 0

            for idx_t, target in enumerate(self.nodes):

                if source != target:

                    target_node = self._node[target]

                    denominator += self.mass(target_node) * friction[idx_s, idx_t]

            for idx_t, target in enumerate(self.nodes):

                if source != target:
                    
                    target_node = self._node[target]

                    self.ratios[source][target] = (
                        source_portion * 
                        self.mass(target_node) *
                        friction[idx_s, idx_t] /
                        denominator
                        ) 

                    self.denominator += self.ratios[source][target]

    def __call__(self, total):

        return self.volumes(total)

    def volumes(self, total):

        volumes = {s: {t: 0 for t in self.ratios.keys()} for s in self.ratios.keys()}

        for source in self.ratios.keys():
            for target in self.ratios.keys():

                volumes[source][target] = self.ratios[source][target] * total

        return volumes

    def flows(self, total, duration):

        flows = {s: {t: 0 for t in self.ratios.keys()} for s in self.ratios.keys()}

        for source in self.ratios.keys():
            for target in self.ratios.keys():

                flows[source][target] = self.ratios[source][target] * total / duration

        return flows

    def to_array(self, field):

        data = getattr(self, field)

        return np.array([list(v.values()) for v in data.values()])
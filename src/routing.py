import time
import numpy as np
import networkx as nx

from scipy.special import factorial
from scipy.stats import rv_histogram
from scipy.optimize import minimize, Bounds
from scipy.interpolate import RegularGridInterpolator

from heapq import heappop, heappush
from itertools import count

from .progress_bar import ProgressBar
from .floyd_warshall import shortest_paths
from .queue import Queue

def traffic(graph, paths, values = None):

    _node = graph._node
    _adj = graph._adj

    n = graph.number_of_nodes()

    if values is None:

        values = np.ones((n, n))

    traffic = {n: {n: 0 for n in graph.nodes} for n in graph.nodes}

    for source in paths.keys():
        for target in paths[source].keys():

            value = values[source][target]
            path = paths[source][target]

            traffic[source][source] += value

            for idx in range(1, len(path)):

                traffic[path[idx]][path[idx]] += value

                traffic[path[idx - 1]][path[idx]] += value

    return traffic

def shortest_path_graph(graph, fields, **kwargs):

    return_paths = kwargs.get('return_paths', False)

    _, values, all_paths = shortest_paths(graph, fields = fields, **kwargs)
    
    all_pairs = nx.DiGraph()

    all_pairs.add_nodes_from([(k, v) for k, v in graph._node.items()])

    edges = []

    for source, paths in all_paths.items():
        for target, path in paths.items():

            data = {**values[source][target]}

            if return_paths:

                data['path'] = all_paths[source][target]

            edges.append((source, target, data))

    all_pairs.add_edges_from(edges)

    return all_pairs
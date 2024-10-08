'''
Module for computing adjacency for a graph via routing on another graph. An example
would be computing travel times between cities connected by highways or latency between
computers connected via the internet. Another case would be compting network distances
between all points in a subset of a greater network. In any case, the nodes of the former
network must be coincident, or nearly so, with nodes in the latter network.

In this module the graph for which adjacency is being computed will be referred to as the
"graph" while the graph on which the routing occurs will be referred to as the "atlas". In
cases where either could be used "graph" will be used as default.
'''

import numpy as np

from sys import maxsize

from scipy.spatial import KDTree
from itertools import count
from heapq import heappop, heappush

from .progress_bar import ProgressBar

from .dijkstra import dijkstra
from .graph import graph_from_nlg, cypher
from .utilities import pythagorean, haversine

def direct_adjacency(graph, **kwargs):

    limits = kwargs.get('limits', (0, np.inf))
    distance_function = kwargs.get('distance_function', haversine)

    n = graph.number_of_nodes()

    x = np.array([v['x'] for v in graph._node.values()])
    y = np.array([v['y'] for v in graph._node.values()])
    nodes = np.array(list(graph.nodes))

    encoder, decoder = cypher(graph)

    for idx_s in ProgressBar(range(n), **kwargs.get('progress_bar', {})):

        distances = haversine(x[idx_s], y[idx_s], x, y)

        indices = np.argwhere(
            (distances >= limits[0]) & (distances <= limits[1])
            ).flatten()

        # print(x, limits, (x >= limits[0]) & (x <= limits[1]))

        # print(indices)

        edges = []

        for idx_t in indices:

            edges.append((nodes[idx_s], nodes[idx_t], {'distance': distances[idx_t]}))

        graph.add_edges_from(edges)

        # print(edges)
        # break

    return graph

# Routing functions and related objects
def closest_nodes_from_coordinates(graph, x, y):
    '''
    Creates an assignment dictionary mapping between points and closest nodes
    '''

    nodes = list(graph.nodes)

    # Pulling coordinates from graph
    xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
    xy_graph = xy_graph.reshape((-1,2))

    # Creating spatial KDTree for assignment
    kd_tree = KDTree(xy_graph)

    # Shaping input coordinates
    xy_query = np.vstack((x, y)).T

    # Computing assignment
    result = kd_tree.query(xy_query)

    node_assignment = []

    for idx in range(len(x)):

        distance = result[0][idx]
        node = result[1][idx]

        node_assignment.append({
            'id': nodes[node],
            'query': xy_query[idx],
            'result': xy_graph[node],
            'distance': distance,
            })

    return node_assignment

def relate(atlas, graph):
    '''
    Creates an assignment dictionary mapping between points and closest nodes
    '''

    # Pulling coordinates from atlas
    xy_atlas = np.array([(n['x'], n['y']) for n in atlas._node.values()])
    xy_atlas = xy_atlas.reshape((-1,2))

    # Pulling coordinates from graph
    xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
    xy_graph = xy_graph.reshape((-1,2))

    # Creating spatial KDTree for assignment
    kd_tree = KDTree(xy_atlas)

    # Computing assignment
    result = kd_tree.query(xy_graph)

    node_assignment = []

    for idx in range(len(xy_graph)):

        distance = result[0][idx]
        node = result[1][idx]
        

        node_assignment.append({
            'id':node,
            'query':xy_graph[idx],
            'result':xy_atlas[node],
            'distance': haversine(*xy_graph[idx], *xy_atlas[node]),
            })

    return node_assignment

def node_assignment(atlas, graph):

    x, y = np.array(
        [[val['x'], val['y']] for key, val in graph._node.items()]
        ).T

    graph_nodes = np.array(
        [key for key, val in graph._node.items()]
        ).T

    atlas_nodes = closest_nodes_from_coordinates(atlas, x, y)

    graph_to_atlas = (
        {graph_nodes[idx]: atlas_nodes[idx]['id'] for idx in range(len(graph_nodes))}
        )
    
    atlas_to_graph = {}

    for key, val in graph_to_atlas.items():

        if val in atlas_to_graph.keys():

            atlas_to_graph[val] += [key]

        else:

            atlas_to_graph[val] = [key]

    return graph_to_atlas, atlas_to_graph

class Graph_From_Atlas():

    def __init__(self, **kwargs):

        self.fields = kwargs.get('fields', ['time', 'distance', 'price'])
        self.weights = kwargs.get('weights', [1, 0, 0])
        self.limits = kwargs.get('limits', [np.inf, np.inf, np.inf])
        self.n = len(self.fields)

    def initial(self):

        return {field: 0 for field in self.fields}

    def infinity(self):

        return {self.fields[idx]: np.inf for idx in range(self.n)}

    def update(self, values, link):

        feasible = True

        values_new = {}

        for idx in range(self.n):

            values_new[self.fields[idx]] = (
                values[self.fields[idx]] + link.get(self.fields[idx], 0)
                )

            feasible *= values_new[self.fields[idx]] <= self.limits[idx]

        # print('bbb', values_new)

        return values_new, feasible

    def compare(self, values, comparison):

        cost_new = 0
        cost_current = 0

        for idx in range(self.n):

            # print(values, self.fields[idx])

            cost_new += values[self.fields[idx]] * self.weights[idx]
            cost_current += comparison[self.fields[idx]] * self.weights[idx]

        savings = (cost_new < cost_current) or np.isnan(cost_current)

        return cost_new, savings

def reduction(atlas, **kwargs):

    origins = kwargs.get('origins', None)
    objective = kwargs.get('objective', 'distance')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_depth = kwargs.get('maximum_depth', np.inf)
    fields = kwargs.get('fields', ['distance', 'time'])
    snowball = kwargs.get('snowball', False)
    verbose = kwargs.get('verbose', False)

    if origins is None:

        intersections = []

        for source, adj in atlas._adj.items():

            if len(adj) != 2:

                intersections.append(source)

        origins = intersections

    heap = []
    c = count()

    for node in origins:

        heappush(heap, (next(c), node))

    _node = atlas._node

    nodes = []
    links = []

    while heap:

        idx, origin = heappop(heap)

        if verbose:
            print(f'{idx} done, {len(heap)} in queue                 ', end = '\r')

        node = _node[origin]
        node['id'] = origin

        nodes.append(node)

        costs, values, terminal = dijkstra(
            atlas,
            [origin],
            destinations = origins,
            objective = objective,
            maximum_cost = maximum_cost,
            maximum_depth = maximum_depth,
            fields = fields,
            terminate_at_destinations = True,
            return_paths = False,
            )

        terminal_nodes = [k for k, v in terminal.items() if v]

        destinations_reached = np.intersect1d(
            terminal_nodes,
            origins,
            )

        for destination in destinations_reached:

            link = {**values.get(destination, {})}

            link['source'] = origin
            link['target'] = destination

            links.append(link)

        if snowball:

            new_destinations = np.setdiff1d(
                terminal_nodes,
                origins,
                )

            for destination in new_destinations:

                heappush(heap, (next(c), destination))

                origins.append(destination)

    return graph_from_nlg({'nodes': nodes, 'links': links})

def adjacency(atlas, graph, **kwargs):
    '''
    Adds adjacency to graph by routing on atlas
    '''
    objective = kwargs.get('objective', 'distance')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_depth = kwargs.get('maximum_depth', np.inf)
    fields = kwargs.get('fields', ['distance', 'time'])
    pb_kw = kwargs.get('progress_bar', {})

    graph_to_atlas, atlas_to_graph = node_assignment(atlas, graph)

    destinations = list(graph.nodes)

    destinations_atlas = [graph_to_atlas[node] for node in destinations]

    for origin in ProgressBar(destinations, **pb_kw):

        origin_atlas = graph_to_atlas[origin]
        # print(origin, origin_atlas)

        costs, values, terminal = dijkstra(
            atlas,
            [origin_atlas],
            destinations = destinations,
            objective = objective,
            maximum_cost = maximum_cost,
            maximum_depth = maximum_depth,
            fields = fields,
            terminate_at_destinations = False,
            return_paths = False,
            )

        adj = {}

        destinations_reached = np.intersect1d(
            list(values.keys()),
            destinations_atlas,
            )

        for destination in destinations_reached:

            nodes = atlas_to_graph[destination]

            for node in nodes:

                adj[node] = values[destination]

        graph._adj[origin] = adj

        # break

    return graph

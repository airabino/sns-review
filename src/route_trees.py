import time
import numpy as np
import networkx as nx

from heapq import heappop, heappush
from itertools import count

from .progress_bar import ProgressBar

default_condition = lambda edge: True

def route_trees(graph, **kwargs):

    origins = kwargs.get('origins', graph.nodes)
    destinations = kwargs.get('destinations', graph.nodes)
    pivots = kwargs.get('pivots', graph.nodes)
    field = kwargs.get('field', 'distance')
    upper = kwargs.get('upper', default_condition)
    lower = kwargs.get('lower', default_condition)

    out = {o: {} for o in origins}

    for origin in origins:
        for destination in destinations:
            if origin != destination:

                out[origin][destination] = route_tree(
                    graph,
                    origin,
                    destination,
                    pivots,
                    field,
                    upper,
                    lower,
                    )

    return out

def successors(graph, **kwargs):

    origins = kwargs.get('origins', graph.nodes)
    destinations = kwargs.get('destinations', graph.nodes)
    pivots = kwargs.get('pivots', graph.nodes)
    field = kwargs.get('field', 'distance')
    upper = kwargs.get('upper', default_condition)
    lower = kwargs.get('lower', default_condition)

    out = {o: {} for o in origins}

    for origin in origins:
        for destination in ProgressBar(destinations):
            if origin != destination:

                out[origin][destination] = successors_pair(
                    graph,
                    origin,
                    destination,
                    pivots,
                    field,
                    upper,
                    lower,
                    )

    return out

def successors_pair(graph, origin, destination, pivots, field, upper, lower):

    _node = graph._node
    _adj = graph._adj

    successors = {s: [] for s in [origin] + pivots + [destination]}

    if destination in _adj[origin]:

        if upper(_adj[origin][destination]):

            successors[origin] = [destination]

            return successors

    added = {s: False for s in [origin] + pivots}

    heap = []
    c = count()

    heappush(heap, (next(c), origin))
    added[origin] = True

    idx = 0

    while heap:

        idx += 1

        # Next priority node
        _, source = heappop(heap)

        for target, edge in _adj[source].items():

            if upper(edge):

                # If the destination is reachable no need to expand the tree
                if target == destination:

                    successors[source].append(target)

                # If not then expand the tree
                elif target in pivots:

                    # Checking allowable edge
                    if (source == origin) | lower(edge):

                        improvement = (
                            _adj[target][destination][field] <
                            _adj[source][destination][field]
                            )

                        if improvement:

                            successors[source].append(target)

                            if not added[target]:

                                heappush(heap, (next(c), target))
                                added[target] = True


    has_successor = {s: True for s in [origin] + pivots + [destination]}

    for source, targets in successors.items():

        if targets == []:

            has_successor[source] = False
    
    has_successor[destination] = True

    for source, targets in successors.items():

        successors[source] = [t for t in targets if has_successor[t]]

    return successors

def shortest_path(graph, successors, origin, destination, fields):

    objective = fields[0]

    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths

    path_values = {}

    paths = {}

    visited = {} # dictionary of costs-to-reach for nodes

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    # Source is seen at the start of iteration and at 0 cost
    visited[origin] = np.inf
    paths[origin] = [origin]

    values = {f: 0 for f in fields}
    heappush(heap, (0, next(c), values, origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, values, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost
        path_values[source] = values

        for target in successors[source]:

            edge = edges[source][target]

            # Updating states for edge traversal
            cost_target = cost + edge.get(objective, 1)

            # Updating the weighted cost for the path
            savings = cost_target <= visited.get(target, np.inf)

            if savings:

                values_target = {k: v + edge.get(k, 0) for k, v in values.items()}
               
                visited[target] = cost_target

                heappush(heap, (cost_target, next(c), values_target, target))

                paths[target] = paths[source] + [target]

    return costs, path_values, paths

def route_tree(graph, origin, destination, pivots, field, upper, lower):

    _node = graph._node
    _adj = graph._adj

    predecessors = {s: [] for s in [origin] + pivots + [destination]}
    successors = {s: [] for s in [origin] + pivots + [destination]}

    added = {s: False for s in [origin] + pivots}

    heap = []
    c = count()

    heappush(heap, (next(c), origin))
    added[origin] = True

    idx = 0

    while heap:

        idx += 1

        # Next priority node
        _, source = heappop(heap)

        for target, edge in _adj[source].items():

            if upper(edge):

                # If the destination is reachable no need to expand the tree
                if target == destination:

                    successors[source].append(target)
                    predecessors[target].append(source)

                # If not then expand the tree
                elif target in pivots:

                    # Checking allowable edge
                    if (source == origin) | lower(edge):

                        improvement = (
                            _adj[target][destination][field] < _adj[source][destination][field]
                            )

                        if improvement:

                            successors[source].append(target)
                            predecessors[target].append(source)

                            if not added[target]:

                                heappush(heap, (next(c), target))
                                added[target] = True

    has_predecessor = {s: True for s in [origin] + pivots + [destination]}

    for target, sources in predecessors.items():

        if sources == []:

            has_predecessor[target] = False

    has_predecessor[origin] = True

    for target, sources in predecessors.items():

        predecessors[target] = [s for s in sources if has_predecessor[s]]

    has_successor = {s: True for s in [origin] + pivots + [destination]}

    for source, targets in successors.items():

        if targets == []:

            has_successor[source] = False
    
    has_successor[destination] = True

    for source, targets in successors.items():

        successors[source] = [t for t in targets if has_successor[t]]

    return {'predecessors': predecessors, 'successors': successors}

def recover_paths(successors, origin, destination):
    '''
    Recovers multiple branching path alternatives
    '''

    paths = []

    heap = []

    c = count()

    heappush(heap, (next(c), [origin]))

    # idx = 0

    while heap:

        # idx += 1

        _, path = heappop(heap)

        nodes = successors[path[-1]]

        for node in nodes:

            if node == destination:

                paths.append(path + [node])

            else:

                heappush(heap, (next(c), path + [node]))

        # if idx >= 100000:
        #     break

    return paths
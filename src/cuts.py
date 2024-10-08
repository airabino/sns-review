import numpy as np
import networkx as nx

from heapq import heappop, heappush
from itertools import count

def level_graph(graph, origin, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_edge_cost = kwargs.get('maximum_edge_cost', np.inf)
    maximum_path_cost = kwargs.get('maximum_path_cost', np.inf)

    _node = graph._node
    _adj = graph._adj

    costs = {} # dictionary of objective values for paths

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    heappush(heap, (0, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost

        for target, edge in _adj[source].items():

            # Updating states for edge traversal
            cost_edge = edge.get(objective, 1)

            cost_target = cost + cost_edge

            # Updating the weighted cost for the path
            savings = cost_target < costs.get(target, np.inf)

            feasible = True

            feasible *= cost_edge <= maximum_edge_cost
            feasible *= cost_target <= maximum_path_cost

            if savings & feasible:

                heappush(heap, (cost_target, next(c), target))

    nodes = []
    edges = []

    for source, adj in _adj.items():

        node = _node[source]
        node['cost'] = costs[source]
        nodes.append((source, node))
        
        for target, edge in adj.items():
            if costs[target] > costs[source]:

                edges.append((source, target, edge))

    level_graph = graph.__class__()
    level_graph.add_nodes_from(nodes)
    level_graph.add_edges_from(edges)

    return level_graph

def edmonds_karp_nodal(graph, origin, destination, **kwargs):

    capacity = kwargs.get('capacity', 'capacity')
    objective = kwargs.get('objective', 'objective')
    maximum_edge_cost = kwargs.get('maximum_edge_cost', np.inf)
    maximum_path_cost = kwargs.get('maximum_path_cost', np.inf)

    _node = graph._node
    _adj = graph._adj

    flow = 0

    node_flows = {k: 0 for k in graph.nodes}

    node_capacities = {k: v[capacity] for k, v in _node.items()}
    node_capacities[origin] = np.inf
    node_capacities[destination] = np.inf

    edge_costs = {s: {t: e[objective] for t, e in _adj[s].items()} for s in graph.nodes}

    for idx in range(graph.number_of_edges()):

        paths = {}

        visited = {origin: 0}

        queue = []
        c = count()

        heappush(queue, (0, next(c), origin, [origin]))

        while queue:

            # print(flow)

            cost, _, source, path = heappop(queue)

            if source == destination:

                paths[source] = path

                break

            if source in paths:

                continue  # already searched this node.

            paths[source] = path

            for target in _adj[source].keys():

                edge_cost = edge_costs[source][target]

                path_cost = cost + edge_cost

                accept = True

                accept *= edge_cost <= maximum_edge_cost
                accept *= path_cost <= maximum_path_cost
                accept *= path_cost < visited.get(target, np.inf)

                if accept:

                    new_path = path + [target]

                    heappush(queue, (path_cost, next(c), target, new_path))

        if destination in paths:

            path = paths[destination]

            if len(path) < 3:

                flow = _node[destination][capacity]
                cutset = [destination]
                reachable = {k for k in paths.keys()}
                unreachable = set(graph.nodes) - reachable
                partition = (list(reachable), list(unreachable))

                return flow, node_flows, cutset, partition

            marginal_flow = min([node_capacities[source] for source in path])

            flow += marginal_flow

            for source in path:

                node_capacities[source] -= marginal_flow
                node_flows[source] += marginal_flow

                if node_capacities[source] <= 0:

                    for target in edge_costs[source].keys():

                        edge_costs[source][target] = np.inf

        else:

            break

    # cutset = [k for k, v in node_capacities.items() if v <= 0]
    reachable = {k for k in paths.keys()}
    unreachable = set(graph.nodes) - reachable
    partition = (list(reachable), list(unreachable))

    cutset = []

    # print(unreachable)

    for source in partition[0]:

        for target, edge in _adj[source].items():

            if edge.get(objective, 1) <= maximum_edge_cost:

                if target in unreachable:

                    cutset.append(source)

                    break

    return flow, node_flows, cutset, partition

def gomory_hu_nodal(graph, **kwargs):

    tree = {}
    labels = {}

    iter_nodes = iter(graph)
    root = next(iter_nodes)
    for n in iter_nodes:
        tree[n] = root

    for source in tree:

        target = tree[source]

        flow, _, _, partition = edmonds_karp_nodal(
            graph, source, target, **kwargs
            )

        if np.isnan(flow):

            print(source, target)

        labels[(source, target)] = flow

        for node in partition[0]:
            if node != source and node in tree and tree[node] == target:

                tree[node] = source
                labels[node, source] = labels.get((node, target), flow)

                if labels[node, source] == np.nan:

                    print(labels[node, source], source, target)

        if target != root and tree[target] in partition[0]:

            # print(labels[source, tree[target]])

            labels[source, tree[target]] = labels[target, tree[target]]

            if labels[source, tree[target]] == np.nan:

                print(source, target)

            # print(labels[source, tree[target]])
            # print(np.isnan(labels[source, tree[target]]))

            labels[target, source] = flow

            if labels[target, source] == np.nan:

                print(source, target)

            tree[source] = tree[target]
            tree[target] = source

            # print(labels[source, tree[target]])

            

    ght = nx.Graph()
    ght.add_nodes_from(graph)
    ght.add_weighted_edges_from(((u, v, labels[u, v]) for u, v in tree.items()))

    # print(labels)

    return ght
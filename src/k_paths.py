import numpy as np
import networkx as nx

from heapq import heappop, heappush
from itertools import count
from sys import maxsize

def _double_dijkstra(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)

    nodes = graph._node
    edges = graph._adj
    edges_r = graph.reverse()._adj

    costs = []
    paths = []

    costs_o = {}
    costs_d = {}

    paths_o = {}
    paths_d = {}

    visited_o = {} # dictionary of costs-to-reach for nodes
    visited_d = {} # dictionary of costs-to-reach for nodes

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap_o = [] # heap is heapq with 3-tuples (cost, c, node)
    heap_d = [] # heap is heapq with 3-tuples (cost, c, node)

    # Source is seen at the start of iteration and at 0 cost
    visited_o[origin] = 0
    visited_d[destination] = 0

    radius = np.inf
    shortest_path = []

    heappush(heap_o, (0, next(c), origin, [origin]))
    heappush(heap_d, (0, next(c), destination, [destination]))

    while heap_o and heap_d: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost_o, _, source_o, path_o = heappop(heap_o)
        cost_d, _, source_d, path_d = heappop(heap_d)

        visited_o[source_o] = True
        visited_d[source_d] = True

        if source_o not in costs_o:

            costs_o[source_o] = cost_o
            paths_o[source_o] = path_o

            for target, edge in edges[source_o].items():

                # Updating states for edge traversal
                cost_target = cost_o + edge.get(objective, 1)

                # Updating the weighted cost for the path
                savings = cost_target <= visited_o.get(target, np.inf)

                feasible = cost_target <= maximum_cost

                if savings & feasible:

                    path_target = path_o + [target]

                    heappush(heap_o, (cost_target, next(c), target, path_target))

                tentative_radius = cost_target + costs_d.get(target, np.inf)

                if visited_d.get(target, False) and (tentative_radius < radius):

                    radius = tentative_radius
                    shortest_path = path_o + paths_d[target]

        if source_d not in costs_d:

            costs_d[source_d] = cost_d
            paths_d[source_d] = path_d

            for target, edge in edges_r[source_d].items():

                # Updating states for edge traversal
                cost_target = cost_d + edge.get(objective, 1)

                # Updating the weighted cost for the path
                savings = cost_target <= visited_d.get(target, np.inf)

                feasible = cost_target <= maximum_cost

                if savings & feasible:

                    path_target = [target] + path_d

                    heappush(heap_d, (cost_target, next(c), target, path_target))

                tentative_radius = cost_target + costs_o.get(target, np.inf)

                if visited_o.get(target, False) and (tentative_radius < radius):

                    radius = tentative_radius
                    shortest_path = paths_o[target] + path_d

        if radius < np.inf:

            costs.append(radius)
            paths.append(shortest_path)

        if costs_o[source_o] + costs_d[source_d] >= radius:

            break



    return costs, paths

def _dijkstra(graph, origin, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)

    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths

    paths = {}

    visited = {} # dictionary of costs-to-reach for nodes

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    # Source is seen at the start of iteration and at 0 cost
    visited[origin] = np.inf

    paths[origin] = [origin]

    heappush(heap, (0, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost

        for target, edge in edges[source].items():

            # Updating states for edge traversal
            cost_target = cost + edge.get(objective, 1)

            # Updating the weighted cost for the path
            savings = cost_target <= visited.get(target, np.inf)

            feasible = cost_target <= maximum_cost

            if savings & feasible:
               
                visited[target] = cost_target

                paths[target] = paths[source] + [target]

                heappush(heap, (cost_target, next(c), target))

    return costs, paths

def dijkstra(graph, origin, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_edge_cost = kwargs.get('maximum_edge_cost', np.inf)
    maximum_path_cost = kwargs.get('maximum_path_cost', np.inf)

    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths

    paths = {}
    predecessors = {}

    visited = {} # dictionary of costs-to-reach for nodes

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    # Source is seen at the start of iteration and at 0 cost
    visited[origin] = 0

    paths[origin] = [origin]
    predecessors[origin] = origin

    heappush(heap, (0, next(c), origin, origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, source, pred = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost
        predecessors[source] = pred

        for target, edge in edges[source].items():

            # Updating states for edge traversal
            cost_edge = edge.get(objective, 1)

            cost_target = cost + cost_edge

            # Updating the weighted cost for the path
            savings = cost_target < visited.get(target, np.inf)

            feasible = True

            feasible *= cost_edge <= maximum_edge_cost
            feasible *= cost_target <= maximum_path_cost
            # feasible = True

            # if not feasible:

                # print(cost_edge)

            if savings & feasible:
               
                visited[target] = cost_target

                paths[target] = paths[source] + [target]

                # predecessors[target] = source

                heappush(heap, (cost_target, next(c), target, source))
    # print(paths)

    return costs, paths, predecessors

def _predecessors(graph, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)

    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths

    predecessors = {}
    paths = {}

    visited = {} # dictionary of costs-to-reach for nodes

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    # Source is seen at the start of iteration and at 0 cost
    visited[destination] = np.inf

    predecessors[destination] = destination
    paths[destination] = [destination]

    heappush(heap, (0, next(c), destination))

    while heap: # Iterating while there are accessible unseen nodes
        # print(heap)

        # Popping the lowest cost unseen node from the heap
        cost, _, target = heappop(heap)

        if target in costs:

            continue  # already searched this node.

        costs[target] = cost

        for source, edge in edges[target].items():

            # Updating states for edge traversal
            cost_source = cost + edge.get(objective, 1)

            # Updating the weighted cost for the path
            savings = cost_source <= visited.get(source, np.inf)

            feasible = cost_source <= maximum_cost

            if savings & feasible:
               
                visited[source] = cost_source

                predecessors[target] = source

                paths[source] = [source] + paths[target]

                heappush(heap, (cost_source, next(c), source))

    return costs, predecessors, paths

def k_pop(graph, origin, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths

    paths = {}

    visited = {} # dictionary of costs-to-reach for nodes

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    # Source is seen at the start of iteration and at 0 cost
    visited[origin] = np.inf

    paths[origin] = [origin]

    heappush(heap, (0, next(c), origin))

    k = 0

    while (k < maximum_paths) and heap:

        k += 1

        # Popping the lowest cost unseen node from the heap
        cost, _, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost

        for target, edge in edges[source].items():

            # Updating states for edge traversal
            cost_target = cost + edge.get(objective, 1)
               
            visited[target] = cost_target

            paths[target] = paths[source] + [target]

            heappush(heap, (cost_target, next(c), target))

    return costs, paths

def algorithm_6(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    if graph.is_directed():

        costs_o, paths_o = _dijkstra(
            graph, origin, objective = objective, maximum_cost = maximum_cost,
            )

        costs_d, paths_d = _dijkstra(
            graph.reverse(), destination, objective = objective, maximum_cost = maximum_cost,
            )

    else:

        costs_o, paths_o = _dijkstra(
            graph, origin, objective = objective, maximum_cost = maximum_cost,
            )

        costs_d, paths_d = _dijkstra(
            graph, destination, objective = objective, maximum_cost = maximum_cost,
            )

    costs = []
    paths = []

    shortest_path = paths_o[destination]

    successors = {k: None for k in graph.nodes()}

    for idx, p in enumerate(shortest_path[:-1]):

        successors[p] = shortest_path[idx + 1]

    c = count()
    queue = [(costs_d[origin], next(c), [origin], shortest_path)]

    k = 0

    while queue and (k < maximum_paths):
        
        k += 1

        cost, _, stub, path = heappop(queue)
        source = stub[-1]

        costs.append(cost)
        paths.append(path)

        for target, edge in graph._adj[source].items():

            # path_stub

            if not successors[source] or (successors[source] != target):

                if target in stub:

                    continue

                # new_cost = edge[objective] + costs_d[target] - costs_d[source]
                new_cost = costs_o[source] + edge[objective] + costs_d[target]
                new_stub = stub + [target]
                new_path = stub + paths_d[target][::-1]

                heappush(queue, (new_cost, next(c), new_stub, new_path))

    return costs, paths

def algorithm_5(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    if graph.is_directed():

        costs_o, paths_o = _dijkstra(
            graph, origin, objective = objective, maximum_cost = maximum_cost,
            )

        costs_d, paths_d = _dijkstra(
            graph.reverse(), destination, objective = objective, maximum_cost = maximum_cost,
            )

    else:

        costs_o, paths_o = _dijkstra(
            graph, origin, objective = objective, maximum_cost = maximum_cost,
            )

        costs_d, paths_d = _dijkstra(
            graph, destination, objective = objective, maximum_cost = maximum_cost,
            )

    costs = []
    paths = []

    reachable = np.intersect1d(list(costs_o.keys()), list(costs_d.keys()))

    for n in reachable:

        cost = costs_o[n] + costs_d[n]

        path_f = paths_o[n]
        path_r = paths_d[n]

        for p in path_f:

            if p in path_r:

                path_f = path_f[:path_f.index(p) + 1]
                path_r = path_r[:path_r.index(p)]
                break

        path = path_f + path_r[::-1]

        costs.append(cost)
        paths.append(path)

    return costs, paths

def algorithm_7(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_ratio = kwargs.get('maximum_ratio', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    if graph.is_directed():

        costs_o, paths_o = _dijkstra(
            graph, origin, objective = objective,
            )

        costs_d, paths_d = _dijkstra(
            graph.reverse(), destination, objective = objective,
            )

    else:

        costs_o, paths_o = _dijkstra(
            graph, origin, objective = objective,
            )

        costs_d, paths_d = _dijkstra(
            graph, destination, objective = objective,
            )

    reachable = np.intersect1d(list(costs_o.keys()), list(costs_d.keys()))
    edge_list = list(nx.to_edgelist(graph, nodelist = reachable))

    c = count()
    queue = []

    heappush(queue, (costs_o[destination], next(c), paths_o[destination]))

    for edge in edge_list:

        source, target, info = edge

        # if costs_d[source] <= costs_d[target]:

        #     continue

        cost = costs_o[source] + info[objective] + costs_d[target]

        path_f = paths_o[source]
        path_r = paths_d[target]

        for p in path_f:

            if p in path_r:

                path_f = path_f[:path_f.index(p) + 1]
                path_r = path_r[:path_r.index(p)]
                break

        path = path_f + path_r[::-1]

        heappush(queue, (cost, next(c), path))

    costs = []
    paths = []

    k = 0
    cost = costs_o[destination]
    ratio = 1
    # print(queue)

    # print(queue and (k < maximum_paths) and (cost <= maximum_cost))

    while (
        queue and
        (k < maximum_paths) and
        (ratio <= maximum_ratio) and
        (cost <= maximum_cost)
        ):

        k += 1

        # print(k)

        cost, _, path = heappop(queue)
        ratio = cost / costs_o[destination]
        # print(cost, k)

        # print((k < maximum_paths), (cost <= maximum_cost))

        costs.append(cost)
        paths.append(path)

    return costs, paths

def algorithm_2(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    _adj = graph._adj

    edges = []

    direct_costs, predecessors, direct_paths = _predecessors(
        graph.reverse(), destination, objective = objective, maximum_cost = maximum_cost,
        )

    remove = [(s, t) for t, s in predecessors.items()]

    graph_prime = graph.copy()

    graph_prime.remove_edges_from(remove)

    _costs, _paths = k_pop(
        graph, origin, objective = objective, maximum_paths = maximum_paths
        )

    costs = []
    paths = []

    for source, _ in _costs.items():

        costs.append(_costs[source] + direct_costs[source])
        paths.append(_paths[source] + direct_paths[source][1:])

    return costs, paths

def algorithm_1(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    nodes = graph._node
    edges = graph._adj

    costs = []
    paths = []

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    counts = {k: 0 for k in nodes.keys()}

    heappush(heap, (0, next(c), origin, [origin]))

    while (counts[destination] < maximum_paths) and heap:

        cost, _, source, path = heappop(heap)

        if counts[source] == maximum_paths:

            continue

        counts[source] += 1

        if source == destination:

            costs.append(cost)
            paths.append(path)

        for target, edge in edges[source].items():

            if target in path:

                continue

            new_cost = cost + edge.get(objective, 1)
            new_path = path + [target]

            heappush(heap, (new_cost, next(c), target, new_path))

    return costs, paths

def yen(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    direct_costs, predecessors, direct_paths = _dijkstra(
        graph, origin, objective = objective, maximum_cost = maximum_cost,
        )

    od_path = direct_paths[destination]

    paths = []

    shortest_paths = [od_path]

    for k in range(1, maximum_paths):

        g = graph.copy()

        _node = g._node
        _adj = g._adj

        for idx in range(len(od_path) - 2):

            spur_node = shortest_paths[k - 1][idx]
            next_node = shortest_paths[k - 1][idx + 1]
            root_path = shortest_paths[k - 1][:idx]

            print(root_path, od_path)

            for shortest_path in shortest_paths:

                if root_path == shortest_path[:idx]:

                    store_edge = (spur_node, next_node, _adj[spur_node][next_node])

                    g.remove_edge(spur_node, next_node)

                g.remove_nodes_from([n for n in root_path if n != spur_node])

                spur_costs, _, spur_paths = _dijkstra(
                    g, origin, objective = objective, maximum_cost = maximum_cost,
                    )

                print(spur_path)



                spur_path = spur_paths[destination]

                print(root_path, spur_path)

                total_path = root_path + spur_path

                if total_path not in paths:

                    paths.append(total_path)

    return paths

def algorithm_3(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    if graph.is_directed():

        direct_costs, predecessors, direct_paths = _dijkstra(
            graph.reverse(), destination, objective = objective, maximum_cost = maximum_cost,
            )

    else:

        direct_costs, predecessors, direct_paths = _dijkstra(
            graph, destination, objective = objective, maximum_cost = maximum_cost,
            )

    # print(direct_costs)

    nodes = graph._node
    edges = graph._adj

    costs = []
    paths = []
    ctgs = []

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    counts = {k: 0 for k in nodes.keys()}

    heappush(heap, (0, next(c), origin, [origin], [direct_costs[origin]]))

    while (counts[destination] < maximum_paths) and heap:

        cost, _, source, path, ctg = heappop(heap)

        if counts[source] == maximum_paths:

            continue

        counts[source] += 1

        if source == destination:

            costs.append(cost)
            paths.append(path)
            ctgs.append(ctg)

        for target, edge in edges[source].items():

            if target in path:

                continue

            if direct_costs[target] > direct_costs[source]:

                continue

            new_cost = cost + edge.get(objective, 1)
            new_path = path + [target]
            new_ctg = ctg + [direct_costs[target]]

            heappush(heap, (new_cost, next(c), target, new_path, new_ctg))

    return costs, paths, ctgs

def algorithm_4(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)
    slack_f = kwargs.get('slack_f', 1)
    slack_r = kwargs.get('slack_r', 1)

    if graph.is_directed():

        costs_o, predecessors, paths_o = _dijkstra(
            graph, origin, objective = objective, maximum_cost = maximum_cost,
            )

        costs_d, predecessors, paths_d = _dijkstra(
            graph.reverse(), destination, objective = objective, maximum_cost = maximum_cost,
            )

    else:

        costs_o, predecessors, paths_o = _dijkstra(
            graph, origin, objective = objective, maximum_cost = maximum_cost,
            )

        costs_d, predecessors, paths_d = _dijkstra(
            graph, destination, objective = objective, maximum_cost = maximum_cost,
            )

    nodes = graph._node
    edges = graph._adj

    costs = []
    paths = []
    ctgs = []

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    counts = {k: 0 for k in nodes.keys()}

    heappush(heap, (next(c), origin))

    while (len(paths) < maximum_paths) and heap:

        _, source = heappop(heap)

        # if counts[source] == maximum_paths:

        #     continue

        # counts[source] += 1

        # if source == destination:

        #     costs.append(cost)
        #     paths.append(path)
            # ctgs.append(ctg)

        for target, edge in edges[source].items():

            # if target in path:

            #     continue

            # if costs_o[target] < costs_o[source] / slack_f:

            #     continue

            if costs_d[target] > costs_d[source] * slack_r:

                continue

            costs.append(costs_o[target] + costs_d[target])
            paths.append(paths_o[target] + paths_d[target][::-1][1:])

            # new_cost = cost + edge.get(objective, 1)
            # new_path = path + [target]
            # new_ctg = ctg + [direct_costs[target]]

            heappush(heap, (next(c), target))

    print(len(paths))

    return costs, paths, ctgs

class LeftistHeap():

    def __init__(self, rank, key, value, left, right):

        self.rank = rank
        self.key = key
        self.value = value
        self.left = left
        self.right = right

    @staticmethod
    def insert(a, k, v):

        if not a or k < a.key:

            return LeftistHeap(1, k, v, a, None)

        l, r = a.left, LeftistHeap.insert(a.right, k, v)

        if not l or r.rank > l.rank:

            l, r = r, l

        return LeftistHeap(r.rank + 1 if r else 1, a.key, a.value, l, r)

    def __lt__(self, _):

        return False

class TwoHeap():

    def __init__(self, rank, cost, root, tail, left, right):

        self.rank = rank
        self.cost = cost
        self.root = root
        self.tail = tail
        self.left = left
        self.right = right

    @staticmethod
    def insert(element, cost, root, tail):
        '''
        Each node is the root of a tree with a left and right child where the cost
        of the left child is lower than that of the right child. The left and right
        children are the lowest-cost options out of the adjacency of the root.
        element is the tree at a given node (u) which may or may not have been created
        yet. Cost is the cost of the sidetrack (u, v, ..., t) compared to (u, p_u, ..., t)
        created by deviation from the optimal path from u to t through v. Root is node v.

        If there is no heap at node then a rank 1 tree is created at node v with no
        children.

        If the cost of the sidetrack through node v is lower than the cost of the path
        from u then u is a child of v and a rank 1 node is created at v where u is the
        left child

        Otherwise v is considered as a child of u

        If u has no right child or v is lower cost than the right child then v becomes
        the right child.

        If the right child is lower cost than the left child then the children are
        swapped

        Finally, the rank of the root is fixed to be one higher than the rank of the
        right child
        '''
        if not element or cost < element.cost:

            out = TwoHeap(1, cost, root, tail, element, None)

            return out

        left = element.left
        right = TwoHeap.insert(element.right, cost, root, tail)

        if not left or right.rank > left.rank:

            left, right = right, left

        out = TwoHeap(
            right.rank + 1 if right else 1,
            element.cost, element.root, element.tail, left, right
            )

        return out

    def __lt__(self, _):

        return False

def algorithm_8(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_ratio = kwargs.get('maximum_ratio', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    if graph.is_directed():

        costs, paths = _dijkstra(
            graph.reverse(), destination, objective = objective,
            )

    else:

        costs, paths = _dijkstra(
            graph, destination, objective = objective,
            )


    # creating the sidetrack graph
    sidetracks = {}

    for source, adj in graph._adj.items():

        if source not in costs:

            continue

        sidetracks[source] = {}

        for target, edge in adj.items():

            if target == paths[source][-2]:

                continue

            cost = edge[objective] + costs[target] - costs[source]

            sidetracks[source][target] = cost

    shortest_path_cost = costs[origin]

    costs_out = []
    paths_out = []

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    heappush(heap, (0, next(c), origin, [origin]))

    while (len(costs_out) < maximum_paths) and heap:

        cost, _, source, path = heappop(heap)

        # print(cost, end = '\r')
        pt = path[:-1] + paths[path[-1]][::-1]

        costs_out.append(
            sum(
                [graph._adj[pt[idx]][pt[idx + 1]][objective] for idx in range(len(pt) - 1)]
                )
            )
        paths_out.append(pt)

        for target, edge in graph._adj[source].items():

            if target in path:

                continue

            if costs[target] >= costs[source]:

                continue

            new_cost = costs[target] + edge.get(objective, 1) - costs[source]

            # print(costs[target], edge.get(objective, 1), costs[source], new_cost)

            new_path = path + [target]

            heappush(heap, (new_cost, next(c), target, new_path))

    return costs_out, paths_out

def algorithm_9(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_ratio = kwargs.get('maximum_ratio', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    # print(origin, destination)

    if graph.is_directed():

        costs, paths, successors = dijkstra(
            graph.reverse(), destination, objective = objective,
            # maximum_cost = maximum_cost,
            )

        _, paths_f, _ = dijkstra(
            graph, origin, objective = objective,
            # maximum_cost = maximum_cost,
            )

    else:

        # print('s')

        costs, paths, successors = dijkstra(
            graph, destination, objective = objective,
            # maximum_cost = maximum_cost,
            )

        _, paths_f, _ = dijkstra(
            graph, origin, objective = objective,
            # maximum_cost = maximum_cost,
            )

    predecessors = {k: [] for k in successors.keys()}

    for k, v in successors.items():

        if k != destination:

            predecessors[v].append(k)

    # print(paths[origin])
    # print(paths_f[destination])

    # print()
    
    # print(costs)
    # print(paths)
    # # print(successors)
    # print(predecessors)

    # creating the sidetrack graph
    sidetracks = {k: None for k in paths.keys()}

    queue = []
    c = count()

    heappush(queue, (destination, next(c)))

    while queue:

        source, _ = heappop(queue)

        seen_optimal = False

        for target, edge in graph._adj[source].items():

            cost = edge[objective] + costs[target] - costs[source]

            if not seen_optimal and (target == successors[source]) and cost == 0:

                seen_optimal = True

                continue

            sidetracks[source] = TwoHeap.insert(
                sidetracks[source], cost, target, source,
                )

        for predecessor in predecessors[source]:

            sidetracks[predecessor] = sidetracks[source]

            heappush(queue, (predecessor, next(c)))

    # print(sum([v is not None for k, v in sidetracks.items()]))

    optimal_path_cost = costs[origin]
    optimal_path = paths_f[destination]

    costs_out = []
    paths_out = []
    # print(paths[origin], optimal_path_cost, optimal_path)

    queue = []
    c = count()

    heappush(
        queue,
        (optimal_path_cost + sidetracks[origin].cost, next(c), sidetracks[origin], [])
        )

    while queue and (len(costs_out) < maximum_paths):

        cost, _, heap, deviations = heappop(queue)

        path = optimal_path[:]

        valid_path = True

        try:

            for tail, head in deviations:
                # print('d', path, deviations)

                # print('e', paths[head][::-1])

                path = path[:path.index(tail) + 1] + paths[head][::-1]

            # print('a', path, deviations)

        except:
            # print('s')

            # print(deviations)

            valid_path = False

        if len(path) == len(set(path)) and valid_path:

            costs_out.append(cost)
            paths_out.append(path)

        # If the root node of the heap has a heap then add it to the queue
        if sidetracks[heap.root]:

            new_cost = cost + sidetracks[heap.root].cost
            new_deviations = deviations + [(heap.tail, heap.root)]

            heappush(queue, (new_cost, next(c), sidetracks[heap.root], new_deviations))

        # If the left child node of the heap has a heap then add it to the queue
        if heap.left:

            new_cost = cost + heap.left.cost - heap.cost
            new_deviations = deviations + [(heap.left.tail, heap.left.root)]

            heappush(queue, (new_cost, next(c), heap.left, new_deviations))

        # If the right child node of the heap has a heap then add it to the queue
        if heap.right:

            new_cost = cost + heap.right.cost - heap.cost
            new_deviations = deviations + [(heap.right.tail, heap.right.root)]

            heappush(queue, (new_cost, next(c), heap.right, new_deviations))

    return costs_out, paths_out

def algorithm_10(graph, origin, destination, **kwargs):

    print(graph.number_of_nodes(), graph.number_of_edges())

    objective = kwargs.get('objective', 'objective')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_ratio = kwargs.get('maximum_ratio', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)

    # print(graph.is_directed())

    # if graph.is_directed():

    print(destination, objective)

    costs, paths, successors = dijkstra(
        graph.reverse(), destination, objective = objective,
        )

        # _, paths_f, _ = dijkstra(
        #     graph, origin, objective = objective,
        #     )

    return successors

    # else:

        # costs, paths, successors = dijkstra(
        #     graph, destination, objective = objective,
        #     )

        # _, paths_f, _ = dijkstra(
        #     graph, origin, objective = objective,
        #     )

    predecessors = {k: [] for k in paths.keys()}

    # for k, v in successors.items():

    #     if k !=  destination:

    #         predecessors[v].append(k)

    # creating the sidetrack graph
    sidetracks = {k: None for k in paths.keys()}

    # queue = []
    # c = count()

    # heappush(queue, (destination, next(c)))

    # k = 0

    # while queue:

    #     k += 1

    #     source, _ = heappop(queue)

    #     seen_optimal = False

    #     for target, edge in graph._adj[source].items():

    #         cost = edge[objective] + costs[target] - costs[source]

    #         if not seen_optimal and (target == successors[source]) and cost == 0:

    #             seen_optimal = True

    #             continue

    #         sidetracks[source] = TwoHeap.insert(
    #             sidetracks[source], cost, target, source,
    #             )

    #     for predecessor in predecessors[source]:

    #         sidetracks[predecessor] = sidetracks[source]

    #         heappush(queue, (predecessor, next(c)))

    optimal_path_cost = costs[origin]
    optimal_path = paths_f[destination]

    costs_out = [optimal_path_cost]
    paths_out = [optimal_path]

    # queue = []
    # c = count()
    
    # heappush(
    #     queue,
    #     (optimal_path_cost + sidetracks[origin].cost, next(c), sidetracks[origin], [])
    #     )

    # while queue and (len(costs_out) < maximum_paths):
    #     # break

    #     cost, _, heap, deviations = heappop(queue)

    #     path = optimal_path[:]

    #     valid_path = True

    #     try:

    #         for tail, head in deviations:

    #             path = path[:path.index(tail) + 1] + paths[head][::-1]

    #     except:

    #         valid_path = False

    #     if len(path) == len(set(path)) and valid_path:

    #         costs_out.append(cost)
    #         paths_out.append(path)

    #     # If the root node of the heap has a heap then add it to the queue
    #     if sidetracks[heap.root]:

    #         new_cost = cost + sidetracks[heap.root].cost
    #         new_deviations = deviations + [(heap.tail, heap.root)]

    #         heappush(queue, (new_cost, next(c), sidetracks[heap.root], new_deviations))

    #     # If the left child node of the heap has a heap then add it to the queue
    #     if heap.left:

    #         new_cost = cost + heap.left.cost - heap.cost
    #         new_deviations = deviations + [(heap.left.tail, heap.left.root)]

    #         heappush(queue, (new_cost, next(c), heap.left, new_deviations))

    #     # If the right child node of the heap has a heap then add it to the queue
    #     if heap.right:

    #         new_cost = cost + heap.right.cost - heap.cost
    #         new_deviations = deviations + [(heap.right.tail, heap.right.root)]

    #         heappush(queue, (new_cost, next(c), heap.right, new_deviations))

    return costs_out, paths_out, sidetracks, predecessors, successors

def algorithm_11(graph, origin, destination, **kwargs):

    objective = kwargs.get('objective', 'objective')
    maximum_ratio = kwargs.get('maximum_ratio', np.inf)
    maximum_paths = kwargs.get('maximum_paths', np.inf)
    bounds = kwargs.get('bounds', (0, np.inf))
    dijkstra_kw = kwargs.get('dijkstra', {})

    _adj = graph._adj

    if graph.is_directed():

        costs_r, paths_r, _ = dijkstra(
            graph.reverse(), destination, objective = objective, **dijkstra_kw
            )

        costs_f, paths_f, _ = dijkstra(
            graph, origin, objective = objective, **dijkstra_kw
            )

    else:

        costs_r, paths_r, _ = dijkstra(
            graph, destination, objective = objective, **dijkstra_kw
            )

        costs_f, paths_f, _ = dijkstra(
            graph, origin, objective = objective, **dijkstra_kw
            )
    
    optimal_cost = costs_f[destination]
    optimal_path = paths_f[destination]

    queue = []
    c = count()

    heappush(queue, (0, next(c), origin, optimal_path))

    costs_out = []
    paths_out = []

    while queue and (len(costs_out) < maximum_paths):
        # print('s')

        cost, _, source, path = heappop(queue)

        costs_out.append(optimal_cost + cost)
        paths_out.append(path)

        for target, edge in graph._adj[source].items():
            # print('1')

            # Monotonic approach
            source_dest_cost = _adj[source].get(destination, {}).get(objective, np.inf)
            target_dest_cost = _adj[target].get(destination, {}).get(objective, np.inf)

            # print(source_dest_cost, target_dest_cost)

            if source_dest_cost < target_dest_cost:

                continue

            # Cost range
            # print('2')
            edge_cost = edge.get(objective, np.inf)
            # print(edge_cost, bounds)

            if (edge_cost < bounds[0]) or (edge_cost > bounds[1]):

                continue

            # Unnecessary detours
            # print('3')
            if edge_cost > source_dest_cost:

                continue

            # Max cost ratio
            # print('4')
            new_path = path[:path.index(source) + 1] + paths_r[target][::-1]
            new_cost = cost + edge_cost + costs_r[target] - costs_r[source]

            # print(edge_cost + costs_r[target] - costs_r[source])
            path_cost = optimal_cost +  new_cost

            cost_ratio = path_cost / optimal_cost

            if cost_ratio > maximum_ratio:

                continue

            # print('5')
            heappush(queue, (new_cost, next(c), target, new_path))

    return costs_out, paths_out
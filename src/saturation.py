import time
import numpy as np
import networkx as nx

from copy import copy

from .progress_bar import ProgressBar
from .graph import cypher
from .floyd_warshall import floyd_warshall, recover_flow

class Saturation():

    def __init__(self, graph, ratios, queue, **kwargs):

        self.graph = graph
        self.nodes = list(graph.nodes)
        self._node = graph._node

        self.queue = queue
        self.ratios = ratios

        self.build(**kwargs)

    def build(self, **kwargs):

        # Building the adjacency matrix
        field = kwargs.get('field', None)
        self.adjacency = nx.to_numpy_array(self.graph, weight = field)

        self.terminals = np.array(kwargs.get('terminals', self.nodes))
        self.pivots = np.array(kwargs.get('pivots', self.nodes))
        self.batch_size = kwargs.get('batch_size', len(self.pivots))

    def performance(self, values):

        return np.mean(values * self.ratios)

    def run(self, **kwargs):

        limits = kwargs.get('limits', {})

        encode, decode = cypher(self.graph)
        terminals_idx = np.array([encode[n] for n in self.terminals])
        # terminals_indices = np.meshgrid(terminals_idx, terminals_idx)

        adjacency = self.adjacency.copy()

        n = self.graph.number_of_nodes()

        # Nodal status arrays

        traffic = np.array([0. for v in self._node.values()])
        capacity = np.array([v['capacity'] for v in self._node.values()])
        cutoff = np.array([v['cutoff'] for v in self._node.values()])
        saturated = np.array([v['saturated'] for v in self._node.values()])
        available = ~saturated

        functional = (
            np.array([v['functional'] for v in self._node.values()])
            )
        functional[functional == 0] = 1

        service_rate = (
            np.array([v['service_rate'] for v in self._node.values()])
            )
        service_rate[service_rate == 0] = 1

        rho = traffic / (service_rate * functional)
        queue_time = np.atleast_2d(self.queue(functional, rho) / service_rate).T

        node_ratios = self.ratios.sum(axis = 1)

        _network_load = []
        _network_performance = []
        _network_availability = []
        # _network_connectivity = []

        _node_load = []
        _node_performance = []
        _node_availability = []
        # _node_connectivity = []

        total_flow = 0

        null_predecessors = np.tile(list(range(n)), (n, 1)).T

        for idx in ProgressBar(range(sum(available)), **kwargs.get('progress_bar', {})):

            # Computing shortest paths for current graph state
            kw = {
                'pivots': np.array(self.nodes)[available][:self.batch_size],
                'adjacency': adjacency,
            }

            costs, predecessors = floyd_warshall(
                self.graph, **kw,
            )

            # Back-propogating O/D flow ratios to pivots
            flows = np.zeros(predecessors.shape)
            flows = recover_flow(
                predecessors.copy(), flows, terminals_idx, terminals_idx, self.ratios
            )
            node_flows = flows.sum(axis = 0)

            #Network performance
            if idx  == 0:

                baseline_performance = costs[terminals_idx][:, terminals_idx].mean()
                baseline_connectivity = predecessors != null_predecessors
                baseline_connectivity_sum = baseline_connectivity.sum()
                baseline_connectivity_rows = baseline_connectivity.sum(axis = 1)


            connectivity = predecessors != null_predecessors
            od_connectivity = connectivity[terminals_idx][:, terminals_idx].sum()
            od_performance = costs[terminals_idx][:, terminals_idx].mean()

            # Finding the blocking flow rate
            gap = capacity - traffic
            allowable = gap / node_flows
            blocking_flow = min(allowable[available])
            total_flow += blocking_flow
            traffic[available] += blocking_flow * node_flows[available]
            
            # Updating nodes
            saturated[available] = traffic[available] >= capacity[available]
            available = ~saturated

            # Updating adjacency
            rho = traffic / (service_rate * functional)
            queue_time = self.queue(functional, rho) / service_rate
            queue_time[queue_time > cutoff] = cutoff[queue_time > cutoff]
            adjacency += np.atleast_2d(queue_time).T

            # Gathering results
            _network_load.append(total_flow)
            _network_performance.append(od_performance)
            # _network_connectivity.append(od_connectivity / baseline_connectivity_sum)
            _network_availability.append(sum(available))

            _node_load.append({k: traffic[idx] for idx, k in enumerate(self.nodes)})
            _node_performance.append(
                {k: queue_time[idx] for idx, k in enumerate(self.nodes)}
                )
            _node_availability.append(
                {k: available[idx] for idx, k in enumerate(self.nodes)}
                )
            # _node_connectivity.append(
            #     {k: connectivity[idx].sum() / baseline_connectivity_rows[idx]\
            #      for idx, k in enumerate(self.nodes)}
            #     )

            # Assessing loop termination conditions
            conditions = (
                sum(available) == limits.get('availability', 0),
                od_connectivity == limits.get('connectivity', 0),
                od_performance / baseline_performance > limits.get('performance', np.inf),
                idx >= limits.get('iterations', np.inf),
                )

            if any(conditions):

                break

        results = {
            'network': {
                'load': np.array(_network_load),
                'performance': np.array(_network_performance),
                'availability': np.array(_network_availability),
                # 'connectivity': np.array(_network_connectivity),
                },
            'node': {
                'load': _node_load,
                'performance': _node_performance,
                'availability': _node_availability,
                # 'connectivity': _node_connectivity,
                },
            }

        return results
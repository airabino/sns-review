import sys
import time
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import pyomo.environ as pyomo
import networkx as nx

from copy import copy

from scipy.spatial import KDTree

from heapq import heappop, heappush
from itertools import count, combinations

from .progress_bar import ProgressBar
from .graph import cypher, subgraph
from .utilities import pythagorean, haversine

def recover_paths(pair_volumes, pairs):

    paths = []
    flows = []

    for pair in pairs:

        origin = pair['origin']
        destination = pair['destination']
        demand = pair['demand']

        origin_virtual = f'{origin}_o'
        destination_virtual = f'{destination}_d'

        volumes = pair_volumes[(origin, destination)]

        queue = []
        c = count()
        
        heappush(queue, (next(c), origin_virtual, [origin_virtual], demand))

        while queue:
        
            _, source, path, flow = heappop(queue)
        
            total = sum([v for v in volumes[source].values()])
        
            for target, volume in volumes[source].items():
        
                if volume > 0:
        
                    new_path = path + [target]
                    new_flow = flow * volume / total
        
                    heappush(queue, (next(c), target, new_path, new_flow))
        
                    if target == destination_virtual:
        
                        paths.append(new_path)
                        flows.append(new_flow)

    return paths, flows

def recover_tree(graph, volumes, origin, flow):

    flows = {k: 0 for k in volumes.keys()}
    flows[origin] = flow

    nodes = []
    edges = []

    for source, outflows in volumes.items():
        for target, outflow in outflows.items():

            # print(source, target)

            if outflow > 0:
                # print(outflow, target)

                flows[target] += outflow

                edges.append((source, target, {'volume': outflow}))

    nodes = [(k, {**graph._node[k], 'volume': v}) for k, v in flows.items() if v > 0]
    # print(nodes)

    tree = nx.DiGraph()
    tree.add_nodes_from(nodes)
    # print(tree.number_of_nodes())

    tree.add_edges_from(edges)
    # print(tree.number_of_nodes())

    return tree

def recover_flow(graph, volumes, production, epsilon = 1e-10):

    flows = {k: 0 for k in graph.nodes}

    nodes = []
    edges = []

    for origin, flow in production.items():

        flows[origin] += flow

        for source, outflows in volumes[origin].items():
            for target, outflow in outflows.items():

                if outflow > epsilon:

                    flows[target] += outflow

                    edges.append(
                        (source, target, {**graph._adj[source][target], 'volume': outflow})
                        )
    
    nodes = [(k, {**graph._node[k], 'volume': v}) for k, v in flows.items() if v > epsilon]

    flow = nx.DiGraph()
    flow.add_nodes_from(nodes)
    flow.add_edges_from(edges)

    return flow

def flow_cost(flow, objective):

    cost = 0

    for source, adj in flow._adj.items():
        for target, edge in adj.items():

            cost += edge['volume'] * edge.get(objective, 1)

    return cost

class MOMD_FC():

    def __init__(self, graph, production, attraction, **kwargs):

        self.graph = graph # Must be all-pairs-graph

        # Dictionary containing trip production for all origins
        self.production = production

        # Dictionary containing trip attraction for all destinations from all origins
        self.attraction = attraction

        # Edge field to minimize (minimize volume * objective)
        self.objective = kwargs.get('objective', 'objective')

        # List of functions which allow edge to be used if True
        self.conditions = kwargs.get('conditions', [])

        # Cost multiplier applied for taking the direct path between an origin and
        # destination i.e. not using resupply stations - only applied where the direct
        # edge violates conditions.
        self.direct_path_penalty = kwargs.get('direct_path_penalty', 1.5)

        # Budget allowed for adding station capacity
        self.budget = kwargs.get('budget', np.inf)

        # Multipliers for objective function
        self.capacity_multiplier = kwargs.get('capacity_multiplier', 1e-9)
        self.cost_multiplier = kwargs.get('cost_multiplier', 1)

        self.build()

    def solve(self, **kwargs):

        #Generating the solver object
        solver = pyomo.SolverFactory(**kwargs)

        # Building and solving as a linear problem
        self.result = solver.solve(self.model)

        # Making solution dictionary
        self.collect_results()

    def collect_results(self, **kwargs):

        self.results = {}

        self.volumes = (
            {n: {n: {n: 0 \
            for n in self.graph.nodes} \
            for n in self.graph.nodes} \
            for n in self.places}
            )

        self.volumes_direct = (
            {n: {n: 0 \
            for n in self.graph.nodes} \
            for n in self.places}
            )

        for name in self.flow_names:

            value = list(getattr(self.model, name).extract_values().values())[0]

            if value is None:

                print(name, getattr(self.model, name).extract_values().values())

            self.results[name] = 0. if value is None else value

            parts = name.split(':')
            source = parts[0]
            target = parts[1]
            origin = parts[2]
            direct = parts[4] == 'direct_flow'

            self.volumes[origin][source][target] += 0. if value is None else value

            if direct:

                self.volumes_direct[origin][target] += 0. if value is None else value

        self.capacities = {n: 0 for n in self.graph.nodes}

        for name in self.capacity_names:

            value = list(getattr(self.model, name).extract_values().values())[0]

            self.results[name] = 0. if value is None else value

            parts = name.split(':')
            source = parts[0]

            self.capacities[source] += 0. if value is None else value

    def build(self):

        self._node = self.graph._node
        self._adj = self.graph._adj

        self.places = list(self.production.keys())
        self.stations = list(set(self.graph.nodes) - set(self.places))

        self.model = pyomo.ConcreteModel()

        self.model.direct_path_penalty = pyomo.Param(
            initialize = self.direct_path_penalty, mutable = True)
        self.model.budget = pyomo.Param(initialize = self.budget, mutable = True)

        cost = 0

        self.capacity_names = []
        expenditure = 0

        for station in self.stations:

            name = f'{station}::capacity'
            self.capacity_names.append(name)

            variable = pyomo.Var(domain = pyomo.NonNegativeReals, initialize = 0)
            setattr(self.model, name, variable)

            expenditure += getattr(self.model, name)
            cost += getattr(self.model, name) * self.capacity_multiplier

        name = f'budget_constraint'

        constraint = pyomo.Constraint(
            expr = expenditure <= self.model.budget
            )

        setattr(self.model, name, constraint)

        self.flow_names = []

        for origin in self.production.keys():

            for source, adj in self._adj.items():

                for target, edge in adj.items():

                    feasible = np.product([fun(edge) for fun in self.conditions])

                    if not feasible:

                        continue

                    approaching = (
                        self._adj[origin][target][self.objective] >
                        self._adj[origin][source][self.objective]
                        )

                    if not approaching:

                        continue

                    name = f'{source}:{target}:{origin}::flow'
                    self.flow_names.append(name)

                    variable = pyomo.Var(domain = pyomo.NonNegativeReals, initialize = 0)
                    setattr(self.model, name, variable)

                    cost += (
                        getattr(self.model, name) *
                        edge.get(self.objective, 0.) *
                        self.cost_multiplier
                        )

            for target, edge in self._adj[origin].items():

                name = f'{origin}:{target}:{origin}::direct_flow'
                self.flow_names.append(name)

                variable = pyomo.Var(domain = pyomo.NonNegativeReals, initialize = 0)
                setattr(self.model, name, variable)

                cost += (
                    getattr(self.model, name) *
                    edge.get(self.objective, 0.) *
                    self.cost_multiplier *
                    self.model.direct_path_penalty
                    )

        self.model.objective = pyomo.Objective(
            expr = cost, sense = pyomo.minimize
            )

        utilization = {k: 0 for k in self.graph.nodes}
        
        for origin, flow in self.production.items():

            net_flows = {k: 0 for k in self.graph.nodes}

            net_flows[origin] += flow

            for source, node in self._node.items():

                if source != origin:

                    net_flows[source] += self.attraction[origin].get(source, 0)

                name = f'{origin}:{source}:{origin}::direct_flow'
                flow = getattr(self.model, name, None)

                net_flows[origin] -= flow
                net_flows[source] += flow

                for successor in self.graph.successors(source):

                    name = f'{source}:{successor}:{origin}::flow'
                    flow = getattr(self.model, name, None)

                    if flow is not None:

                        net_flows[source] -= flow
                        utilization[source] += flow

                for predecessor in self.graph.predecessors(source):

                    name = f'{predecessor}:{source}:{origin}::flow'
                    flow = getattr(self.model, name, None)

                    if flow is not None:

                        net_flows[source] += flow
                        
            for source, node in self._node.items():

                name = f'{source}:{origin}::net_flow'

                constraint = pyomo.Constraint(
                    expr = net_flows[source] == 0
                    )

                setattr(self.model, name, constraint)

        for station in self.stations:

            if type(utilization[station]) == int:

                continue

            name = f'{station}::capacity'
            capacity = getattr(self.model, name)

            name = f'{station}::utilization'

            constraint = pyomo.Constraint(
                expr = utilization[station] <= capacity
                )

            setattr(self.model, name, constraint)

class MOMD():

    def __init__(self, graph, production, attraction, **kwargs):

        self.graph = graph # Must be all-pairs-graph
        self.production = production
        self.attraction = attraction

        self.build(**kwargs)

    def solve(self, **kwargs):

        #Generating the solver object
        solver = pyomo.SolverFactory(**kwargs)

        # Building and solving as a linear problem
        self.result = solver.solve(self.model)

        # Making solution dictionary
        self.collect_results()

    def collect_results(self, **kwargs):

        self.results = {}

        self.volumes = (
            {n: {n: {n: 0 \
            for n in self.graph.nodes} \
            for n in self.graph.nodes} \
            for n in self.places}
            )

        for name in self.var_names:

            value = list(getattr(self.model, name).extract_values().values())[0]

            self.results[name] = 0. if value is None else value

            parts = name.split(':')
            source = parts[0]
            target = parts[1]
            origin = parts[2]

            self.volumes[origin][source][target] += 0. if value is None else value

    def build(self, **kwargs):

        self.objective = kwargs.get('objective', 'objective')
        self.conditions = kwargs.get('conditions', [])
        self.penalty = kwargs.get('penalty', 1)

        self._node = self.graph._node
        self._adj = self.graph._adj

        self.places = list(self.production.keys())
        self.stations = list(set(self.graph.nodes) - set(self.places))

        self.model = pyomo.ConcreteModel()

        cost = 0

        self.var_names = []

        for origin in self.production.keys():

            for source, adj in self._adj.items():

                for target, edge in adj.items():

                    feasible = np.product([fun(edge) for fun in self.conditions])

                    if not feasible:

                        continue

                    approaching = (
                        self._adj[origin][target][self.objective] >
                        self._adj[origin][source][self.objective]
                        )

                    if not approaching:

                        continue

                    name = f'{source}:{target}:{origin}::flow'
                    self.var_names.append(name)

                    variable = pyomo.Var(domain = pyomo.NonNegativeReals)
                    setattr(self.model, name, variable)

                    cost += getattr(self.model, name) * edge.get(self.objective, 0.)

            for target, edge in self._adj[origin].items():

                name = f'{origin}:{target}:{origin}::direct_flow'
                self.var_names.append(name)

                variable = pyomo.Var(domain = pyomo.NonNegativeReals)
                setattr(self.model, name, variable)

                cost += (
                    getattr(self.model, name) * edge.get(self.objective, 0.) * self.penalty
                    )

        self.model.objective = pyomo.Objective(
            expr = cost, sense = pyomo.minimize
            )

        utilization = {k: 0 for k in self.graph.nodes}
        
        for origin, flow in self.production.items():

            net_flows = {k: 0 for k in self.graph.nodes}

            net_flows[origin] += flow

            for source, node in self._node.items():

                if source != origin:

                    net_flows[source] += self.attraction[origin].get(source, 0)

                name = f'{origin}:{source}:{origin}::direct_flow'
                flow = getattr(self.model, name, None)

                net_flows[origin] -= flow
                net_flows[source] += flow

                for successor in self.graph.successors(source):

                    name = f'{source}:{successor}:{origin}::flow'
                    flow = getattr(self.model, name, None)

                    if flow is not None:

                        net_flows[source] -= flow
                        utilization[source] += flow

                for predecessor in self.graph.predecessors(source):

                    name = f'{predecessor}:{source}:{origin}::flow'
                    flow = getattr(self.model, name, None)

                    if flow is not None:

                        net_flows[source] += flow
                        
            for source, node in self._node.items():

                name = f'{source}:{origin}::net_flow'

                constraint = pyomo.Constraint(
                    expr = net_flows[source] == 0
                    )

                setattr(self.model, name, constraint)

        for station in self.stations:

            if type(utilization[station]) == int:

                continue

            name = f'{station}::utilization'

            constraint = pyomo.Constraint(
                expr = utilization[station] <= self._node[station]['capacity']
                )

            setattr(self.model, name, constraint)

class SOMD():

    def __init__(self, graph, origin, demand, **kwargs):

        self.graph = graph # Must be all-pairs-graph
        self.origin = origin
        self.demand = demand

        self.build(**kwargs)

    def solve(self, **kwargs):

        #Generating the solver object
        solver = pyomo.SolverFactory(**kwargs)

        # Building and solving as a linear problem
        self.result = solver.solve(self.model)

        # Making solution dictionary
        self.collect_results()

    def collect_results(self, **kwargs):

        self.results = {}

        self.volumes = {n: {n: 0 for n in self.graph.nodes} for n in self.graph.nodes}

        for name in self.var_names:

            value = list(getattr(self.model, name).extract_values().values())[0]

            self.results[name] = 0. if value is None else value

            parts = name.split(':')
            source = parts[0]
            target = parts[1]

            self.volumes[source][target] += 0. if value is None else value

    def build(self, **kwargs):

        self.objective = kwargs.get('objective', 'objective')
        self.conditions = kwargs.get('conditions', [])
        self.penalty = kwargs.get('penalty', 1)

        self._node = self.graph._node
        self._adj = self.graph._adj


        self.model = pyomo.ConcreteModel()

        cost = 0

        self.var_names = []

        origin = self.origin

        for source, adj in self._adj.items():

            for target, edge in adj.items():

                feasible = np.product([fun(edge) for fun in self.conditions])

                if not feasible:

                    continue

                approaching = (
                    self._adj[origin][target][self.objective] >
                    self._adj[origin][source][self.objective]
                    )

                if not approaching:

                    continue

                name = f'{source}:{target}::flow'
                self.var_names.append(name)

                variable = pyomo.Var(domain = pyomo.NonNegativeReals)
                setattr(self.model, name, variable)

                cost += getattr(self.model, name) * edge.get(self.objective, 0.)

        for target, edge in self._adj[origin].items():

            name = f'{origin}:{target}::direct_flow'
            self.var_names.append(name)

            variable = pyomo.Var(domain = pyomo.NonNegativeReals)
            setattr(self.model, name, variable)

            cost += getattr(self.model, name) * edge.get(self.objective, 0.) * self.penalty

        self.model.objective = pyomo.Objective(
            expr = cost, sense = pyomo.minimize
            )

        out_flows = {k: 0 for k in self.graph.nodes}
        net_flows = {k: 0 for k in self.graph.nodes}

        for source, node in self._node.items():

            # out_flows[source] += self.demand[source]
            net_flows[source] += self.demand[source]

            name = f'{origin}:{source}::direct_flow'
            flow = getattr(self.model, name, None)
            # print(flow)
            net_flows[origin] -= flow
            net_flows[source] += flow

            for successor in self.graph.successors(source):

                name = f'{source}:{successor}::flow'
                flow = getattr(self.model, name, None)

                if flow is not None:

                    net_flows[source] -= flow
                    out_flows[source] += flow

            for predecessor in self.graph.predecessors(source):

                name = f'{predecessor}:{source}::flow'
                flow = getattr(self.model, name, None)

                if flow is not None:

                    net_flows[source] += flow
                    

        for source, node in self._node.items():

            name = f'{source}::net_flow'

            constraint = pyomo.Constraint(
                expr = net_flows[source] == 0
                )

            setattr(self.model, name, constraint)

        for source, node in self._node.items():

            if source == origin:

                continue

            if type(out_flows[source]) == int:

                continue

            node_type  = node.get('type', '')

            name = f'{source}::out_flow'

            constraint = pyomo.Constraint(
                expr = out_flows[source] <= node['capacity']
                )

            setattr(self.model, name, constraint)
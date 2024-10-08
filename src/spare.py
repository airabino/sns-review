def load_centrality(graph, all_paths):

    # _, _, all_paths = floyd_warshall(graph, fields, **kwargs)

    load = {k: 0 for k in graph.nodes()}

    total_paths = 0

    for paths in all_paths.values():
        # print(paths)
        for path in paths.values():

            total_paths += 1

            for node in path[1:-1]:
            # for node in path[1:-1]:

                load[node] += 1

    centrality = {k: v / total_paths for k, v in load.items()}

    return centrality

# def chargers_in_range(graph)

def in_range(x, lower, upper):

    return (x >= lower) & (x <= upper)

def super_quantile(x, p = (0, 1), n = 100):
    
    p_k = np.linspace(p[0], p[1], n)

    q_k = np.quantile(x, p_k)

    return np.nan_to_num(q_k.mean(), nan = np.inf)

def origins_destinations(graph, origins, destinations):

    for source, adj in graph._adj.items():
        for target, edge in adj.items():

            # Edges from a non-origin destination are disallowed
            if (source in destinations) and (source not in origins):

                edge['feasible'] = False

            # Edges to and origin are disallowed
            if target in origins:

                edge['feasible'] = False

    return graph

def shortest_paths(graph, origins, method = 'dijkstra', **kwargs):
    '''
    Return path costs, path values, and paths using Dijkstra's or Bellman's method

    Produces paths to each destination from closest origin

    Depends on an Objective object which contains the following four functions:

    values = initial() - Function which produces the starting values of each problem state
    to be applied to the origin node(s)

    values = infinity() - Function which produces the starting values for each non-origin
    node. The values should be intialized such that they are at least higher than any
    conceivable value which could be attained during routing.

    values, feasible = update(values, edge, node) - Function which takes current path state
    values and updates them based on the edge traversed and the target node and whether the
    proposed edge traversal is feasible. This function returns the values argument and a
    boolean feasible.

    values, savings = compare(values, approximation) - Function for comparing path state
    values with the existing best approximation at the target node. This function returns
    the values argument and a boolean savings.
    '''

    destinations = kwargs.get('destinations', list(graph.nodes))

    # graph = origins_destinations(graph.copy(), origins, destinations)

    if method == 'dijkstra':

        costs, values, paths = dijkstra(graph, origins, **kwargs)

    elif method == 'bellman':

        costs, values, paths = bellman(graph, origins, **kwargs)

    # return costs, values, paths

    costs_d = {}
    values_d = {}
    paths_d = {}

    for destination in np.intersect1d(list(costs.keys()), destinations):

        costs_d[destination] = costs[destination]
        values_d[destination] = values[destination]
        paths_d[destination] = paths[destination]

    return costs_d, values_d, paths_d

def all_pairs_shortest_paths(graph, origins, method = 'dijkstra', **kwargs):
    '''
    Return path costs, path values, and paths using Dijkstra's or Bellman's method

    Produces paths to each origin from each origin

    Depends on an Objective object which contains the following four functions:

    values = initial() - Function which produces the starting values of each problem state
    to be applied to the origin node(s)

    values = infinity() - Function which produces the starting values for each non-origin
    node. The values should be intialized such that they are at least higher than any
    conceivable value which could be attained during routing.

    values, feasible = update(values, edge, node) - Function which takes current path state
    values and updates them based on the edge traversed and the target node and whether the
    proposed edge traversal is feasible. This function returns the values argument and a
    boolean feasible.

    values, savings = compare(values, approximation) - Function for comparing path state
    values with the existing best approximation at the target node. This function returns
    the values argument and a boolean savings.
    '''

    if method == 'dijkstra':

        routing_function = dijkstra

    elif method == 'bellman':

        routing_function = bellman

    costs = {}
    values = {}
    paths = {}

    for origin in ProgressBar(origins, **kwargs.get('progress_bar_kw', {})):

        result = shortest_paths(
            graph, [origin],
            destinations = origins,
            method = method, 
            **kwargs
            )

        costs[origin] = result[0]
        values[origin] = result[1]
        paths[origin] = result[2]

    # costs, values, paths = floyd_warshall(graph, fields, **kwargs)

    return costs, values, paths

def gravity(values, origins = {}, destinations = {}, **kwargs):

    field = kwargs.get('field', 'total_time')
    expectation = kwargs.get('expectation', np.mean)
    constant = kwargs.get('constant', 1)
    adjustment = kwargs.get('adjustment', 1)


    if not origins:

        origins = {k: 1 for k in values.keys()}

    if not destinations:

        destinations = {k: 1 for k in values.keys()}

    sum_cost = 0

    sum_cost_0 = 0

    n = 0

    for origin, mass_o in origins.items():

        val_o = values[origin]

        for destination, mass_d in destinations.items():

            if origin != destination:

                # sum_cost += (
                #     constant * mass_o * mass_d /
                #     (
                #         expectation(
                #             np.atleast_1d(values[origin][destination][field])) / adjustment
                #         ) ** 2
                #     )

                # if values[origin][destination][field] == 0:
                #     print(origin, destination)

                sum_cost += (
                    constant * mass_o * mass_d / (
                        max([val_o[destination][field], 600]) / adjustment
                        ) ** 2
                    )

                # sum_cost_0 += (
                #     constant * mass_o * mass_d *
                #     max([val_o[destination]['driving_time'], 600])
                #     )

        n += 1
    
    # print(n)
    return sum_cost / (n ** 2)

def impedance(values, origins = {}, destinations = {}, **kwargs):

    field = kwargs.get('field', 'total_time')
    expectation = kwargs.get('expectation', np.mean)
    constant = kwargs.get('constant', 1)


    if not origins:

        origins = {k: 1 for k in values.keys()}

    if not destinations:

        destinations = {k: 1 for k in values.keys()}

    sum_cost = 0

    sum_cost_1 = 0

    n = 0

    for origin, mass_o in origins.items():

        val_o = values[origin]

        for destination, mass_d in destinations.items():

            if origin != destination:

                sum_cost += (
                    constant * mass_o * mass_d *
                    max([val_o[destination][field], 600])
                    )


            n += 1

    return sum_cost / n, sum_cost_0 / n

def specific_impedance(values, destinations = {}, **kwargs):

    field = kwargs.get('field', 'total_time')
    expectation = kwargs.get('expectation', np.mean)
    constant = kwargs.get('constant', 1)


    if not destinations:

        destinations = {k: 1 for k in values.keys()}

    sum_cost = 0

    n = 0

    for destination, mass_d in destinations.items():

        sum_cost += (
            constant * mass_d *
            expectation(np.atleast_1d(values[destination][field]))
            )

        n += 1

    return sum_cost / n

class Objective():

    def __init__(self, field = 'weight', edge_limit = np.inf, path_limit = np.inf):

        self.field = field
        self.edge_limit = edge_limit
        self.path_limit = path_limit

    def initial(self):

        return 0

    def infinity(self):

        return np.inf

    def update(self, values, link):

        edge_value = link.get(self.field, 1)

        values += edge_value

        return values, (values <= self.path_limit) and (edge_value <= self.edge_limit)

    def compare(self, values, approximation):

        return values, values < approximation

def edge_types(graph):

    _adj = graph._adj
    _node = graph._node

    for source, adj in _adj.items():
        for target, edge in adj.items():

            edge['type'] = (
                f"to_{_node[target].get('type', 'none')}"
                )

    return graph

def reserve_range(graph, node, n = 3):

    if graph._node[node].get('type', '') == 'place':

        return 0

    distances = [v['distance'] for v in graph._adj[node].values()]

    return np.sort(distances)[n + 1]

def supply_costs(graph, vehicle, station_kw):

    t0 = time.time()

    for source, node in graph._node.items():

        if node['type'] in station_kw:

            node['reserve_range'] = reserve_range(graph, source)

            kw = station_kw[node['type']]

            node['station'] = Station(node, **kw)

    for source, adj in graph._adj.items():

        for target, edge in adj.items():

            station = graph._node[source]['station']

            if station is not None:

                distance = edge['distance']
                effective_distance = distance + graph._node[target]['reserve_range']

                edge['distance'] = effective_distance
                edge['effective_distance'] = effective_distance

                edge = station.update(vehicle, edge)

                edge['distance'] = distance

    return graph

def expect_supply_costs(graph):
    
    for source, adj in graph._adj.items():

        for target, edge in adj.items():

            station = graph._node[source]['station']

            if station is not None:

                edge = station.expect(edge)

    return graph

class Vehicle():

    def __init__(self, **kwargs):

        self.cases = kwargs.get('cases', 1) # [-]

        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.consumption = kwargs.get('consumption', 550) # [J/m]
        self.power = kwargs.get('power', 80e3) # [W]

        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])
        self.max_charge_start_soc = kwargs.get('max_charge_start_soc', 1) # [-]
        self.linear_fraction = kwargs.get('linear_fraction', .8) # [-]

        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # ([-], [-])

        self.out_of_charge_penalty = kwargs.get('out_of_charge_penalty', 4 * 3600) # [s]

        self.cost = kwargs.get('cost', 'routing_time')

        if self.cases == 1:

            self.expectation = kwargs.get(
                'expectation',
                lambda x: x[0],
                )

        else:

            self.expectation = kwargs.get(
                'expectation',
                lambda x: super_quantile(x, self.risk_attitude),
                )
            
        self.initial_values = kwargs.get(
            'initial_values',
            {
                'total_time': np.zeros(self.cases), # [s]
                'driving_time': np.zeros(self.cases), # [s]
                'delay_time': np.zeros(self.cases), # [s]
                'charging_time': np.zeros(self.cases), # [s]
                'routing_time': np.zeros(self.cases), # [s]
                'distance': np.zeros(self.cases), # [m]
                'price': np.zeros(self.cases), # [$]
            },
        )

        self.usable_capacity = (
            (self.soc_bounds[1] - self.soc_bounds[0]) * self.capacity
            )

        self.range = self.usable_capacity / self.consumption

        self.min_edge_distance = (1 - self.max_charge_start_soc) * self.range

    def select_case(self, case):

        new_object = deepcopy(self)
        new_object.expectation = lambda x: x[min([len(x) - 1, case])]

        return new_object

    def initial(self):

        return self.initial_values

    def infinity(self):

        return {k: np.ones(self.cases) * np.inf for k in self.initial_values.keys()}

    def update(self, values, edge):

        updated_values = {}

        updated_values['total_time'] = values['total_time'] + edge['total_time']
        updated_values['routing_time'] = values['routing_time'] + edge['routing_time']
        updated_values['driving_time'] = values['driving_time'] + edge['time']
        updated_values['charging_time'] = values['charging_time'] + edge['charging_time']
        updated_values['delay_time'] = values['delay_time'] + edge['delay_time']
        updated_values['distance'] = values['distance'] + edge['distance']
        updated_values['price'] = values['price'] + edge['price']

        return updated_values, True

    def compare(self, values, approximation):

        values_expectation = self.expectation(values[self.cost])
        approximation_expectation  = self.expectation(approximation[self.cost])

        savings = values_expectation < approximation_expectation 

        return values_expectation, savings

    def edge_feasible(self, edge):

        feasible = edge['distance'] <= self.range

        return feasible

    def dc_charge(self, initial_soc, final_soc, power, capacity):

        final_soc = min([final_soc, .99])
        
        alpha = power / capacity / (1 - self.linear_fraction) # Exponential charging factor

        duration_linear = 0

        if self.linear_fraction > initial_soc:

            delta_soc_linear = min([final_soc, self.linear_fraction]) - initial_soc

            duration_linear = (
                delta_soc_linear * capacity / power
                )

        duration_exponential = 0

        if self.linear_fraction < final_soc:

            delta_soc_exponential = final_soc - max([initial_soc, self.linear_fraction])

            duration_exponential = (
                -np.log(
                    1 - delta_soc_exponential / (1 - self.linear_fraction)
                    ) / alpha
                )

        return duration_linear + duration_exponential

    def ac_charge(self, initial_soc, final_soc, power, capacity):
        
        duration_linear = (final_soc - initial_soc) * capacity / power

        return duration_linear

    def energy(self, station, edge):

        power = min([self.power, station.power])

        edge_energy = self.consumption * edge['distance']

        initial_soc = self.soc_bounds[0]

        final_soc = initial_soc + edge_energy / self.capacity

        feasible = self.edge_feasible(edge)

        if feasible:

            if station.type == 'ac':

                charge_duration = self.ac_charge(
                    initial_soc, final_soc, power, self.capacity
                    )

            elif station.type == 'dc':

                charge_duration = self.dc_charge(
                    initial_soc, final_soc, power, self.capacity
                    )

            return feasible, edge_energy, charge_duration

        else:

            if station.type == 'ac':

                charge_duration = self.ac_charge(
                    self.soc_bounds[0], self.soc_bounds[1], power, self.usable_capacity
                    ) + self.out_of_charge_penalty

            elif station.type == 'dc':

                charge_duration = self.dc_charge(
                    self.soc_bounds[0], self.soc_bounds[1], power, self.usable_capacity
                    ) + self.out_of_charge_penalty

            return feasible, edge_energy, charge_duration

class Station():

    def __init__(self, node = {}, **kwargs):

        self.seed = kwargs.get('seed', None)
        self.rng = kwargs.get('rng', np.random.default_rng(self.seed))

        self.type = kwargs.get('type', 'ac') # {'ac', 'dc'}

        # Private chargers do not generate queues, public ones do
        self.access = kwargs.get('access', 'private') # {'private', 'public'}

        self.cases = kwargs.get('cases', 1) # [-]

        self.reserve_range = kwargs.get('reserve_range', 0) # [-]

        # Parameters for charge events
        power_input = kwargs.get('power', np.inf)

        if type(power_input) == dict:

            
            self.power = self.rng.choice(
                power_input.get(node.get('network', ''), power_input['default'])
                )

        elif type(power_input) == float:

            self.power = power_input

        self.price = kwargs.get('price', .5 / 3.6e6) # [$/J]

        self.reliability = kwargs.get('reliability', 1)

        self.ports = kwargs.get('ports', node.get('n_dcfc', 1))

        self.usable_ports = sum(
            [self.rng.random() < self.reliability for idx in range(self.ports)]
            )

        self.setup_time = kwargs.get('setup_time', 0) # [s]

        self.queue_kw = kwargs.get('queue', {})

        self.rho_dist_bounds = kwargs.get('rho_dist_bounds', (2, 4))
        self.rho_dist_bounds_flip = np.flip(self.rho_dist_bounds)
        self.traffic = kwargs.get('traffic', .5)
        
        self.vehicle = None
        self.delay_time = None
        self.delay_time_expected = None

    def estimate(self):

        if self.access == 'public':

            if self.usable_ports > 0:

                # rho = np.linspace(*self.vehicle.risk_attitude, 100)
                rho = np.linspace((.001, .999), 100)
                # rho = np.random.rand(100)

                dist_args = (
                    np.interp(self.traffic, [0, 1], self.rho_dist_bounds),
                    np.interp(self.traffic, [0, 1], self.rho_dist_bounds_flip),
                    )

                rho = beta(*dist_args).rvs(size = 100, random_state = self.rng)
                # self.rho = rho
                # print(rho.mean(), rho)

                self.queue_time = queuing_time_distribution(
                    self.usable_ports, rho, self.power, **self.queue_kw,
                    ).rvs(size = self.cases, random_state = self.rng)

                if self.usable_ports != self.ports:

                    self.queue_time_nominal = queuing_time_distribution(
                        self.ports, rho, self.power, **self.queue_kw,
                        ).rvs(size = self.cases, random_state = self.rng)
                else:

                    self.queue_time_nominal = self.queue_time

            else:

                self.queue_time = np.inf
                self.queue_time_nominal = np.inf

        else:

            self.queue_time = 0
            self.queue_time_nominal = 0

        self.delay_time = np.zeros(self.cases) + self.queue_time + self.setup_time
        self.delay_time_expected = super_quantile(
            self.delay_time, self.vehicle.risk_attitude
            )

        self.delay_time_nominal = (
            np.zeros(self.cases) + self.queue_time_nominal + self.setup_time
            )
        self.delay_time_nominal_expected = super_quantile(
            self.delay_time_nominal, self.vehicle.risk_attitude
            )

    def update(self, vehicle, edge):

        if vehicle is not self.vehicle:

            self.vehicle = vehicle

            self.estimate()

        if self.vehicle.cases == 1:

            delay_time = self.delay_time_expected
            delay_time_nominal = self.delay_time_nominal_expected

        else:

            delay_time = self.delay_time
            delay_time_nominal = self.delay_time_nominal
        
        feasible, charge_energy, charge_duration = self.vehicle.energy(self, edge)

        if (self.power == np.inf) and feasible:

            charge_duration = 0
            delay_time = 0

        edge['feasible'] = feasible

        if feasible:

            edge['energy'] = charge_energy
            edge['charging_time'] = charge_duration
            edge['delay_time'] = delay_time
            edge['driving_time'] = edge['time']
            edge['total_time'] = edge['time'] + delay_time + charge_duration
            edge['routing_time'] = edge['time'] + delay_time_nominal + charge_duration
            edge['charge_event'] = int(charge_duration > 0)

        else:

            ratio = charge_energy / vehicle.usable_capacity

            edge['energy'] = charge_energy
            edge['charging_time'] = charge_duration * ratio
            edge['delay_time'] = delay_time * ratio
            edge['driving_time'] = edge['time']
            edge['total_time'] = edge['time'] + delay_time * ratio + charge_duration * ratio
            edge['routing_time'] = (
                edge['time'] + delay_time_nominal * ratio + charge_duration * ratio
                )
            edge['charge_event'] = int(charge_duration > 0) * np.ceil(ratio)

        return edge

    def expect(self, edge):

        edge['delay_time'] = self.delay_time_expected

        edge['total_time'] = (
            edge['time'] + self.delay_time_expected + edge['charging_time']
            )

        edge['routing_time'] = (
            edge['time'] + self.delay_time_nominal_expected + edge['charging_time']
            )


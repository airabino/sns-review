import time
import numpy as np

from .queue import Queue

class Station():

    default_queue = Queue()
    default_rng = np.random.default_rng()

    def __init__(self, **kwargs):

        # Objects
        self.rng = kwargs.get('rng', self.default_rng)
        self.queue = kwargs.get('queue', self.default_queue)

        # Station parameters
        self.size = kwargs.get('size', 1)
        self.reliability = kwargs.get('reliability', 1)
        self.power = kwargs.get('power', 80e3)
        self.energy = kwargs.get('energy', 30 * 3.6e6)
        self.setup_time = kwargs.get('setup_time', 0)
        self.cutoff = kwargs.get('cutoff', np.inf)
        self.saturated = kwargs.get('saturated', False)

        # Station position
        self.node = kwargs.get('node', None) # Pointer to node
        self.edges = kwargs.get('edges', []) # Pointers to edges

        # Queuing parameters
        self.service_rate = kwargs.get('service_rate', self.power / self.energy)
        self.arrival_rate = kwargs.get('arrival_rate', 0)
        self.functional = self.reliability * self.size
        
        # self.roll()

        self.update()

    def roll(self):

        self.functional = sum(
            self.reliability >= self.rng.random(size = self.size)
            )

    def update(self):

        if (self.functional < 1) | (self.power == 0):

            self.rho = 0
            self.waiting_time = np.inf
            self.capacity = 0
            self.saturated = True
        
        else:

            self.rho = self.arrival_rate / (self.service_rate * self.functional)

            self.waiting_time = self.queue(self.functional, self.rho) / self.service_rate

            if self.waiting_time > self.cutoff:

                self.waiting_time = self.cutoff
                self.saturated = True

            if self.cutoff == np.inf:

                self.capacity = np.inf

            else:

                rho = np.linspace(0, 1, 1000)
                w_q = self.queue(self.functional, rho)
                indices = ~np.isnan(w_q)

                self.capacity = np.interp(
                    self.cutoff, w_q[indices], rho[indices],
                    ) * self.service_rate * self.functional

        return self.waiting_time

    def compute(self, **kwargs):

        functional = kwargs.get('functional', self.functional)
        rho = kwargs.get('rho', self.rho)

        self.waiting_time = (
            self.queue(rho, functional) / self.service_rate + self.setup_time
            )

        return self.waiting_time

    def update_graph(self, vehicle):

        self.update_node()
        self.update_edges(vehicle)

    def update_node(self):

        self.node['capacity'] = self.capacity
        self.node['cutoff'] = self.cutoff
        self.node['saturated'] = self.saturated
        self.node['functional'] = self.functional
        self.node['service_rate'] = self.service_rate
        self.node['waiting_time'] = self.waiting_time

    def update_edges(self, vehicle):

        for edge in self.edges:

            energy, feasible = vehicle.energy(edge['distance'])

            if feasible:

                charging_time = vehicle.charge_duration(self, energy)
                penalty_time = 0

                edge['waiting_time'] = self.waiting_time
                edge['charging_time'] = charging_time
                edge['penalty_time'] = penalty_time

            else:
                
                charging_time = vehicle.charge_duration(self, vehicle.capacity_usable)
                charging_time += vehicle.charge_duration(
                    self, energy - vehicle.capacity_usable
                    )

                penalty_time = vehicle.out_of_charge_penalty

                edge['waiting_time'] = self.waiting_time
                edge['charging_time'] = charging_time
                edge['penalty_time'] = penalty_time


            total_time_charge = (
                    edge['time'] + self.waiting_time + charging_time
                    )

            total_time_tow = (
                edge['time'] + charging_time + penalty_time
                )

            edge['total_time'] = min([total_time_charge, total_time_tow])

class Place(Station):

    def update(self):

        self.waiting_time = 0
        self.size = 0
        self.service_rate = 0
        self.functional = 0
        self.saturated = True
        self.capacity = 0

    def update_edges(self, vehicle):

        for edge in self.edges:

            energy, feasible = vehicle.energy(edge['distance'])

            if feasible:

                penalty_time = 0

            else:

                penalty_time = vehicle.out_of_charge_penalty * 2

            edge['waiting_time'] = 0
            edge['charging_time'] = 0
            edge['penalty_time'] = penalty_time
            edge['total_time'] = edge['time'] + penalty_time

class Vehicle():

    def __init__(self, **kwargs):

        self.capacity = kwargs.get('capacity', 80 * 3.6e6)
        self.consumption = kwargs.get('consumption', 687.6)
        self.power = kwargs.get('power', 80e3)

        self.soc_max = kwargs.get('soc_max', .99)
        self.soc_min = kwargs.get('soc_min', .01)
        self.linear_fraction = kwargs.get('linear_fraction', .8)

        self.risk_attitude = kwargs.get('risk_attitude', .5)
        self.out_of_charge_penalty = kwargs.get('out_of_charge_penalty', 4 * 3600)

        self.soc_usable = self.soc_max - self.soc_min
        self.capacity_usable = self.soc_usable * self.capacity

        self.range = self.capacity_usable / self.consumption

    def roll(self):

        pass

    def energy(self, distance):

        energy = self.consumption * distance

        feasible = distance <= self.range

        return energy, feasible

    def charge_duration(self, station, energy):

        event_power = min([station.power, self.power])

        event_duration = self.dc_charge(
            self.soc_min, energy, event_power, self.capacity
            )

        return event_duration

    def dc_charge(self, initial_soc, energy, power, capacity):

        final_soc = min([(initial_soc * capacity + energy) / capacity, self.soc_max])
        
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

            # print(1 - delta_soc_exponential, (1 - self.linear_fraction))

            duration_exponential = (
                -np.log(
                    1 - delta_soc_exponential / (1 - self.linear_fraction)
                    ) / alpha
                )



        return duration_linear + duration_exponential

    def ac_charge(self, initial_soc, energy, power, capacity):

        final_soc = (initial_soc * capacity + energy) / capacity

        duration_linear = (final_soc - initial_soc) * capacity / power

        return duration_linear, feasible

class RandomVehicle(Vehicle):

    default_rng = np.random.default_rng()

    def __init__(self, **kwargs):

        self.rng = kwargs.get('rng', self.default_rng)

        self.capacity_gen = kwargs.get(
            'capacity_gen', lambda: self.rng.normal(80, 20) * 3.6e6
            )

        self.power_gen = kwargs.get(
            'power_gen', lambda: self.rng.normal(80, 20) * 1e3
            )

        self.roll()

        self.consumption = kwargs.get('consumption', 550)

        self.soc_max = kwargs.get('soc_max', .99)
        self.soc_min = kwargs.get('soc_min', .01)
        self.linear_fraction = kwargs.get('linear_fraction', .8)

        self.risk_attitude = kwargs.get('risk_attitude', .5)
        self.out_of_charge_penalty = kwargs.get('out_of_charge_penalty', 4 * 3600)

        self.soc_usable = self.soc_max - self.soc_min
        self.capacity_usable = self.soc_usable * self.capacity

        self.range = self.capacity_usable / self.consumption

    def __call__(self):

        self.roll()

    def roll(self):

        self.capacity = self.capacity_gen()
        self.power = self.power_gen()

def random_station(rng, functions, **kwargs):

    for key, function in functions.items():

        kwargs[key] = function()

    return Station(**kwargs)

def reset_stations(graph, vehicle):

    for source, node in graph._node.items():

        node['object'].arrival_rate = 0

        node['object'].update()
        node['object'].update_graph(vehicle)

    return graph

def update_stations(graph, flows, all_paths, vehicle):

    for source in all_paths.keys():
        for target in all_paths[source].keys():

            flow = flows[source][target]
            path = all_paths[source][target]

            for node in path[1:-1]:

                graph._node[node]['object'].arrival_rate = (
                    graph._node[node]['object'].arrival_rate + 
                    flow
                    )

    for source, node in graph._node.items():

        node['object'].update()
        node['object'].update_graph(vehicle)

    return graph
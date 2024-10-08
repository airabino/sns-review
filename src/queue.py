import time
import numpy as np

from scipy.special import factorial
from scipy.interpolate import RegularGridInterpolator

def mmc_queue(arrival_rate, service_rate, servicers, cutoff = np.inf):

    arrival_rate = np.atleast_1d(arrival_rate)
    service_rate = np.atleast_1d(service_rate)

    rho = arrival_rate / (service_rate * servicers)

    probability_empty_denominator = 0

    for k in range(servicers):

        probability_empty_denominator += (servicers * rho) ** k / factorial(k)

    probability_empty_denominator += (
        (servicers * rho) ** servicers / factorial(servicers) / (1 - rho)
        )

    probability_empty = 1 / probability_empty_denominator

    waiting_time = (
        probability_empty * rho * (servicers * rho) ** servicers /
        (arrival_rate * (1 - rho) ** 2 * factorial(servicers))
        )

    waiting_time[rho == 0] = 0
    waiting_time[rho >= 1] = cutoff
    waiting_time[np.isnan(rho)] = cutoff

    return waiting_time

class Queue():

    def __init__(self, **kwargs):

        self.rho = kwargs.get('rho', np.linspace(0, 1, 100))
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', list(range(1, 101)))
        self.cutoff = kwargs.get('cutoff', np.inf)

        self.build()

    def __call__(self, c, rho):

        result = self.interpolate((c, rho))
        result[result > self.cutoff] = self.cutoff

        return result

    def build(self):

        self.waiting_times = np.zeros((len(self.c), len(self.rho)))

        for idx, c in enumerate(self.c):

            arrival_rate = self.rho * c * self.m

            self.waiting_times[idx] = mmc_queue(
                arrival_rate, self.m, c, cutoff = self.cutoff,
                )

        self.interpolate = RegularGridInterpolator(
            (self.c, self.rho), self.waiting_times,
            bounds_error = False, fill_value = np.inf,
            )
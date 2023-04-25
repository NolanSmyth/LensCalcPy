# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_stats.ipynb.

# %% auto 0
__all__ = ['power_law', 'likelihood', 'generate_observed_counts_with_bump']

# %% ../nbs/06_stats.ipynb 3
import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import chi2

# %% ../nbs/06_stats.ipynb 5
# Define the power law function with index -2
def power_law(x, a=1, index=-2):
    return a * (x**index)

# Define the likelihood function for the Poisson process
def likelihood(observed_counts, expected_counts):
    poisson_pmf = poisson.pmf(observed_counts, expected_counts)
    return np.prod(poisson_pmf)

def generate_observed_counts_with_bump(bin_centers, a, index, bump_position, bump_height):
    power_law_counts = power_law(bin_centers, a=a, index=index)
    closest_bin = np.argmin(np.abs(bin_centers - bump_position))
    power_law_counts[closest_bin] += bump_height
    return np.random.poisson(power_law_counts)

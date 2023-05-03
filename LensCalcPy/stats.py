# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_stats.ipynb.

# %% auto 0
__all__ = ['power_law', 'likelihood', 'generate_observed_counts_with_bump', 'neg_log_likelihood', 'get_MLE_params', 'llr_test',
           'get_observed_counts']

# %% ../nbs/06_stats.ipynb 3
import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import differential_evolution
from .survey import *
from .parameters import *

# %% ../nbs/06_stats.ipynb 6
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

# %% ../nbs/06_stats.ipynb 8
# Define the negative log-likelihood function for optimization with continuous bump_position
def neg_log_likelihood(params, bin_centers, observed_counts):
    a, index, bump_position, bump_height = params
    expected_counts = power_law(bin_centers, a=a, index=index)
    closest_bin = np.argmin(np.abs(bin_centers - bump_position))
    expected_counts[closest_bin] += bump_height
    return -np.sum(poisson.logpmf(observed_counts, expected_counts))

def get_MLE_params(bin_centers, observed_counts):
    
    # Set bounds for the parameters
    bounds = [(1, 300), (-4, -1), (min(bin_centers), max(bin_centers)), (0, 200)]

    # Use differential evolution to optimize the parameters
    result_bump = differential_evolution(neg_log_likelihood, bounds=bounds, args=(bin_centers, observed_counts), strategy='best1bin', popsize=10, tol=1e-4)
    optimized_params_bump = result_bump.x
    return optimized_params_bump


# %% ../nbs/06_stats.ipynb 10
def llr_test(bin_centers, observed_counts):

    optimized_params = get_MLE_params(bin_centers, observed_counts)
    expected_counts_opt = generate_observed_counts_with_bump(bin_centers, *optimized_params)

    # Calculate the likelihood ratio test statistic

    #log likelihood of the null hypothesis (0 bump height)
    ll_null = -neg_log_likelihood([optimized_params[0], optimized_params[1], optimized_params[2], 0], bin_centers, observed_counts)
    #log likelihood of the alternative hypothesis (non-zero bump height)
    ll_alt = -neg_log_likelihood(optimized_params, bin_centers, observed_counts)

    llr = 2 * (ll_alt - ll_null)

    return llr
    

# %% ../nbs/06_stats.ipynb 17
def get_observed_counts(lens_masses, bin_edges):
    observed_counts, _ = np.histogram(lens_masses, bins=bin_edges)
    return observed_counts

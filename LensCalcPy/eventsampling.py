# from LensCalcPy.parameters import *
from LensCalcPy.utils import *
from LensCalcPy.lens import *
from LensCalcPy.pbh import *
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# import scipy.integrate
# import numba
import pickle
# import tqdm
import emcee
from bisect import bisect_left


def sample_density(params, # galactic longitude (degrees)
                 # b, # galactic latitide (degrees)
                 # dl,  # lens distance from Earth (kpc)
                 # ds,  # source distance from Earth (kpc)
                 this_pbh, # LensCalcPy.pbh object
                 lbounds=(-10,10),
                 bbounds=(-10,10),
                 #umin=.5 # minimum impact parameter - u=2 ~50 mmag
):
    """
    Compute density of microlensing event space in differential volume.

    Parameters
    ----------
    params : np.array
       l  - galactic longitude (degrees)
       b  - galactic latitide (degrees) 
       dl - lens distance from Earth (kpc)
       ds - source distance from Earth (kpc)
       umin - minimum impact parameter
       crossing time - timescale of microlensing event

    this_pbh : LensCalcPy.Pbh
        Wrapper object with function to compute event rate.
        Member variables are overwritten!

    lbounds : tuple(float, float)
        bounds on galactic longitude in degrees
    bbounds : tuple(float, float)
        bounds on galactic latitude in degrees

    Returns
    -------
    float
        Event rate in (hours)**-(2) * (kpc)**(-2) * (degrees)**(-2)
    """

    this_pbh.l, this_pbh.b, dl, this_pbh.ds, umin, crossing_time = params
    if this_pbh.l < lbounds[0] \
    or this_pbh.l > lbounds[1] \
    or this_pbh.b > bbounds[1] \
    or this_pbh.b < bbounds[0] \
    or dl < 0 or dl > this_pbh.ds \
    or umin <= 0 \
    or crossing_time <= 0:
        return 0
    prob = this_pbh.differential_rate_integrand(umin ,dl, crossing_time, this_pbh.mw_model)
    # multiply by expected number of stars in volume element 
    expected_stars = (np.pi/180)**2 * this_pbh.ds**2 * np.cos(this_pbh.b*np.pi/180) * \
            this_pbh.mw_model.density_stars(this_pbh.ds, \
                                            this_pbh.l,  \
                                            this_pbh.b) 
    prob*=expected_stars   
    if prob < 0 or np.isnan(prob):
        print(f"error: expected stars = {expected_stars}")
        print(f"density: {this_pbh.mw_model.density_stars(this_pbh.ds, this_pbh.l, this_pbh.b) }")
        print(f"cos(b): {np.cos(this_pbh.b)}")
        return 0
    return prob

def sample_density_log(params,
                 lbounds=(-10,10),
                 bbounds=(-10,10)
):
    """
        Computes log of sample_density (see above)
    """
    return np.log(sample_density(params, this_pbh, lbounds, bbounds))

def coord_to_bin_indices(edges, coords):
    return tuple(bisect_left(edges[:,icoord], coords[icoord]) for icoord in range(len(coords)))

def box_from_indices(edges, indices):
    return np.array([[edges[idx-1][i],edges[idx][i]] for i, idx in enumerate(indices)])

def region_volume(region):
    return np.product(region[:,1]-region[:,0])

def mc_integrate(f, region, nsamples=10000):
    '''
        Perform uniform Monte Carlo integral of f
        over len(region) dimensions.

        Returns
        -------
        - float : integral
        - float : statistical error on the integral
    '''
    sample_points = np.random.random(size=(nsamples, len(region)))
    sample_points = (region[:,1]-region[:,0])*sample_points+region[:,0]
    samples_values = [f(*_) for _ in sample_points]
    return np.mean(samples_values)*region_volume(region), np.std(samples_values)*region_volume(region)/np.sqrt(nsamples)

def grab_initial_states_from_pkl(pbh,
                                 nwalkers,
                                 pickled_events_file = 'eventsamples.pkl'
):
    with open(pickled_events_file, 'rb') as eventsamples:
        exsamples = pickle.load(eventsamples)
    p0 = exsamples[np.random.choice(exsamples.shape[0], 
                                    size=nwalkers,
                                    replace=False)]
    # ensure initial states have nonzero probability
    for i,p in enumerate(p0):
        this_prob = sample_density_log(p, pbh)
        while np.isinf(this_prob):
            p0[i] = exsamples[np.random.choice(exsamples.shape[0], size=1)]
            this_prob = sample_density_log(p, pbh)
    return p0

def generate_events(this_pbh,
                    nsteps = 200000,
                    ndims = 6,
                    nwalkers = 12,
                    lbounds = (-90,90),
                    bbounds = (-45,45)
):

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndims,
                                    sample_density_log, 
                                    args=[this_pbh,lbounds,bbounds])

    p0 = grab_initial_states_from_pkl(nwalkers)
    state = sampler.run_mcmc(p0,nsteps)
    return sampler.get_chain(flat=True, discard=10000)
# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_utils.ipynb.

# %% auto 0
__all__ = ['ut_interp', 'm_low_interp', 'm_high_interp', 'dist_mw', 'dist_m31', 'density_mw', 'density_m31', 'mass_enclosed_mw',
           'mass_enclosed_m31', 'velocity_dispersion_mw', 'velocity_dispersion_m31', 'dist', 'einstein_rad',
           'velocity_radial', 'get_primed_coords', 'scientific_format', 'w_func', 'rho_func', 'magnification',
           'magnification_wave', 'displacement', 'integrand_polar_wave', 'integrand_polar', 'magnification_finite_wave',
           'magnification_finite', 'u_t_finite', 'u_t_finite_wave', 'make_ut_interp']

# %% ../nbs/04_utils.ipynb 3
from .parameters import *
import numpy as np
from numpy import pi
from scipy.integrate import quad, nquad
from scipy.optimize import brentq
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.interpolate import interp2d
import pickle
from .interpolations.interps import ut_interp

import matplotlib.pyplot as plt

from fastcore.test import *

# %% ../nbs/04_utils.ipynb 4
#Put 0 indent assignments so that variables will be in __all__
ut_interp = ut_interp

# %% ../nbs/04_utils.ipynb 5
def dist_mw(d: float, # distance from the Sun in kpc
            ) -> float: #distance to the MW center in kpc
    return np.sqrt(d**2 + rEarth**2 - 2*d*rEarth*np.cos(np.radians(l))*np.cos(np.radians(b)))

def dist_m31(d: float, # distance from the Sun in kpc
             ) -> float: #distance to the M31 center in kpc
            return dsM31 - d

def density_mw(r: float, # distance to MW center in kpc
                ) -> float: # DM density in Msun/kpc^3
    return rhoc / ((r/rs) * (1 + r/rs)**2)

def density_m31(r: float, # distance to M31 center in kpc
                ) -> float: # DM density in Msun/kpc^3
    return rhocM31 / ((r/rsM31) * (1 + r/rsM31)**2)

def mass_enclosed_mw(r: float  # distance to MW center in kpc
                      ) -> float : # enclosed DM mass in Msun
    return 4*pi * rhoc * rs**3 * (np.log(1 + r/rs) - (r/rs)/(1 + r/rs))

def mass_enclosed_m31(r: float  # distance to M31 center in kpc
                        ) -> float : # enclosed DM mass in Msun
    return 4*pi * rhocM31 * rsM31**3 * (np.log(1 + r/rsM31) - (r/rsM31)/(1 + r/rsM31))

def velocity_dispersion_mw(r: float, # distance from the MW center in kpc
                        ) -> float: # velocity dispersion in km/s
    if r == 0:
        return 0
    return np.sqrt(G * mass_enclosed_mw(r) / r) 

def velocity_dispersion_m31(r: float, # distance from the M31 center in kpc
                        ) -> float: # velocity dispersion in km/s
    if r == 0:
        return 0
    return np.sqrt(G * mass_enclosed_m31(r) / r)

def dist(d: float, # distance from the Sun in kpc
         ds: float = ds, # distance to the source in kpc
         ) -> float: #weighted lensing distance in kpc
    if d > ds:
         raise ValueError("Distance of lens must be less than source distance but d = " + str(d) + " and ds = " + str(ds))
    return d * (1 - d/ds)

def einstein_rad(d: float, # distance from the Sun in kpc
                 mass: float, # mass of the lens in Msun
                 ds: float = ds, # distance to the source in kpc
                 ) -> float: # Einstein radius in kpc
    return (4 * G * mass * dist(d, ds)/c**2)**(1/2)

def velocity_radial(d: float, # distance from the Sun in kpc
                    mass: float, # mass of the lens in Msun
                    umin: float, # minimum impact parameter
                    t: float, # crossing time in hours
                    ut: float, # threshold impact parameter
                    ) -> float: # radial velocity in km/s
    return 2*einstein_rad(d, mass) * (ut**2 - umin**2)**(1/2) / t * kpctokm

# from below 16 of https://iopscience.iop.org/article/10.3847/1538-4357/ac07a8/pdf*)
# alphabar = 27 Degrees xp-axis is aligned with the major axis
# of the Galactic bar,where \[Alpha]bar=27\[Degree] is applied as the bar angle.
# galactocentric coordniates x', y', z' as function of d, distance from Sun

def get_primed_coords(d: float, # distance from Sun in km
                      l: float = l, # galactic longitude in degrees
                      b: float = b, # galactic latitude in degrees
                      )-> tuple:
    """Get galactocentric coordinates x', y' given galactic latitude and longitude l, b, and distance d
    """
    # convert angles from degrees to radians
    l_rad = np.deg2rad(l)
    b_rad = np.deg2rad(b)
    alpha_rad = np.deg2rad(alphabar)

    # calculate unrotated Cartesian coordinates
    x_unrot = rEarth - d * np.cos(b_rad) * np.cos(l_rad)
    y_unrot = d * np.cos(b_rad) * np.sin(l_rad)

    # rotate the coordinates
    x_prime = x_unrot * np.cos(alpha_rad) - y_unrot * np.sin(alpha_rad)
    y_prime = x_unrot * np.sin(alpha_rad) + y_unrot * np.cos(alpha_rad)

    z_prime = d * np.sin(b_rad)

    return x_prime, y_prime, z_prime

def scientific_format(x, pos):
    """
    Formats a number in scientific notation in latex
    """
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

# %% ../nbs/04_utils.ipynb 9
# Add finite size calculation following https://arxiv.org/pdf/1905.06066.pdf

# Compute 'w' parameter given the mass of the primordial black hole and the wavelength
def w_func(m_pbh, lam):
    return 5.98 * (m_pbh / 1e-10) * (lam / 6210)**(-1)

# Compute 'rho' parameter given the mass of the primordial black hole and the lens distance
def rho_func(m_pbh, dl, ds):
    if dl >= ds:
        raise ValueError("dl must be less than ds to prevent division by zero.")
    if dl == 0:
        return 1e-2
    x = dl / ds
    return 5.9 * (m_pbh / 1e-10)**(-1/2) * (x / (1-x))**(1/2)

# Compute magnification given the impact parameter 'u'
def magnification(u):
        if u == 0:
            return np.inf
        else:
            return (u**2 + 2) / (u * (u**2 + 4)**0.5)

# Compute magnification in the wave optics regime
def magnification_wave(w, u):
    # Note this is taking the maximum value of the wave optics magnification
    # In reality, need to evaluate the hypergeometric function
    return np.minimum(magnification(u), np.pi * w / (1 - np.exp(-np.pi * w)))

# Compute displacement given 'x', 'y', and 'u'
def displacement(x, y, u):
    return ((x - u)**2 + y**2)**0.5

# Compute integrand in polar coordinates
def integrand_polar_wave(r, theta, w, u):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return magnification_wave(w, displacement(x, y, u)) * r

def integrand_polar(r, theta, u):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return magnification(displacement(x, y, u)) * r

# Compute finite magnification
def magnification_finite_wave(m_pbh, lam, u, dl, ds):
    w = w_func(m_pbh, lam)
    rho = rho_func(m_pbh, dl, ds)
    integrand = lambda r, theta: integrand_polar_wave(r, theta, w, u)
    result, _ = nquad(integrand, [[0, rho], [0, 2 * pi]])
    return result / (pi * rho**2)

def magnification_finite(m_pbh, u, dl, ds):
    rho = rho_func(m_pbh, dl, ds)
    integrand = lambda r, theta: integrand_polar(r, theta, u)
    result, _ = nquad(integrand, [[0, rho], [0, 2 * pi]])
    return result / (pi * rho**2)

# Compute 'u' at threshold
def u_t_finite(m_pbh, dl, ds):
    A_thresh = 1.34
    func = lambda u: magnification_finite(m_pbh, u, dl, ds) - A_thresh
    u_min = 0
    u_max = 10

    try:
        return brentq(func, u_min, u_max)
    except ValueError:
        return 0
    
def u_t_finite_wave(m_pbh, lam, dl, ds):
    A_thresh = 1.34
    func = lambda u: magnification_finite_wave(m_pbh, lam, u, dl, ds) - A_thresh
    u_min = 0
    u_max = 10

    try:
        return brentq(func, u_min, u_max)
    except ValueError:
        return 0

m_low_interp = 1e-15
m_high_interp = 1e0

def make_ut_interp(n_points=40, ds = 770):
    d_arr = np.linspace(0, ds, n_points)
    m_arr = np.logspace(np.log10(m_low_interp), np.log10(m_high_interp), n_points) #solar masses

    def calc_ut_arr(m):
    # Calculate ut_arr for the current m
        return np.array([u_t_finite(m, lam, d, ds) for d in d_arr])
    
    with Pool() as p:
        ut_values = list(p.map(calc_ut_arr, m_arr))

    # Convert ut_values to a 2D array
    ut_values = np.array(ut_values)
    
    # Create the 2D interpolation table
    ut_interp = interp2d(d_arr, m_arr, ut_values)
    return ut_interp

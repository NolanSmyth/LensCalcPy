# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_ffp.ipynb.

# %% auto 0
__all__ = ['Ffp']

# %% ../nbs/01_ffp.ipynb 3
from .parameters import *
from .utils import *
from .lens import *
from .galaxy import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.integrate import quad, nquad, dblquad, tplquad
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import minimize_scalar
import pickle
import functools
from pathos.multiprocessing import ProcessingPool as Pool

import functools

from fastcore.test import *
from tqdm import tqdm


# %% ../nbs/01_ffp.ipynb 5
class Ffp(Lens):
    """A class to represent a PBH population"""

    def __init__(self,
                p: float = 1, # Mass function power law index
                m_min: float = 1e-15, # Minimum mass in Msun
                m_max: float = 1e-3, # Maximum mass in Msun
                mw_model: MilkyWayModel = None,
                m31_model: M31Model = None, 
                l = None, # Galactic longitude
                b = None, # Galactic latitude
                u_t = 1, #threshold impact parameter in point-source limit
                ds = 770,
                ):
        """
        Initialize the PBH population
        """

        self.ut_interp = ut_func_new

        self.p = p
        #Define range of power law we want to consider
        self.m_min = m_min
        self.m_max = m_max
        self.M_norm = 1 #solar mass
        self.Z = self.pl_norm()
        self.Zprime = self.dN_dM_norm()

        # Instantiate or use existing MilkyWayModel and M31Model
        self.mw_model = mw_model or MilkyWayModel(mw_parameters)
        self.m31_model = m31_model or M31Model(m31_parameters)

        if l is None:
            raise ValueError("Galactic longitude must be specified")
        if b is None:
            raise ValueError("Galactic latitude must be specified")
        self.l = l
        self.b = b
        self.u_t = u_t
        self.ds = ds

    
    def __str__(self) -> str:
        return f"FFP with power law dN / dlogM ~ m^-{self.p}"
    __repr__ = __str__

    def dN_dM(self, M, M_norm, p):
        return (M/M_norm)**(-p-1) / M_norm
    
    def dN_dM_wrapper(self, M):
        return self.dN_dM(M, self.M_norm, self.p)

    def dN_dM_norm(self):
        return 1/abs(nquad(self.dN_dM_wrapper,[[self.m_min, self.m_max]], opts={'points': [self.m_min, self.m_min*1e3, self.m_min*1e6, self.m_max]})[0])
    
    def f_m(self, M):
        return self.dN_dM_wrapper(M)*self.Zprime

    def dN_dlogM(self, A, log10M, M_norm, p):
        M = 10**log10M
        return A * (M/self.M_norm)**-p
    
    def dN_dlogM_wrapper(self, M):
        return self.dN_dlogM(1, M, self.M_norm, self.p)
    
    def pl_norm(self):
        return 1/abs(nquad(self.dN_dlogM_wrapper,[[np.log10(self.m_min), np.log10(self.m_max)]], opts={'points': [np.log10(self.m_min), np.log10(self.m_min*1e3), np.log10(self.m_max*1e3)]})[0])

    def mass_func(self, log10m):
        #M_norm = 1 solar mass for now. This is dN/dlogM
        m = 10**log10m
        return self.Z * (m/1)**-self.p

    def differential_rate_integrand(self, umin, d, t, mf, model, finite=False, v_disp=None, t_e=False, tmax=np.inf, tmin=0):
        r = model.dist_center(d, self.l, self.b)
        ut = self.umin_upper_bound(d, mf) if finite else self.u_t
        if ut <= umin:
            return 0
        if t_e: 
            #Calculate radial velocity in terms of einstein crossing time
            v_rad = einstein_rad(d, mf, self.ds) * kpctokm / (t * htosec) 
            #crossing duration determined by extent of source in extreme finite limit
            t_duration = max(ut, self.umin_upper_bound(d, mf)) * 2 * einstein_rad(d, mf) * kpctokm / v_rad / htosec #event duration in hours

            if t_duration > tmax or t_duration < tmin:
                return 0     
        else:
            #Calculate radial velocity in terms of event duration (t_fwhm)
            v_rad = velocity_radial(d, mf, umin, t * htosec, ut) 
        if v_disp is None: 
            v_disp = model.velocity_dispersion_stars(r)
        return 2 * (1 / (ut**2 - umin**2)**0.5 *
                        #For FFP number density, use stellar density for 1 solar mass stars
                model.density_stars(d, self.l, self.b) / (1 * v_disp**2) *  
                v_rad**4 * (htosec / kpctokm)**2 *
                np.exp(-(v_rad**2 / v_disp**2)) *
                1)

    def differential_rate(self, t, integrand_func, finite=False, epsabs = 1.49e-08, epsrel = 1.49e-08):

        def inner_integrand(u, d, m):
            return integrand_func(u, d, m, t)
            
        # Second integral (over u) - bounds given by d
        def second_integral(d, m):
            point = self.sticking_point(m)

            if finite:
                u_min, u_max = 0, self.umin_upper_bound(d, m)
            else:
                u_min, u_max = 0, self.u_t

            result, error = quad(inner_integrand, u_min, u_max, args=(d, m), epsabs=epsabs, epsrel=epsrel, points=[point])

            return result
            
        # Third integral (over d)
        def third_integral(m):
            if finite:
                d_min, d_max = 0, self.d_upper_bound(m)
            else:
                d_min, d_max = 0, self.ds

            result, error = quad(second_integral, d_min, d_max, args=(m,), epsabs=epsabs, epsrel=epsrel)
            return result
        
        # Outermost integral (over m in log scale)
        def integrand_logm(logm):
            m = 10**logm
            result = third_integral(m)
            return result * ((m/self.M_norm) ** -self.p)  # multiply by mass function. This is for dN/dlogM

        logm_min = np.log10(self.m_min)
        logm_max = np.log10(self.m_max)

        result, error = quad(integrand_logm, logm_min, logm_max, epsabs=epsabs, epsrel=epsrel)

        result *= self.Z  # normalization
        return result

    
    def differential_rate_mw(self, t, finite=True, v_disp=None, t_e=False, epsabs = 1.49e-08, epsrel = 1.49e-08, tmax=np.inf, tmin=0):
        def integrand_func(umin, d, mf, t):
            return self.differential_rate_integrand(umin, d, t, mf, self.mw_model, finite=finite, v_disp=v_disp, t_e=t_e, tmax=tmax, tmin=tmin)
        return self.differential_rate(t, integrand_func, finite=finite, epsabs=epsabs, epsrel=epsrel)

    def differential_rate_m31(self, t, finite=True, v_disp=None, epsabs = 1.49e-08, epsrel = 1.49e-08,):
        def integrand_func(umin, d, mf, t):
            return self.differential_rate_integrand(umin, d, t, mf, self.m31_model, finite=finite, v_disp=v_disp)
        return self.differential_rate(t, integrand_func, finite=finite, epsabs=epsabs, epsrel=epsrel)

    def differential_rate_mw_mass(self, m, finite=True, v_disp=None, tcad=0.07, tobs=3, epsabs = 1.49e-08, epsrel = 1.49e-08, efficiency=None, monochromatic=False):
        def integrand_func(umin, d, t, mf):
            return self.differential_rate_integrand(umin, d, t, mf, self.mw_model, finite=finite, v_disp=v_disp)
        return self.differential_rate_mass(m, integrand_func, finite=finite, tcad=tcad, tobs=tobs, epsabs = epsabs, epsrel = epsrel, efficiency=efficiency, monochromatic=monochromatic)
    
    def differential_rate_m31_mass(self, m, finite=True, v_disp=None, tcad=0.07, tobs=3, epsabs = 1.49e-08, epsrel = 1.49e-08, efficiency=None, monochromatic=False):
        def integrand_func(umin, d, t, mf):
            return self.differential_rate_integrand(umin, d, t, mf, self.m31_model, finite=finite)
        return self.differential_rate_mass(m, integrand_func, finite=finite, tcad=tcad, tobs=tobs, epsabs = epsabs, epsrel = epsrel, efficiency=efficiency, monochromatic=monochromatic)

    def umin_upper_bound(self, d, m):
        rho = rho_func(m, d, self.ds)
        return self.ut_interp(rho, magnification(self.u_t))

    def d_upper_bound(self, m):
        #Determine upper limit for d otherwise for low masses, the contribution which only comes from d << 1 gets missed
        d_arr = np.logspace(-3, np.log10(self.ds*0.99), 100)
        for d in d_arr:
            if self.umin_upper_bound(d, m) == 0:
                return d
        return self.ds
    
    def sticking_point(self,m):
        #Determine where u_t is maximized. This speeds up the integral in m31
        result = minimize_scalar(lambda d:-self.umin_upper_bound(d, m), bounds=(0, self.ds), method='bounded')
        if result.success:
            return result.x[0] if isinstance(result.x, (list, np.ndarray)) else result.x
        else:
            return self.ds
    
    def differential_rate_total(self, t, finite=False):
        return self.differential_rate_mw(t, finite=finite) + self.differential_rate_m31(t, finite=finite)
 
    def compute_differential_rate(self, ts, finite=False):
        return [self.differential_rate_total(t, finite=finite) for t in ts]
    
    def differential_rate_mass(self, m, integrand_func, finite=True, tcad=0.07, tobs=3, epsabs = 1.49e-08, epsrel = 1.49e-08, efficiency=None, monochromatic=False):        
        
        if efficiency is None:
            def efficiency(t):
                return 1

        point = self.sticking_point(m)

        def inner_integrand(u, d, t, m):
            return integrand_func(u, d, t, m) * efficiency(t)
            
        # Second integral (over u) - bounds given by d
        def second_integral(d, t, m):
            if finite:
                u_min, u_max = 0, self.umin_upper_bound(d, m)[0]
            else:
                u_min, u_max = 0, self.u_t
            result, error = quad(inner_integrand, u_min, u_max, args=(d, t, m), epsabs=epsabs, epsrel=epsrel, points=[point])
            return result
            
        # Third integral (over d)
        def third_integral(t, m):
            if finite:
                d_min, d_max = 0, self.d_upper_bound(m)
            else:
                d_min, d_max = 0, self.ds

            result, error = quad(second_integral, d_min, d_max, args=(t, m), epsabs=epsabs, epsrel=epsrel)
            return result
                
        # Outermost integral (over t)
        t_min = tcad
        t_max = tobs
        result, error = quad(third_integral, t_min, t_max, args=(m,), epsabs=epsabs, epsrel=epsrel)
        
        if monochromatic:
            return result
        return result * self.f_m(m)



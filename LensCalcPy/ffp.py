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
from scipy.integrate import quad, nquad, dblquad, tplquad
from scipy.interpolate import interp1d, interp2d
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
        # self.Z = self.pl_norm(self.p)
        self.Z = self.pl_norm_new()

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


    def dN_dM(self, A, M, M_norm, p):
        return A * (M/M_norm)**-p / M
    
    def dN_dM_wrapper(self, M):
        return self.dN_dM(1, M, self.M_norm, self.p)
    
    def dN_dlogM(self, A, log10M, M_norm, p):
        M = 10**log10M
        return A * (M/self.M_norm)**-p
    
    def dN_dlogM_wrapper(self, M):
        return self.dN_dlogM(1, M, self.M_norm, self.p)
    
    def pl_norm_new(self):
        return 1/abs(nquad(self.dN_dlogM_wrapper,[[np.log10(self.m_min), np.log10(self.m_max)]], opts={'points': [np.log10(self.m_min), np.log10(self.m_min*1e3)]})[0])

    def mass_func(self, log10m):
        #M_norm = 1 solar mass for now. This is dN/dlogM
        m = 10**log10m
        return self.Z * (m/1)**-self.p
    
    def pl_norm(self, p):
        N_ffp = 1 # Number of FFPs per star
        return N_ffp/abs(nquad(self.mass_func,[[self.m_min, self.m_max]], opts={'points': [self.m_min, self.m_min*1e3]})[0])

    def differential_rate_integrand(self, umin, d, mf, t, dist_func, density_func, v_disp_func, finite=False):
        r = dist_func(d, self.l, self.b)
        ut = self.umin_upper_bound(d, mf) if (self.ut_interp and finite) else self.u_t
        if ut <= umin:
            return 0
        v_rad = velocity_radial(d, mf, umin, t * htosec, ut)  
        v_disp = v_disp_func(r)
        return 2 * (1 / (ut**2 - umin**2)**0.5 *
                        #For FFP number density, use stellar density for 1 solar mass stars
                density_func(d) / (1 * v_disp**2) *  
                v_rad**4 * (htosec / kpctokm)**2 *
                np.exp(-(v_rad**2 / v_disp**2)) *
                1)

    def differential_rate(self, t, integrand_func, finite=False):
        num = 40  # number of discretization points, empirically, result levels off for >~ 40
        mf_values = np.linspace(np.log10(self.m_min), np.log10(self.m_max), num=num)

        result = 0
        for i in range(num):
            mf = mf_values[i]
            if i == 0:  # for the first point
                dm = mf_values[i+1] - mf_values[i]
            elif i < num - 1:  # for middle points
                dm = ((mf_values[i+1] - mf_values[i]) + (mf_values[i] - mf_values[i-1])) / 2
            else:  # for the last point
                dm = mf_values[i] - mf_values[i-1]
            if finite:
                single_result, error = dblquad(integrand_func, 
                                            0, self.ds, 
                                            lambda d: 0, 
                                            lambda d: self.umin_upper_bound(d, 10**mf),
                                            # args=(mf, t),
                                            args=(10**mf, t),
                                            epsabs=0,
                                            epsrel=1e-1,
                                            )
            else:
                single_result, error = dblquad(integrand_func,
                                               #Without finite size effects, integral blows up at M31 center
                                            0, self.ds*0.99,
                                            lambda d: 0, 
                                            lambda d: self.u_t,
                                            args=(10**mf, t),
                                            # epsabs=0,
                                            # epsrel=1e-2,
                                            )
            # if single_result != 0 and error/abs(single_result) >=1:
                # print("Warning: error in differential rate integration is large: {}".format(error/abs(single_result)))
            
            result += single_result * ((10**mf/1) ** -self.p) * dm # multiply by mass function and by dlogm. This is for dN/dlogM

        result *= self.Z  # normalization
        return result
    
    def differential_rate_monochromatic(self, t, integrand_func, finite=False, m=1e-10):

        # todo rescale the integration distance to be x = dl/ds instead of d
    
        if finite:
            result, error = dblquad(integrand_func, 
                                        0, self.ds, 
                                        lambda d: 0, 
                                        lambda d: self.umin_upper_bound(d, m),
                                        args=(m, t),
                                        epsabs=0,
                                        epsrel=1e-1,
                                        )
        
        else:
            result, error = dblquad(integrand_func,
                                            #Without finite size effects, integral blows up at M31 center
                                        0, self.ds*0.99,
                                        lambda d: 0, 
                                        lambda d: self.u_t,
                                        args=(m, t),
                                        )
        return result
        
    def differential_rate_integrand_mw(self, umin, d, mf, t, finite=False, vel_func=None):
        if vel_func is None:
            vel_func = self.mw_model.velocity_dispersion_stars
        return self.differential_rate_integrand(umin, d, mf, t, self.mw_model.dist_center, self.mw_model.density_stars, vel_func, finite=finite)

    def differential_rate_mw(self, t, finite=False, v_disp=None):
        if v_disp:
            vel_func = lambda r: v_disp
            f = functools.partial(self.differential_rate_integrand_mw, vel_func=vel_func)
            return self.differential_rate(t, f, finite=finite)
        return self.differential_rate(t, self.differential_rate_integrand_mw, finite=finite)
    
    def differential_rate_mw_monochromatic(self, t, finite=False, m=1e-10):
        return self.differential_rate_monochromatic(t, self.differential_rate_integrand_mw, finite=finite, m=m)

    def differential_rate_integrand_m31(self, umin, d, mf, t, finite=False, vel_func=None):
        if vel_func is None:
            vel_func = self.m31_model.velocity_dispersion_stars
        return self.differential_rate_integrand(umin, d, mf, t, self.m31_model.dist_center, self.m31_model.density_stars, vel_func, finite=finite)

    def differential_rate_m31(self, t, finite=False, v_disp=None):
        if v_disp:
            vel_func = lambda r: v_disp
            f = functools.partial(self.differential_rate_integrand_m31, vel_func=vel_func)
            return self.differential_rate(t, f, finite=finite)
        return self.differential_rate(t, self.differential_rate_integrand_m31, finite=finite)
    
    def differential_rate_m31_monochromatic(self, t, finite=False, m=1e-10):
        return self.differential_rate_monochromatic(t, self.differential_rate_integrand_m31, finite=finite, m=m)

    def umin_upper_bound(self, d, m):
        if self.ut_interp is None:
            self.make_ut_interp()
        rho = rho_func(m, d, self.ds)
        return self.ut_interp(rho, magnification(self.u_t))
        # return self.ut_interp(d, m)[0]
    
    def differential_rate_total(self, t, finite=False):
        return self.differential_rate_mw(t, finite=finite) + self.differential_rate_m31(t, finite=finite)
 
    def compute_differential_rate(self, ts, finite=False):
        return [self.differential_rate_total(t, finite=finite) for t in ts]

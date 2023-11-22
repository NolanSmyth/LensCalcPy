# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_pbh.ipynb.

# %% auto 0
__all__ = ['Pbh']

# %% ../nbs/00_pbh.ipynb 3
from .parameters import *
from .utils import *
from .lens import *
from .galaxy import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad, dblquad
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import functools
import pickle
from abc import ABC, abstractmethod

from numba import njit, types
from numba.experimental import jitclass

from fastcore.test import *

# %% ../nbs/00_pbh.ipynb 5
# @jitclass
class Pbh(Lens):
    """A class to represent a PBH population"""
    mw_model : MilkyWayModel
    m31_model : M31Model
    l : float
    b : float
    mass : float
    f_dm : float
    u_t : float
    ds : float
    sigma : float
    # ut_interp : types.func

    def __init__(self,
                mass: float, # PBH mass in solar masses
                f_dm: float, # PBH fraction of the DM density
                mw_model = None,
                m31_model = None,
                l : float = 0,
                b : float = 0,
                u_t = 1,
                ds = 770,
                lognorm=False,
                sigma=None
                ):
        """
        Initialize the PBH population
        """

        self.mw_model = mw_model or MilkyWayModel()
        self.m31_model = m31_model or M31Model()
        
        # if l is None:
        #     raise ValueError("Galactic longitude must be specified")
        # if b is None:
        #     raise ValueError("Galactic latitude must be specified")
        self.l = l
        self.b = b

        self.mass = mass
        if f_dm < 0 or f_dm > 1:
            raise ValueError("f_dm must be between 0 and 1")
        self.f_dm = f_dm
        # self.ut_interp = ut_interp
        self.ut_interp = ut_func_new
        self.u_t = u_t
        self.ds = ds
        self.lognorm = lognorm
        if self.lognorm and sigma is None:
            raise ValueError("Sigma must be specified for lognormal mass function")
        self.sigma = sigma
    
    def __str__(self) -> str:
        return f"PBH population with mass={self.mass} and f_dm={self.f_dm}"
    __repr__ = __str__

    def mass_function(self, m):
        """The mass function of the PBH population"""
        if not self.sigma or not self.lognorm:
            raise ValueError("Lognormal mass function not initialized")
        return 1 / (m * self.sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(m/self.mass))**2 / (2 * self.sigma**2))

    def differential_rate_integrand(self, umin, d, t, model, finite=False, v_disp=None, t_e = False, tmax=np.inf, tmin=0):
    # def differential_rate_integrand(self, umin, d, t, mf, model, finite=False, v_disp=None, t_e = False, tmax=np.inf, tmin=0):

        r = model.dist_center(d, self.l, self.b)
        ut = self.umin_upper_bound(d) if finite else self.u_t
        if ut <= umin:
            return 0
        if t_e: 
            #Calculate radial velocity in terms of einstein crossing time
            v_rad = einstein_rad(d, self.mass, self.ds) * kpctokm / (t * htosec) 
            t_duration = max(ut, self.umin_upper_bound(d)) * 2 * einstein_rad(d, self.mass) * kpctokm / v_rad / htosec #event duration in hours

            if t_duration > tmax or t_duration < tmin:
                return 0     
        else:
            #Calculate radial velocity in terms of event duration (t_fwhm)
            v_rad = velocity_radial(d, self.mass, umin, t * htosec, ut) 
        if v_disp is None: 
            v_disp = model.velocity_dispersion_dm(r)
        return 2 * (1 / (ut**2 - umin**2)**0.5 *
                model.density_dm(d, self.l, self.b) / (self.mass * v_disp**2) *  
                v_rad**4 * (htosec / kpctokm)**2 *
                np.exp(-(v_rad**2 / v_disp**2)) *
                1)

    def differential_rate_integrand_mw(self, umin, d, t, finite=False):
        return self.differential_rate_integrand(umin, d, t, self.mw_model, finite=finite)
    
    def differential_rate_integrand_m31(self, umin, d, t, finite=False):
        return self.differential_rate_integrand(umin, d, t, self.m31_model, finite=finite)
    
    def differential_rate(self, t, integrand_func, finite=False):
        if finite:
            result, error = dblquad(integrand_func, 
                                    0, self.d_upper_bound(),
                                    lambda d: 0, 
                                    lambda d: self.umin_upper_bound(d),
                                    args=[t],
                                    epsabs=0,
                                    epsrel=1e-1,
                                    )
        else:
            result, error = dblquad(integrand_func,
                                        #Without finite size effects, integral blows up at M31 center
                                    0, self.ds*0.99,
                                    lambda d: 0, 
                                    lambda d: self.u_t,
                                    args=[t],
                                    epsabs=0,
                                    epsrel=1e-1,
                                    )
        return result

    def rate_total(self, integrand_func, finite=True, tcad=0.07, tobs=3, epsabs = 1.49e-08, epsrel = 1.49e-08, efficiency=None):        
        
        if efficiency is None:
            def efficiency(t):
                return 1
            
        point = self.sticking_point()

        def integrand(t, d, finite):
            if finite:
                u_bounds = [0, self.umin_upper_bound(d)]
            else:
                u_bounds = [0, self.u_t]
            u_result, _ = nquad(integrand_func, [u_bounds], args=(d, t))
            return u_result * efficiency(t)

        bounds_t = [tcad, tobs]

        if finite:
            bounds_d = [0, min(self.d_upper_bound(), self.ds)]
        else:
            bounds_d = [0, self.ds]

        opts = {"epsabs": epsabs, "epsrel": epsrel, "points":[point, self.ds]}

        result, error = nquad(integrand, [bounds_t, bounds_d], args=[finite], opts=opts)
        return result
    
    
    def d_upper_bound(self, m=None):
        if m is None:
            m = self.mass
        #Determine upper limit for d otherwise for low masses, the contribution which only comes from d << 1 gets missed
        d_arr = np.logspace(-3, np.log10(self.ds*0.99), 100)
        for d in d_arr:
            if self.umin_upper_bound(d,m=m) == 0:
                return d
        return self.ds
    
    def rate_mw(self, finite=False, tcad = 0.07, tobs = 3):
        return self.rate_total(self.differential_rate_integrand_mw, finite=finite, tcad=tcad, tobs=tobs)
    
    def rate_m31(self, finite=False):
        result = self.rate_total(self.differential_rate_integrand_m31, finite=finite)
        if not np.isnan(result):
            return result
        else:
            return 0
        
    def rate_tot(self, finite=False):
        return self.rate_mw(finite=finite) + self.rate_m31(finite=finite)
    
    def differential_rate_mw(self, t, finite=True, v_disp=None, t_e=False, tmax=np.inf, tmin=0):
        def integrand_func(umin, d, t):
            return self.differential_rate_integrand(umin, d, t, self.mw_model, finite=finite, v_disp=v_disp, t_e = t_e, tmax=tmax, tmin=tmin)
        return self.differential_rate(t, integrand_func, finite=finite)
    
    def differential_rate_m31(self, t, finite=True, v_disp=None):
        def integrand_func(umin, d, t):
            return self.differential_rate_integrand(umin, d, t, self.m31_model, finite=finite, v_disp=v_disp)
        return self.differential_rate(t, integrand_func, finite=finite)

    def umin_upper_bound(self, d, m=None):
        if m is None:
            m = self.mass
        rho = rho_func(m, d, self.ds)
        return self.ut_interp(rho, magnification(self.u_t))
    
    def differential_rate_total(self, t, finite=True):
        return self.differential_rate_mw(t, finite=finite) + self.differential_rate_m31(t, finite=finite)
 
    def compute_differential_rate(self, ts, finite=False):
        return [self.differential_rate_total(t, finite=finite) for t in ts]
    
    def sticking_point(self):
        #Determine where u_t is maximized. This speeds up the integral in m31
        result = minimize_scalar(lambda d: -self.umin_upper_bound(d), bounds=(0, self.ds), method='bounded')
        if result.success:
            return result.x[0] if isinstance(result.x, (list, np.ndarray)) else result.x
        else:
            return self.ds
        
    def differential_rate_lognorm(self, t, integrand_func, finite=False, epsabs = 1.49e-08, epsrel = 1.49e-08):
        #rewrite using tplquad?
        num = 20  # number of discretization points, empirically, result levels off for >~ 40
        mf_values = np.linspace(np.log10(self.mass)-2, np.log10(self.mass)+2, num=num)

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
                                            # 0, 10, 
                                            0, self.d_upper_bound(10**mf),
                                            lambda d: 0, 
                                            lambda d: self.umin_upper_bound(d, 10**mf),
                                            args=(10**mf, t),
                                            epsabs=epsabs,
                                            epsrel=epsrel,
                                            )
            else:
                single_result, error = dblquad(integrand_func,
                                               #Without finite size effects, integral blows up at M31 center
                                            0, self.ds,
                                            lambda d: 0, 
                                            lambda d: self.u_t,
                                            args=(10**mf, t),
                                            epsabs=epsabs,
                                            epsrel=epsrel,
                                            )
            
            # result += single_result * ((10**mf/1) ** -self.p) * dm # multiply by mass function and by dlogm. This is for dN/dlogM
            result += single_result * self.mass_function(10**mf) * dm * 10**mf # multiply by mass function and by dlogm * m. 
        return result
    
    def differential_rate_integrand_lognorm(self, umin, d, t, mf, model, finite=False, v_disp=None, t_e = False, tmax=np.inf, tmin=0):

        r = model.dist_center(d, self.l, self.b)
        ut = self.umin_upper_bound(d) if finite else self.u_t
        if ut <= umin:
            return 0 
        else:
            #Calculate radial velocity in terms of event duration (t_fwhm)
            v_rad = velocity_radial(d, mf, umin, t * htosec, ut) 
        if v_disp is None: 
            v_disp = model.velocity_dispersion_dm(r)
        return 2 * (1 / (ut**2 - umin**2)**0.5 *
                model.density_dm(d, self.l, self.b) / (mf * v_disp**2) *  
                v_rad**4 * (htosec / kpctokm)**2 *
                np.exp(-(v_rad**2 / v_disp**2)) *
                1)
    
    def differential_rate_mw_lognorm(self, t, finite=True, v_disp=None, t_e=False, tmax=np.inf, tmin=0):
        def integrand_func(umin, d, mf, t):
            return self.differential_rate_integrand_lognorm(umin, d, t, mf, self.mw_model, finite=finite, v_disp=v_disp, t_e = t_e, tmax=tmax, tmin=tmin)
        return self.differential_rate_lognorm(t, integrand_func, finite=finite)

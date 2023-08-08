# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_ffp.ipynb.

# %% auto 0
__all__ = ['zthin', 'rho_thin_mw', 'rho_thick_mw', 'rsf', 'fE', 'cut', 'rho_bulge_mw', 'rho_FFPs_mw',
           'velocity_dispersion_stars_mw', 'einasto', 'rho_bulge_m31', 'rho_disk_m31', 'rho_nucleus_m31',
           'rho_FFPs_m31', 'velocity_dispersion_stars_m31', 'Ffp']

# %% ../nbs/01_ffp.ipynb 3
from .parameters import *
from .utils import *
from .lens import *

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
# Add Koshimoto Parametric MW Model
# https://iopscience.iop.org/article/10.3847/1538-4357/ac07a8/pdf

# Disk Density
def zthin(r):
    if r > 4.5:
        return zthinSol - (zthinSol - zthin45) * (rsol - r) / (rsol - 4.5)
    else:
        return zthin45

def rho_thin_mw(r, 
             z,
            ) -> float: # FFP density in Msun/kpc^3

    if r > rdBreak:
        result = rho_thin_Sol * zthinSol / zthin(r) * \
            np.exp(-((r - rsol) / rthin)) * \
            (1 / np.cosh(-np.abs(z) / zthin(r)))**2
    else:
        result = rho_thin_Sol * zthinSol / zthin(r) * \
            np.exp(-((rdBreak - rsol) / rthin)) * \
            (1 / np.cosh(-np.abs(z) / zthin(r)))**2
    
    return result 

def rho_thick_mw(r, 
              z, 
            ) -> float: # FFP density in Msun/kpc^3
    
    if r > rdBreak:
        result = rho_thick_Sol * np.exp(-((r - rsol) / rthick)) * \
            np.exp(-(np.abs(z) / zthickSol))
    else:
        result = rho_thick_Sol * np.exp(-((rdBreak - rsol) / rthick)) * \
            np.exp(-(np.abs(z) / zthickSol))

    return result 

# Bulge Density
def rsf(xp, yp, zp):
    rs = (((xp/x0)**cperp + (yp/y0)**cperp)**(cpar/cperp) + (zp/z0)**cpar)**(1/cpar)
    return rs

def fE(xp, yp, zp):
    return np.exp(-rsf(xp, yp, zp))

def cut(x):
    if x > 0:
        return np.exp(-x**2)
    else:
        return 1

def rho_bulge_mw(d: float,
            ) -> float: # FFP density in Msun/kpc^3
    xp, yp, zp = get_primed_coords(d)
    xp, yp, zp = abs(xp), abs(yp), abs(zp)
    r = (xp**2 + yp**2 + zp**2)**0.5
    return rho0_B * fE(xp, yp, zp) * cut((r - Rc) / 0.5)

# Total FFP Density
def rho_FFPs_mw(d: float, # distance from Sun in kpc
             ) -> float: # FFP density in Msun/kpc^3
    r = dist_mw(d)
    _, _, z = get_primed_coords(d)
    return (rho_thin_mw(r, z) + rho_thick_mw(r, z) + rho_bulge_mw(d)) 

def velocity_dispersion_stars_mw(r,
                                #  v_c: float = 30 # km/s
                                 v_c: float = 15 # km/s
                                ):
    #Take 30 km/s following table 2 of https://arxiv.org/pdf/2306.12464.pdf
    return v_c

# %% ../nbs/01_ffp.ipynb 8
# Add stellar distribution of M31 following: https://www.aanda.org/articles/aa/pdf/2012/10/aa20065-12.pdf

def einasto(a, rhoc, dn, ac, n):
    return rhoc * np.exp(-dn *((a/ac)**(1/n) - 1))

def rho_bulge_m31(d, 
                ) -> float: # FFP density in Msun/kpc^3
    q = 0.72
    i = np.deg2rad(90-77)  # inclination angle of m31 disk in radians
    z = d * np.sin(i)
    r = d * np.cos(i)
    a = (r**2 + z**2/q**2)**0.5
    rhoc = 9.201e-1 * (1e3)**3 #Msun/kpc^3
    dn = 7.769
    ac = 1.155 #kpc
    n = 2.7
    return einasto(a, rhoc, dn, ac, n) 

def rho_disk_m31(d,
                    ) -> float: # FFP density in Msun/kpc^3
    q = 0.17
    i = np.deg2rad(90-77)  # inclination angle of m31 disk in radians
    z = d * np.sin(i)
    r = d * np.cos(i)
    a = (r**2 + z**2/q**2)**0.5
    rhoc = 1.307e-2 * (1e3)**3 #Msun/kpc^3
    dn = 3.273
    ac = 10.67 #kpc
    n = 1.2
    return einasto(a, rhoc, dn, ac, n) 

def rho_nucleus_m31(d,
                    ) -> float: # FFP density in Msun/kpc^3
    q = 0.99
    i = np.deg2rad(90-77)  # inclination angle of m31 disk in radians
    z = d * np.sin(i)
    r = d * np.cos(i)
    a = (r**2 + z**2/q**2)**0.5
    rhoc = 1.713 * (1e3)**3 #Msun/kpc^3
    dn = 11.668
    ac = 0.0234 #kpc
    n = 4.0
    return einasto(a, rhoc, dn, ac, n) 

def rho_FFPs_m31(d: float, # distance from center of M31 in kpc
             ) -> float: # FFP density in Msun/kpc^3
    # return (rho_bulge_m31(a) + rho_disk_m31(a) + rho_nucleus_m31(a))

    #The bulge/nucleus is excluded from the m31 survey
    if use_max_density:
        return rho_disk_m31(d) * 1.5
    return rho_disk_m31(d) 

def velocity_dispersion_stars_m31(r,
                                #  v_c: float = 60 # km/s
                                 v_c: float = 30 # km/s

                                ):
    # Use 60 km/s for disk following https://iopscience.iop.org/article/10.1088/0004-637X/695/1/442/pdf
    return v_c


# %% ../nbs/01_ffp.ipynb 12
class Ffp(Lens):
    """A class to represent a PBH population"""

    def __init__(self,
                p: float = 1, # Mass function power law index
                m_min: float = 1e-15, # Minimum mass in Msun
                # m_max: float = 1e-5, # Maximum mass in Msun
                m_max: float = 1e-3, # Maximum mass in Msun
                use_mw_source: bool = False,
                ):
        """
        Initialize the PBH population
        """
        
        if use_mw_source:
            self.ut_interp = ut_interp_mw #assuming source is 8.5 kpc away
        else:
            self.ut_interp = ut_interp # assuming source is in m31, 770 kpc away
        # self.ut_interp = ut_interp_mw

        self.p = p
        #Define range of power law we want to consider
        self.m_min = m_min
        self.m_max = m_max
        self.M_norm = 1 #solar mass
        # self.Z = self.pl_norm(self.p)
        self.Z = self.pl_norm_new()

    def __str__(self) -> str:
        return f"FFP with power law dN / dlogM ~ m^-{self.p}"
    __repr__ = __str__

    # def mass_func(self, m):
    #     #M_norm = 1 solar mass for now
    #     return (m / 1) ** -self.p

    def dN_dM(self, A, M, M_norm, p):
        return A * (M/M_norm)**-p / M
    
    def dN_dM_wrapper(self, M):
        return self.dN_dM(1, M, self.M_norm, self.p)
    
    def dN_dlogM(self, A, log10M, M_norm, p):
        M = 10**log10M
        return A * (M/self.M_norm)**-p
    
    def dN_dlogM_wrapper(self, M):
        return self.dN_dlogM(1, M, self.M_norm, self.p)
    
    # def pl_norm_new(self, p):
    #     return 1/abs(nquad(self.dN_dM_wrapper,[[self.m_min, self.m_max]], opts={'points': [self.m_min, self.m_min*1e3]})[0])

    def pl_norm_new(self):
        return 1/abs(nquad(self.dN_dlogM_wrapper,[[np.log10(self.m_min), np.log10(self.m_max)]], opts={'points': [np.log10(self.m_min), np.log10(self.m_min*1e3)]})[0])
    
    # def mass_func(self, m):
    #     #M_norm = 1 solar mass for now
    #     return self.Z * (m/1)**-self.p / m

    def mass_func(self, log10m):
        #M_norm = 1 solar mass for now. This is dN/dlogM
        m = 10**log10m
        return self.Z * (m/1)**-self.p
    
    def pl_norm(self, p):
        N_ffp = 1 # Number of FFPs per star
        return N_ffp/abs(nquad(self.mass_func,[[self.m_min, self.m_max]], opts={'points': [self.m_min, self.m_min*1e3]})[0])

    def differential_rate_integrand(self, umin, d, mf, t, dist_func, density_func, v_disp_func, finite=False, density_func_uses_d=False):
        r = dist_func(d)
        ut = self.umin_upper_bound(d, mf) if (self.ut_interp and finite) else 1
        if ut <= umin:
            return 0
        v_rad = velocity_radial(d, mf, umin, t * htosec, ut)  
        v_disp = v_disp_func(r)
        density_input = d if density_func_uses_d else r
        return 2 * (1 / (ut**2 - umin**2)**0.5 *
                        #For FFP number density, use stellar density for 1 solar mass stars
                density_func(density_input) / (1 * v_disp**2) *  
                v_rad**4 * (htosec / kpctokm)**2 *
                np.exp(-(v_rad**2 / v_disp**2)) *
                1)

    def differential_rate(self, t, integrand_func, finite=False):
        num = 40  # number of discretization points, empirically, result levels off for >~ 40
        # mf_values = np.logspace(np.log10(self.m_min), np.log10(self.m_max), num=num)
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
                                            0, ds, 
                                            lambda d: 0, 
                                            lambda d: self.umin_upper_bound(d, 10**mf),
                                            # args=(mf, t),
                                            args=(10**mf, t),

                                            # epsabs=0,
                                            # epsrel=1e-2,
                                            )
            else:
                single_result, error = dblquad(integrand_func,
                                               #Without finite size effects, integral blows up at M31 center
                                            0, ds*0.99,
                                            lambda d: 0, 
                                            lambda d: ut,
                                            args=(10**mf, t),
                                            # epsabs=0,
                                            # epsrel=1e-2,
                                            )
            # if single_result != 0 and error/abs(single_result) >=1:
                # print("Warning: error in differential rate integration is large: {}".format(error/abs(single_result)))

            # result += single_result * (mf ** -self.p) * dm  # multiply by mass function and by dm
            # result += single_result * (mf ** -self.p) / mf * dm  # multiply by mass function and by dm
            # print(10**mf, single_result* ((10**mf/1) ** -self.p) * dm)
            
            result += single_result * ((10**mf/1) ** -self.p) * dm # multiply by mass function and by dlogm. This is for dN/dlogM

        result *= self.Z  # normalization
        return result
    
    def differential_rate_monochromatic(self, t, integrand_func, finite=False, m=1e-10):
    
        if finite:
            result, error = dblquad(integrand_func, 
                                        0, ds, 
                                        lambda d: 0, 
                                        lambda d: self.umin_upper_bound(d, m),
                                        args=(m, t),
                                        )
        else:
            result, error = dblquad(integrand_func,
                                            #Without finite size effects, integral blows up at M31 center
                                        0, ds*0.99,
                                        lambda d: 0, 
                                        lambda d: ut,
                                        args=(m, t),
                                        )
        return result
        
    def differential_rate_integrand_mw(self, umin, d, mf, t, finite=False, vel_func = velocity_dispersion_stars_mw):
        return self.differential_rate_integrand(umin, d, mf, t, dist_mw, rho_FFPs_mw, vel_func, finite=finite, density_func_uses_d=True)
        
    def differential_rate_mw(self, t, finite=False, v_disp = 30):
        f = functools.partial(self.differential_rate_integrand_mw, vel_func = lambda r: v_disp)
        return self.differential_rate(t, f, finite=finite)
        # return self.differential_rate(t, self.differential_rate_integrand_mw, finite=finite)
    
    def differential_rate_mw_monochromatic(self, t, finite=False, m=1e-10):
        return self.differential_rate_monochromatic(t, self.differential_rate_integrand_mw, finite=finite, m=m)

    def differential_rate_integrand_m31(self, umin, d, mf, t, finite=False, vel_func = velocity_dispersion_stars_m31):
        return self.differential_rate_integrand(umin, d, mf, t, dist_m31, rho_FFPs_m31, vel_func , finite=finite, density_func_uses_d=False)

    def differential_rate_m31(self, t, finite=False, v_disp = 60):
        f = functools.partial(self.differential_rate_integrand_m31, vel_func = lambda r: v_disp)
        return self.differential_rate(t, f, finite=finite)
        # return self.differential_rate(t, self.differential_rate_integrand_m31, finite=finite)
    
    def differential_rate_m31_monochromatic(self, t, finite=False, m=1e-10):
        return self.differential_rate_monochromatic(t, self.differential_rate_integrand_m31, finite=finite, m=m)

    def umin_upper_bound(self, d, m):
        if self.ut_interp is None:
            self.make_ut_interp()
        return self.ut_interp(d, m)[0]
    
    def differential_rate_total(self, t, finite=False):
        return self.differential_rate_mw(t, finite=finite) + self.differential_rate_m31(t, finite=finite)
 
    def compute_differential_rate(self, ts, finite=False):
        return [self.differential_rate_total(t, finite=finite) for t in ts]

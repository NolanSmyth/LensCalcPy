# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_pbh.ipynb.

# %% auto 0
__all__ = ['Pbh']

# %% ../nbs/00_pbh.ipynb 3
from .parameters import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad

# %% ../nbs/00_pbh.ipynb 5
class Pbh:
    """A class to represent a PBH population"""

    def __init__(self,
                m_pbh: float, # PBH mass in solar masses
                f_dm: float # PBH fraction of the DM density
                ):
        """
        Initialize the PBH population
        """
        self.m_pbh = m_pbh
        if f_dm < 0 or f_dm > 1:
            raise ValueError("f_dm must be between 0 and 1")
        self.f_dm = f_dm
    
    def __str__(self) -> str:
        return f"PBH population with m_pbh={self.m_pbh} and f_dm={self.f_dm}"
    __repr__ = __str__

    def density(self, 
                r: float # distance in kpc
                ) -> float:
        """PBH density at a given distance from the Milky Way center
        Using an NFW profile

        Args:
            r (float): distance in kpc

        Returns:
            float: PBH density in Msol/kpc^-3
        """
        return rhoc / ((r/rs) * (1 + r/rs)**2) * self.f_dm
    
    def mass_enclosed(self, 
                      r: float, # distance from MW center in kpc
                      ) -> float:
        """PBH mass enclosed within a given distance from the Milky Way center

        Returns:
            float: PBH mass in Msol
        """
        return 4*np.pi * rhoc * rs**3 * (np.log(1 + r/rs) - (r/rs)/(1 + r/rs))
    
    def velocity_dispersion(self, 
                            d: float # distance from Sun in kpc
                            ) -> float:
        """PBH velocity dispersion at a given distance from the Milky Way center
        Returns:
            float: PBH velocity dispersion in km/s
        """
        r = self.dist_mw(d)
        return np.sqrt(G * self.mass_enclosed(r) / r) 
    
    def velocity_radial(self,
                        d: float, # distance from the Sun
                        umin: float, # minimum impact parameter
                        t: float # crossing time
                        ) -> float:
        """
        PBH radial velocity at a given distance from the Sun
        """
        return 2*self.einstein_rad(d) * (ut**2 - umin**2)**(1/2) / t * kpctokm

    
    def dist(self,
             d: float # distance from the Sun
             ) -> float:
        return d * (1 - d/ds)

    def einstein_rad(self, 
                     d: float, # distance from the Sun
                     ) -> float:
        return np.real((4 * G * self.m_pbh * self.dist(d)/c**2)**(1/2))
    
    def dist_mw(self, d):
        """returns the distance to the Milky Way center in kpc of a point a distance d to the Sun in kpc"""
        return np.sqrt(d**2 + rEarth**2 - 2*d*rEarth*np.cos(np.radians(l))*np.cos(np.radians(b)))
    
    def differential_rate_integrand(self, umin, d, t):
        r = self.dist_mw(d)
        return (1 / (ut**2 - umin**2)**0.5 *
                self.density(r) / (self.m_pbh * self.velocity_dispersion(d)**2) *
                self.velocity_radial(d, umin, t * htosec)**4 * (htosec / kpctokm)**2 *
                np.exp(-(self.velocity_radial(d, umin, t * htosec)**2 / self.velocity_dispersion(d)**2)))
    
    def differential_rate(self, t):
        umin_bounds = [0, ut]
        d_bounds = [0, rEarth]

        result, error = nquad(self.differential_rate_integrand, [umin_bounds, d_bounds], args=[t])

        return result

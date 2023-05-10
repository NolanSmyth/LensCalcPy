# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_ffp.ipynb.

# %% auto 0
__all__ = ['options', 'zthin', 'rho_thin', 'rho_thick', 'rsf', 'fE', 'cut', 'rho_bulge', 'rho_FFPs', 'dGdt_FFP', 'Ffp']

# %% ../nbs/01_ffp.ipynb 3
from .parameters import *
from .utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.interpolate import interp1d


# %% ../nbs/01_ffp.ipynb 5
# Disk Density
def zthin(r):
    if r > 4.5:
        return zthinSol - (zthinSol - zthin45) * (rsol - r) / (rsol - 4.5)
    else:
        return zthin45

def rho_thin(r, z):
    if r > rdBreak:
        return rho_thin_Sol * zthinSol / zthin(r) * \
            np.exp(-((r - rsol) / rthin)) * (1 / np.cos(np.abs(z) / zthin(r)))**2
    else:
        return rho_thin_Sol * zthinSol / zthin(r) * \
            np.exp(-((rdBreak - rsol) / rthin)) * (1 / np.cos(np.abs(z) / zthin(r)))**2

def rho_thick(r, z):
    if r > rdBreak:
        return rho_thick_Sol * np.exp(-((r - rsol) / rthick)) * \
            np.exp(-(np.abs(z) / zthickSol))
    else:
        return rho_thick_Sol * np.exp(-((rdBreak - rsol) / rthick)) * \
            np.exp(-(np.abs(z) / zthickSol))

# Bulge Density
def rsf(xp, yp, zp):
    R = (xp**cperp / x0**cperp + yp**cperp / y0**cperp)**(cpar/cperp) + (zp / z0)**cpar
    return R**(1/cpar)

def fE(xp, yp, zp):
    return np.exp(-rsf(xp, yp, zp))

def cut(x):
    if x > 0:
        return np.exp(-x**2)
    else:
        return 1

def rho_bulge(xp, yp, zp):
    R = (xp**2 + yp**2 + zp**2)**0.5
    return rho0_B * fE(xp, yp, zp) * cut((R - Rc) / 0.5)

def rho_bulge(d):
    xp, yp = get_primed_coords(d)
    zp = 0
    R = (xp**2 + yp**2 + zp**2)**0.5
    return rho0_B * fE(xp, yp, zp) * cut((R - Rc) / 0.5)

# Total FFP Density
def rho_FFPs(d: float, # distance from Sun in kpc
             ) -> float: # FFP density in Msun/kpc^3
    # TODO Need to weight this by number of FFPs per star and mass of FFPs
    r = dist_mw(d)
    z = 0
    return rho_thin(r, z) + rho_thick(r, z) + rho_bulge(d)



# %% ../nbs/01_ffp.ipynb 7
options = {"epsabs": 1e-10, "epsrel": 1e-10}

def dGdt_FFP(t, mFFP):
    def integrand(umin, d):
        r = dist_mw(d)
        return 2 / (ut**2 - umin**2)**(1/2) * rho_FFPs(d) / \
               (mFFP * velocity_dispersion_mw(r)**2) * velocity_radial(d, mFFP, umin, t * htosec)**4 * \
               (htosec / kpctokm)**2 * np.exp(-velocity_radial(d, mFFP, umin, t * htosec)**2 / velocity_dispersion_mw(r)**2)
    result, _ = nquad(integrand, [(0, ut), (0, rEarth)], opts=options)
    return result

# %% ../nbs/01_ffp.ipynb 9
class Ffp:
    
    def __init__(self, 
                 mlow: float, # lower mass limit in solar masses
                 alpha: float, # power law slope of distribution
                 ): 
        self.mlow = mlow
        self.alpha = alpha
        self.sample_masses = self.generate_sample(int(1e4))
        self.tE_interp = None
    
    def __str__(self):
        return f"FFP: mlow={self.mlow}, alpha={self.alpha}"
    __repr__ = __str__

    def generate_sample(self, 
                        n: int = int(1e4) # number of samples
                        ):
        return self.mlow * (1 - np.random.rand(int(n)))**(-1 / (self.alpha - 1))
    

    def get_ffp_pdf(self,
                    n_bins: int = 10, # number of mass bins
                    ):
        
        bins = np.logspace(np.log10(self.mlow), np.log10(np.max(self.sample_masses) * 1.01), num=n_bins)
        counts, hist_bins, = np.histogram(self.sample_masses, bins=bins, density=True)
        bin_centers = (hist_bins[1:] + hist_bins[:-1]) / 2
        ffpPDF = np.zeros((len(counts), len(bin_centers)))
        ffpPDF[0] = counts/np.sum(counts)
        ffpPDF[1] = bin_centers
        return ffpPDF
    
    def get_weighted_te(self,
                        n_bins: int = 10, # number of mass bins
                        ):
        ffpPDF = self.get_ffp_pdf(n_bins)
        tMin = 1e-2  # hours
        tMax = 1e3
        n_t_bins = 10
        tBins = np.logspace(np.log10(tMin), np.log10(tMax), num=n_t_bins)

        tETable = np.zeros((len(ffpPDF[0])-1, len(tBins)))
        for i in range(len(ffpPDF[0])-1):
            for j in range(len(tBins)):
                tETable[i,j] = ffpPDF[0][i] * dGdt_FFP(tBins[j], ffpPDF[1][i])

        tEWeighted = np.sum(tETable, axis=0)
        # tEInterp = interp1d(np.log10(tBins), np.log10(tEWeighted), kind="cubic")
        return tBins, tEWeighted
    
    def make_te_interp(self,
                       n_bins: int = 10, # number of mass bins
                       ):
        tBins, tEWeighted = self.get_weighted_te(n_bins)
        tEInterp = interp1d(np.log10(tBins), np.log10(tEWeighted), kind="cubic")
        self.tE_interp = tEInterp
    
    def differential_rate(self, t):
        if self.tE_interp is None:
            self.make_te_interp()
        return 10**self.tE_interp(np.log10(t))



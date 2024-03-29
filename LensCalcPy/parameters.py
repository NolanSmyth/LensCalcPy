# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_parameters.ipynb.

# %% auto 0
__all__ = ['rhoc', 'rs', 'G', 'rhocM31', 'rsM31', 'c', 'fpbh', 'ut', 'kpctokm', 'htosec', 'rEarth', 'r_max', 'dsM31', 'l', 'b',
           'ds', 'obs_time', 'survey_area', 'n_sources', 'efficiency', 'cadence', 'lam', 'zthinSol', 'zthickSol',
           'zthin45', 'rsol', 'rdBreak', 'rthin', 'rthick', 'use_max_density', 'rho_thin_Sol', 'rho_thick_Sol', 'x0',
           'y0', 'z0', 'cperp', 'cpar', 'rho0_B', 'Rc', 'alphabar']

# %% ../nbs/05_parameters.ipynb 3
rhoc = 4.88e6 # MW nfw central density parameter in Msol kpc^-3
rs = 21.5 # MW nfw scale radius in kpc
G = 4.3e-6 # kpc Msol^-1 (km/s)^2
rhocM31 = 4.96e6 # M31 nfw central density parameter in Msol kpc^-3
rsM31 = 25 # M31 nfw scale radius in kpc
c = 2.98e5 # km/s
fpbh = 1 # fraction of dark matter in PBHs
ut = 1 # threshold impact parameter
kpctokm = 3e16 # kpc to km
htosec = 60*60 # hours to seconds
rEarth = 8.5 # kpc
r_max = 2*rEarth # maximum distance to find PBH from Earth in kpc


# %% ../nbs/05_parameters.ipynb 4
#Survey parameters
# ds = 9.0 # kpc
dsM31 = 770 # kpc
# survey_area = 0.16 # survey area in square degrees
# obsTime = 1825*24 # observation time in hours

# l = 1.0 #degrees
# b = -1.03 #degrees
l = 121.2 #degrees
b = -21.6 #degrees

#LSST test parameters
# from https://arxiv.org/pdf/2308.00232.pdf
# ds = 8.5 # kpc
# n_sources = 1.2e6 # number of sources in the survey
# obsTime = 12.5 # observation time in hours
# l = 330.0 # degrees
# b = 20.0 # degrees
# efficiency = 0.6 # efficiency of the survey estimated from high S/N sources
# lam = 6000 # approx r band wavelength in angstroms
# survey_area = 9.6 # survey area in square degrees

ds = 770 #distance to source in kpc
dsM31 = 770 # kpc
obs_time = 7 # observation time in hours
# obsTime = 7 * 10# observation time in hours
survey_area = (1.5/2)**2 * 3.14159 # survey area in square degrees
n_sources = 8.7e7 # number of sources in the survey
efficiency = 0.6 # efficiency of the survey
cadence = 0.035 # cadence in hours
lam = 6000 # wavelength in angstroms


# %% ../nbs/05_parameters.ipynb 5
# Disk Params
zthinSol = 0.329  # scale height of thin disk in solar neighborhood, kpc
zthickSol = 0.903  # scale height of thick disk in solar neighborhood
zthin45 = 0.6 * zthinSol  # scale height, kpc
rsol = 8.160  # solar position, kpc
rdBreak = 5.3  # kpc, turnover point for density profile
rthin = 2.6  # kpc
rthick = 2.2  # kpc

# from https://arxiv.org/pdf/1704.05063.pdf 4.0 +/- 0.2 * 10^(-2) Msol pc^-3 for main sequence
use_max_density = True
#Upper density values
rho_thin_Sol = (4.2e-2 + 5.1e-3) * (1e3)**3  # local solar thin disk density main sequence stars and White Dwarfs, Msol kpc^-3
#Lower density values
# rho_thin_Sol = (3.8e-2 + 5.1e-3) * (1e3)**3  # local solar thin disk density main sequence stars and White Dwarfs, Msol kpc^-3

# local solar thick disk density main sequence stars and White Dwarfs. Normalized to be 4% of thin disk
rho_thick_Sol = rho_thin_Sol * 0.04 


# Bulge Params
x0 = 0.67  # kpc
y0 = 0.28  # kpc
z0 = 0.24  # kpc
cperp = 1.4
cpar = 3.3  # kpc
rho0_B = 9.72 * (1e3)**3  # msol kpc^-3

Rc = 2.8  # kpc
alphabar = 27 # degrees

import pickle
from os import path

file_path = path.dirname(__file__)

with open(file_path + "/ut_interp_m31.pkl", "rb") as f:
    ut_interp = pickle.load(f)

with open(file_path + "/ut_interp_mw.pkl", "rb") as f:
    ut_interp_mw = pickle.load(f)

with open(file_path + "/ut_interp_rho.pkl", "rb") as f:
    ut_interp_rho = pickle.load(f)

with open(file_path + "/u_fwhm_interp.pkl", "rb") as f:
    u_fwhm_interp = pickle.load(f)

import pickle
from os import path

file_path = path.dirname(__file__)

with open(file_path + "/ut_interp_m31.pkl", "rb") as f:
    ut_interp = pickle.load(f)

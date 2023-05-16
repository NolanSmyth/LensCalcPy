import pickle
from os import path

file_path = path.dirname(__file__)

with open(file_path + "/ut_interp_m31.pkl", "rb") as f:
    ut_interp = pickle.load(f)

with open(file_path + "/m_avg_interp.pkl", "rb") as f:
    m_avg_interp = pickle.load(f)

# ut_interp = ut_interp
# m_avg_interp = m_avg_interp

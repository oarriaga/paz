import numpy as np

epos = np.load("./predicted_epos.npy")

print(np.where(np.isinf(epos)))
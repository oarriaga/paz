import numpy as np
import pickle
import gzip

epos = np.load("/home/fabian/Dokumente/epos/test/epos_output/epos_output_0000000.npy")
epos = epos.astype(np.float32)
np.save("./epos_float16.npy", epos)

pickle.dump(epos, open("epos.pkl", "wb"))

f = gzip.GzipFile("epos.npy.gz", "w")
np.save(file=f, arr=epos)
f.close()

f = gzip.GzipFile('epos.npy.gz', "r")
epos_gz = np.load(f)
print(epos_gz)
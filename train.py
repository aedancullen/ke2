import os
import sys
import ke2
import h5py
import numpy as np

h5file = h5py.File("dataset.hdf5", 'r')
names = h5file.keys()

chunksets = [h5file[name] for name in names if h5file[name].shape[2] // 4 in [8, 16, 32, 6, 12, 24]]

h5file.close()

max_nsteps = 0
for chunkset in chunksets:
    max_nsteps = max(max_nsteps, chunkset.shape[2])

tiled_chunksets = []
for chunkset in chunksets:
    tiled_chunksets.append(np.tile(chunkset, max_nsteps // chunkset.shape[2]))


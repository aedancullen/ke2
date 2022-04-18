import os
import sys
import ke2
import h5py
import numpy as np

lut = {
    0: "C.",
    1: "C#",
    2: "D.",
    3: "D#",
    4: "E.",
    5: "F.",
    6: "F#",
    7: "G.",
    8: "G#",
    9: "A.",
    10: "A#",
    11: "B.",
}

h5file = h5py.File("dataset.hdf5", 'r')
names = h5file.keys()

chunksets = [h5file[name] for name in names if h5file[name].shape[2] // 4 in [8, 16, 32]]

max_nsteps = 0
for chunkset in chunksets:
    max_nsteps = max(max_nsteps, chunkset.shape[0] * chunkset.shape[2])

print(max_nsteps)

i = 0
for chunkset in chunksets:
    i += 1
    with open("export/chunkset" + str(i) + ".txt", 'w') as outfile:
        for repetition in range(max_nsteps // (chunkset.shape[0] * chunkset.shape[2])):
            for chunkidx in range(chunkset.shape[0]):
                for chunkstep in range(chunkset.shape[2]):
                    sv = chunkset[chunkidx, 0:12, chunkstep]
                    if np.max(sv) > 0.5:
                        nv = np.argmax(sv)
                        outfile.write(lut[nv])
                    else:
                        outfile.write('--')
                    sv = chunkset[chunkidx, 12:24, chunkstep]
                    if np.max(sv) > 0.5:
                        nv = np.argmax(sv)
                        outfile.write(lut[nv])
                    else:
                        outfile.write('--')
                    sv = chunkset[chunkidx, 24:36, chunkstep]
                    if np.max(sv) > 0.5:
                        nv = np.argmax(sv)
                        outfile.write(lut[nv])
                    else:
                        outfile.write('--')

                    outfile.write(' ')

h5file.close()

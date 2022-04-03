import os
import sys
import ke2
import h5py
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import time

audio_loader = AudioAdapter.default()
separator = Separator("spleeter:4stems")

def task(name, separation, sr):
    song_blob = ke2.preprocess(separation, sr)
    best_chunksize, best_offset, chunks = ke2.postprocess(song_blob, 4, [8, 16, 32, 6, 12, 24])
    return name, chunks

def writeout(results):
    name, data = results
    h5file = h5py.File("dataset.hdf5", 'a')
    dset = h5file.create_dataset(name, data=data)
    h5file.close()
    print("====>", name)

import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

for filename in os.listdir(sys.argv[1]):
    name = filename[:filename.index('.')]

    audio, sr = audio_loader.load(sys.argv[1] + '/' + filename)
    separation = separator.separate(audio)

    while len(pool._cache) > mp.cpu_count():
        time.sleep(0.1) # hacky ratelimit to prevent filling memory with waveforms awaiting pool

    pool.apply_async(task, (name, separation, int(sr)), callback=writeout)

pool.close()
pool.join()

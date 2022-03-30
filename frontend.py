import os
import sys
import ke2
import multiprocessing as mp
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

pool = mp.Pool(mp.cpu_count())

audio_loader = AudioAdapter.default()
separator = Separator("spleeter:4stems")

for filename in os.listdir(sys.argv[1]):

    audio, sr = audio_loader.load(sys.argv[1] + '/' + filename)
    separation = separator.separate(audio)

    while len(pool._cache) > mp.cpu_count():
        time.sleep(0.1) # hacky ratelimit to prevent filling memory with waveforms awaiting pool

    def task(separation, sr):
        song_blob = ke2.preprocess(separation, sr)
        return ke2.postprocess(song_blob, 4, 8, 32)

    pool.apply_async(task, (separation, int(sr)), callback=writeout)

#song_blob = ke2.preprocess(separation, int(sr))
#best_chunksize, best_offset, chunks = ke2.postprocess(song_blob, 4, 8, 32)
#print(best_chunksize, best_offset)
#print(len(chunks))
#print(chunks[0].shape)

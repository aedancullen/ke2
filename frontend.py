import sys
import ke2
from spleeter.audio.adapter import AudioAdapter

audio_loader = AudioAdapter.default()

audio, sr = audio_loader.load(sys.argv[1])

song_blob = ke2.preprocess(audio, int(sr))
best_chunksize, best_offset, chunks = ke2.postprocess(song_blob, 4, 8, 32)
print(best_chunksize, best_offset)
print(len(chunks))
print(chunks[0].shape)

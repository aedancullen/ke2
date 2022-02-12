import sys
import ke2
from spleeter.audio.adapter import AudioAdapter

audio_loader = AudioAdapter.default()

sr = 44100
audio, _ = audio_loader.load(sys.argv[1], sample_rate=sr)

song_blob = ke2.preprocess(audio, sr)
chunks, chunksize = ke2.postprocess(song_blob, 4, 4, 32)

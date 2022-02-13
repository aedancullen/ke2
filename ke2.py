import numpy as np
import librosa
from spleeter.separator import Separator

separator = Separator("spleeter:4stems")

def monomix(audio):
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    return audio

def preprocess(audio, sr):
    prediction = separator.separate(audio)
    
    orig = monomix(audio)
    vocals = monomix(prediction["vocals"])
    other = monomix(prediction["other"])
    bass = monomix(prediction["bass"])
    drums = monomix(prediction["drums"])
    
    orig = librosa.resample(orig, sr, 22050)
    vocals = librosa.resample(vocals, sr, 22050)
    other = librosa.resample(other, sr, 22050)
    bass = librosa.resample(bass, sr, 22050)
    drums = librosa.resample(drums, sr, 22050)
    
    song_blob = {}
    
    song_blob["orig_onset_strength"] = librosa.onset.onset_strength(orig)
    song_blob["vocals_onset_strength"] = librosa.onset.onset_strength(vocals)
    song_blob["other_onset_strength"] = librosa.onset.onset_strength(other)
    song_blob["bass_onset_strength"] = librosa.onset.onset_strength(bass)
    song_blob["drums_onset_strength"] = librosa.onset.onset_strength(drums)
    
    song_blob["vocals_stft"] = librosa.stft(vocals)
    song_blob["other_stft"] = librosa.stft(other)
    song_blob["bass_stft"] = librosa.stft(bass)
    song_blob["drums_stft"] = librosa.stft(drums)
    
    return song_blob

def postprocess(song_blob, beat_steps, chunksize_min, chunksize_max):
    tempo, beats = librosa.beat.beat_track(onset_envelope = song_blob["orig_onset_strength"])
    print(tempo)
    
    step_idx = np.interp(np.arange(0, beats.size, 1/beat_steps), np.arange(0, beats.size, 1), beats)
    step_idx = np.unique(np.rint(step_idx).astype(np.int32))
    
    vocals_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["vocals_stft"])**2)
    other_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["other_stft"])**2)
    bass_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["bass_stft"])**2)
    
    cat_feat = np.concatenate((vocals_chroma, other_chroma, bass_chroma), axis=0)
    
    step_feat = np.split(cat_feat, step_idx, axis=1)
    step_feat_mean = np.stack(list(map(lambda c: np.mean(c, axis=1), step_feat)), axis=-1)
    
    best_stat = -np.inf
    best_chunksize = None
    best_norm = None
    for i in range(chunksize_min, chunksize_max + 1):
        diff = step_feat_mean - np.roll(step_feat_mean, i * beat_steps, axis=1)
        norm = np.linalg.norm(diff, ord=2, axis=0)
        stat = -np.sum(norm, axis=0)
        if stat > best_stat:
            best_stat = stat
            best_chunksize = i
            best_norm = norm
    
    print(best_chunksize, best_stat)
    
    maxbeat = best_norm.size // beat_steps
    
    best_stat = -np.inf
    best_offset = None
    for i in range(best_chunksize, maxbeat - best_chunksize + 1):
        sum_l = np.sum(best_norm[(i - best_chunksize) * beat_steps : (i) * beat_steps], axis=0)
        sum_r = np.sum(best_norm[(i) * beat_steps : (i + best_chunksize) * beat_steps], axis=0)
        stat = sum_l - sum_r
        if stat > best_stat:
            best_stat = stat
            best_offset = i
    
    print(best_offset, best_stat)
    
    ef = librosa.frames_to_time(beats, sr=22050)[best_offset]
    print(ef/60, (ef % 60))
    
    return None, None

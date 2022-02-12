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
    
    vocals_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["vocals_stft"])**2)
    other_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["other_stft"])**2)
    bass_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["bass_stft"])**2)
    drums_melspectrogram = librosa.feature.melspectrogram(S=np.abs(song_blob["drums_stft"])**2)
    
    step_idx = np.interp(np.arange(0, beats.size, 1/beat_steps), np.arange(0, beats.size, 1), beats)
    step_idx = np.rint(step_idx).astype(np.int32)
    
    cat_feat = np.concatenate((vocals_chroma, other_chroma, bass_chroma), axis=0)
    step_feat = np.split(cat_feat, np.unique(step_idx), axis=1)

    step_feat_mean = np.stack(list(map(lambda c: np.mean(c, axis=1), step_feat)), axis=-1)
    
    best_stat = -np.inf
    best_chunksize = None
    best_norm = None
    for i in range(chunksize_min, chunksize_max):
        diff = step_feat_mean - np.roll(step_feat_mean, i * beat_steps, axis=1)
        norm = np.linalg.norm(diff, ord=2, axis=0)
        stat = np.var(norm)
        if stat > best_stat:
            best_stat = stat
            best_chunksize = i
            best_norm = norm
    
    print(best_chunksize, best_stat)
    
    maxbeat = step_feat_mean.shape[1] // beat_steps
    
    best_stat_h = -np.inf
    best_offset_h = None
    best_stat_l = np.inf
    best_offset_l = None
    for i in range(best_chunksize, maxbeat - best_chunksize):
        sum_l = np.sum(best_norm[(i - best_chunksize) * beat_steps : (i) * beat_steps])
        sum_r = np.sum(best_norm[(i) * beat_steps : (i + best_chunksize) * beat_steps])
        stat = sum_l - sum_r
        if stat > best_stat_h:
            best_stat_h = stat
            best_offset_h = i
        if stat < best_stat_l:
            best_stat_l = stat
            best_offset_l = i
    
    print(best_offset_h, best_stat_h)
    print(best_offset_l, best_stat_l)

    return None, None

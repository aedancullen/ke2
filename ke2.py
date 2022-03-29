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

    #song_blob["vocals_onset_strength"] = librosa.onset.onset_strength(vocals)
    #song_blob["other_onset_strength"] = librosa.onset.onset_strength(other)
    #song_blob["bass_onset_strength"] = librosa.onset.onset_strength(bass)
    song_blob["drums_onset_strength"] = librosa.onset.onset_strength(drums)
    
    #song_blob["vocals_stft"] = librosa.stft(vocals)
    #song_blob["other_stft"] = librosa.stft(other)
    #song_blob["bass_stft"] = librosa.stft(bass)
    #song_blob["drums_stft"] = librosa.stft(drums)

    song_blob["vocals_cqt"] = librosa.cqt(vocals)
    song_blob["other_cqt"] = librosa.cqt(other)
    song_blob["bass_cqt"] = librosa.cqt(bass)
    #song_blob["drums_cqt"] = librosa.cqt(drums)

    song_blob["vocals_pyin"] = librosa.pyin(vocals, librosa.note_to_hz("C2"), librosa.note_to_hz("C7"))
    song_blob["other_pyin"] = librosa.pyin(other, librosa.note_to_hz("C2"), librosa.note_to_hz("C7"))
    song_blob["bass_pyin"] = librosa.pyin(bass, librosa.note_to_hz("C2"), librosa.note_to_hz("C7"))
    #song_blob["drums_pyin"] = librosa.pyin(drums, librosa.note_to_hz("C2"), librosa.note_to_hz("C7"))
    
    return song_blob

def postprocess(song_blob, beat_steps, chunksize_min, chunksize_max):

    def gen_feat_perstep(in_feat, step_idxs):
        step_feat = np.split(in_feat, step_idxs, axis=1)[1:]
        step_feat_mean = np.stack(list(map(lambda c: np.mean(c, axis=1), step_feat)), axis=-1)
        return np.nan_to_num(step_feat_mean)

    def extract_chunks_from(arr, chunksize, offset, n_chunks):
        chunks = []
        for i in range(n_chunks):
            chunks.append(arr[offset + i*chunksize : offset + (i+1)*chunksize])
        return chunks

    tempo, beats = librosa.beat.beat_track(onset_envelope = song_blob["drums_onset_strength"])

    step_idxs = np.interp(np.arange(0, beats.size, 1/beat_steps), np.arange(0, beats.size, 1), beats)
    step_idxs = np.rint(step_idxs).astype(np.int32)
    
    vocals_chroma = librosa.feature.chroma_cqt(C=np.abs(song_blob["vocals_cqt"]))
    other_chroma = librosa.feature.chroma_cqt(C=np.abs(song_blob["other_cqt"]))
    bass_chroma = librosa.feature.chroma_cqt(C=np.abs(song_blob["bass_cqt"]))

    vocals_chroma_perstep = gen_feat_perstep(vocals_chroma, step_idxs)
    other_chroma_perstep = gen_feat_perstep(other_chroma, step_idxs)
    bass_chroma_perstep = gen_feat_perstep(bass_chroma, step_idxs)

    cat_feat_perstep = np.concatenate((
        vocals_chroma_perstep,
        other_chroma_perstep,
        bass_chroma_perstep
    ), axis=0)
    
    best_stat = -np.inf
    best_chunksize = None
    best_norm = None
    for i in range(chunksize_min, chunksize_max + 1):
        diff = cat_feat_perstep - np.roll(cat_feat_perstep, i * beat_steps, axis=1)
        norm = np.linalg.norm(diff, ord=2, axis=0)
        stat = -np.sum(norm, axis=0)
        if stat > best_stat:
            best_stat = stat
            best_chunksize = i
            best_norm = norm
    
    maxbeat = best_norm.size // beat_steps
    
    best_stat = -np.inf
    best_offset = None
    statsx = []
    stats = []
    for i in range(0, maxbeat - 3*best_chunksize):
        sum_prev = np.sum(best_norm[(i) * beat_steps : (i + best_chunksize) * beat_steps], axis=0)
        sum_next1 = np.sum(best_norm[(i + best_chunksize) * beat_steps : (i + 2*best_chunksize) * beat_steps], axis=0)
        sum_next2 = np.sum(best_norm[(i + 2*best_chunksize) * beat_steps : (i + 3*best_chunksize) * beat_steps], axis=0)
        stat = 2*sum_next1 - sum_prev - sum_next2
        statsx.append(i)
        stats.append(stat)
        if stat > best_stat:
            best_stat = stat
            best_offset = i

    vocals_pyin = song_blob["vocals_pyin"][0].reshape((1, -1))
    other_pyin = song_blob["other_pyin"][0].reshape((1, -1))
    bass_pyin = song_blob["bass_pyin"][0].reshape((1, -1))

    vocals_pyin_perstep = gen_feat_perstep(vocals_pyin, step_idxs)
    other_pyin_perstep = gen_feat_perstep(other_pyin, step_idxs)
    bass_pyin_perstep = gen_feat_perstep(bass_pyin, step_idxs)

    cat_all_perstep = np.concatenate((
        vocals_chroma_perstep,
        other_chroma_perstep,
        bass_chroma_perstep,
        vocals_pyin_perstep,
        other_pyin_perstep,
        bass_pyin_perstep,
    ), axis=0)

    chunks = extract_chunks_from(cat_all_perstep, best_chunksize, best_offset, 3)
    
    return best_chunksize, best_offset, chunks

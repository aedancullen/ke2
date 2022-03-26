import numpy as np
import librosa
import librosa.display
from spleeter.separator import Separator
import sys
import soundfile as sf

import matplotlib.pyplot as plt

plt.style.use('dark_background')

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "#4e5567",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "#4e5567",
    "figure.edgecolor": "#4e5567",
    "savefig.facecolor": "#4e5567",
    "savefig.edgecolor": "#4e5567"})

xl = 100
yl = 110

e = None

separator = Separator("spleeter:4stems")

def axesoff(plt):
    right_side = plt.gca().spines["right"]
    right_side.set_visible(False)
    top_side = plt.gca().spines["top"]
    top_side.set_visible(False)
    bottom_side = plt.gca().spines["bottom"]
    bottom_side.set_visible(False)
    plt.gca().axes.xaxis.set_visible(False)

def monomix(audio):
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    return audio

def preprocess(audio, sr):
    global e
    
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
    
    e = orig
    
    plt.figure(figsize=(20, 2))
    plt.plot(orig)
    axesoff(plt)
    plt.savefig("orig.png")
    plt.figure(figsize=(20, 2))
    plt.plot(vocals)
    axesoff(plt)
    plt.savefig("vocals.png")
    plt.figure(figsize=(20, 2))
    plt.plot(other)
    axesoff(plt)
    plt.savefig("other.png")
    plt.figure(figsize=(20, 2))
    plt.plot(bass)
    axesoff(plt)
    plt.savefig("bass.png")
    plt.figure(figsize=(20, 2))
    plt.plot(drums)
    axesoff(plt)
    plt.savefig("drums.png")
    #sys.exit()
    
    song_blob = {}
    
    #song_blob["orig_onset_strength"] = librosa.onset.onset_strength(orig)
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
    tempo, beats = librosa.beat.beat_track(onset_envelope = song_blob["drums_onset_strength"])
    print(tempo)
    clicks=  librosa.clicks(frames=beats, length = e.size)
    out = e/2 + clicks/2
    sf.write("clicks.wav", out, 22050)
    
    times = librosa.times_like(song_blob["drums_onset_strength"], sr=22050, hop_length=512)
    plt.figure(figsize=(20, 2))
    plt.xlim([xl, yl])
    plt.plot(times, librosa.util.normalize(song_blob["drums_onset_strength"]))
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='b', linestyle='solid', label='Beats')
    axesoff(plt)
    plt.savefig("beats.png")
    
    step_idx = np.interp(np.arange(0, beats.size, 1/beat_steps), np.arange(0, beats.size, 1), beats)
    step_idx = np.rint(step_idx).astype(np.int32)
    
    plt.figure(figsize=(20, 2))
    plt.xlim([xl, yl])
    plt.plot(times, librosa.util.normalize(song_blob["drums_onset_strength"]))
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='b', linestyle='solid', label='Beats')
    plt.vlines(times[step_idx], 0, 1, alpha=0.5, color='y', linestyle='solid', label='Steps')
    axesoff(plt)
    plt.savefig("steps.png")
    
    vocals_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["vocals_stft"])**2)
    other_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["other_stft"])**2)
    bass_chroma = librosa.feature.chroma_stft(S=np.abs(song_blob["bass_stft"])**2)
    
    plt.figure(figsize=(20, 2))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(vocals_chroma[:,:], y_axis='chroma', cmap="Purples"))
    axesoff(plt)
    plt.savefig("cv.png")
    plt.figure(figsize=(20, 2))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(other_chroma[:,:], y_axis='chroma', cmap="Purples"))
    axesoff(plt)
    plt.savefig("co.png")
    plt.figure(figsize=(20, 2))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(bass_chroma[:,:], y_axis='chroma', cmap="Purples"))
    axesoff(plt)
    plt.savefig("cb.png")
    
    cat_feat = np.concatenate((bass_chroma, other_chroma, vocals_chroma), axis=0)
    
    step_feat = np.split(cat_feat, step_idx, axis=1)[1:]
    step_feat_mean = np.stack(list(map(lambda c: np.mean(c, axis=1), step_feat)), axis=-1)
    step_feat_mean = np.nan_to_num(step_feat_mean)
    
    plt.figure(figsize=(20, 5))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(step_feat_mean[:,:], y_axis='chroma', cmap="Purples"))
    #axesoff(plt)
    plt.savefig("sfm.png")
    
    best_stat = -np.inf
    best_chunksize = None
    best_norm = None
    statsx = []
    stats = []
    for i in range(chunksize_min, chunksize_max + 1):
        diff = step_feat_mean - np.roll(step_feat_mean, i * beat_steps, axis=1)
        norm = np.linalg.norm(diff, ord=2, axis=0)
        stat = -np.sum(norm, axis=0)
        statsx.append(i)
        stats.append(-stat)
        if stat > best_stat:
            best_stat = stat
            best_chunksize = i
            best_norm = norm
            
        plt.figure(figsize=(20, 0.5))
        #plt.xlim([xl, yl])
        plt.colorbar(librosa.display.specshow(norm.reshape((1, -1)), cmap="coolwarm"))
        axesoff(plt)
        plt.savefig("norm" + str(i) + ".png")
    
    print(best_chunksize, best_stat)
    
    plt.figure(figsize=(20, 0.5))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(best_norm.reshape((1, -1)), cmap="coolwarm"))
    axesoff(plt)
    plt.savefig("norm_best.png")
    
    plt.figure(figsize=(20, 2))
    #plt.xlim([xl, yl])
    plt.plot(statsx, stats)
    #axesoff(plt)
    right_side = plt.gca().spines["right"]
    right_side.set_visible(False)
    top_side = plt.gca().spines["top"]
    top_side.set_visible(False)
    plt.xticks(statsx)
    plt.savefig("stats.png")
    
    maxbeat = best_norm.size // beat_steps
    
    best_stat = -np.inf
    best_offset = None
    statsx = []
    stats = []
    for i in range(best_chunksize*2, maxbeat - best_chunksize + 1):
        sum_1 = np.sum(best_norm[(i - best_chunksize*2) * beat_steps : (i - best_chunksize) * beat_steps], axis=0)
        sum_l = np.sum(best_norm[(i - best_chunksize) * beat_steps : (i) * beat_steps], axis=0)
        sum_r = np.sum(best_norm[(i) * beat_steps : (i + best_chunksize) * beat_steps], axis=0)
        stat = 2 * sum_l - sum_r - sum_1
        statsx.append(i)
        stats.append(stat)
        if stat > best_stat:
            best_stat = stat
            best_offset = i
    
    print(best_offset, best_stat)
    
    plt.figure(figsize=(20, 2))
    #plt.xlim([xl, yl])
    plt.plot(statsx, stats)
    #axesoff(plt)
    right_side = plt.gca().spines["right"]
    right_side.set_visible(False)
    top_side = plt.gca().spines["top"]
    top_side.set_visible(False)
    #plt.xticks(statsx)
    plt.savefig("stats_offset.png")
    
    ef = librosa.frames_to_time(beats, sr=22050)[best_offset]
    print(ef/60, (ef % 60))
    
    sec = step_feat_mean[:, (best_offset-best_chunksize*2) * beat_steps : (best_offset-best_chunksize) * beat_steps]
    
    plt.figure(figsize=(20, 5))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(sec, y_axis='chroma', cmap="Purples"))
    #axesoff(plt)
    plt.savefig("section.png")
    
    sec = sec.reshape((3, 12, -1))
    b = np.zeros_like(sec)
    for i in range(3):
        b[i, sec[i].argmax(0), np.arange(sec.shape[2])] = 1
        
    b = b.reshape((36, -1))
    
    plt.figure(figsize=(20, 5))
    #plt.xlim([xl, yl])
    plt.colorbar(librosa.display.specshow(b, y_axis='chroma', cmap="Purples"))
    #axesoff(plt)
    plt.savefig("section2.png")
    
    return None, None

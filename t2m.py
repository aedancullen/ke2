from mido import Message, MidiFile, MidiTrack
import librosa

mid = MidiFile()
track_v = MidiTrack()
track_o = MidiTrack()
track_b = MidiTrack()
mid.tracks.append(track_v)
mid.tracks.append(track_o)
mid.tracks.append(track_b)

ticks = mid.ticks_per_beat // 4

notes_v = []
notes_o = []
notes_b = []

def t2m(text):
    tokens = text.split(' ')
    last_v = None
    last_o = None
    last_b = None
    for i, token in enumerate(tokens):
        if len(token) != 6:
            continue
        v = token[0:2].strip('.')
        o = token[2:4].strip('.')
        b = token[4:6].strip('.')
        v = librosa.note_to_midi(v + '3') if v != "--" else last_v
        o = librosa.note_to_midi(o + '4') if o != "--" else last_o
        b = librosa.note_to_midi(b + '5') if b != "--" else last_b
        if v != last_v:
            notes_v.append([v, ticks])
        else:
            notes_v[-1][1] += ticks
        if o != last_o:
            notes_o.append([o, ticks])
        else:
            notes_o[-1][1] += ticks
        if b != last_b:
            notes_b.append([b, ticks])
        else:
            notes_b[-1][1] += ticks
        last_v = v
        last_o = o
        last_b = b
    for note in notes_v:
        track_v.append(Message("note_on", note=note[0], velocity=64, time=0))
        track_v.append(Message("note_off", note=note[0], velocity=127, time=note[1]))
    for note in notes_o:
        track_o.append(Message("note_on", note=note[0], velocity=64, time=0))
        track_o.append(Message("note_off", note=note[0], velocity=127, time=note[1]))
    for note in notes_b:
        track_b.append(Message("note_on", note=note[0], velocity=64, time=0))
        track_b.append(Message("note_off", note=note[0], velocity=127, time=note[1]))
    mid.save("output.mid")

with open("output.txt", 'r') as infile:
    text = infile.read()

print(text)
t2m(text)


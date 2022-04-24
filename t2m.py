from mido import Message, MidiFile, MidiTrack
import librosa

mid = MidiFile()
track_v = MidiTrack()
track_o = MidiTrack()
track_b = MidiTrack()
mid.tracks.append(track_v)
mid.tracks.append(track_o)
mid.tracks.append(track_b)

def t2m(text):
    tokens = text.split(' ')
    last_v = None
    last_o = None
    last_b = None
    for i, token in enumerate(tokens):
        if len(token) != 6:
            continue
        ticks = (mid.ticks_per_beat // 4) * i
        v = token[0:2].strip('.')
        o = token[2:4].strip('.')
        b = token[4:6].strip('.')
        v = librosa.note_to_midi(v + '3') if v != "--" else last_v
        o = librosa.note_to_midi(o + '4') if o != "--" else last_o
        b = librosa.note_to_midi(b + '5') if b != "--" else last_b
        if v != last_v:
            if last_v != None:
                track_v.append(Message("note_off", note=last_v, velocity=127, time=ticks))
            track_v.append(Message("note_on", note=v, velocity=64, time=ticks))
        if o != last_o:
            if last_o != None:
                track_o.append(Message("note_off", note=last_o, velocity=127, time=ticks))
            track_o.append(Message("note_on", note=o, velocity=64, time=ticks))
        if b != last_b:
            if last_b != None:
                track_b.append(Message("note_off", note=last_b, velocity=127, time=ticks))
            track_b.append(Message("note_on", note=b, velocity=64, time=ticks))
        last_v = v
        last_o = o
        last_b = b
    mid.save("output.mid")

with open("output.txt", 'r') as infile:
    text = infile.read()

print(text)
t2m(text)

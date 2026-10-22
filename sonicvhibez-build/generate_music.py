import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pathlib import Path

OUT = Path("sonicvhibez_original_energetic_bed.wav")
sr = 44100
duration = 22.0
n = int(sr * duration)
t = np.arange(n, dtype=np.float64) / sr
rng = np.random.default_rng(20261022)
mono = np.zeros(n, dtype=np.float64)

def add_at(start, sig, gain=1.0):
    i = int(start * sr)
    if i >= n or i < 0:
        return
    end = min(n, i + len(sig))
    if end > i:
        mono[i:end] += gain * sig[:end-i]

def drum_kick(length=0.23):
    tt = np.arange(int(sr*length)) / sr
    phase = 2*np.pi*(82*tt + 28*(1-np.exp(-tt*22)))
    env = np.exp(-tt*19)
    return np.sin(phase)*env + 0.18*rng.normal(0,1,len(tt))*np.exp(-tt*45)

def tom(length=0.20, f0=170):
    tt = np.arange(int(sr*length)) / sr
    phase = 2*np.pi*(f0*tt + 22*(1-np.exp(-tt*16)))
    env = np.exp(-tt*18)
    noise = sosfilt(butter(2, 1800, fs=sr, output='sos'), rng.normal(0,1,len(tt)))
    return 0.8*np.sin(phase)*env + 0.12*noise*np.exp(-tt*34)

def clap(length=0.11):
    m = int(sr*length)
    tt = np.arange(m)/sr
    noise = rng.normal(0,1,m)
    filt = sosfilt(butter(2, [850, 6500], fs=sr, btype='bandpass', output='sos'), noise)
    env = np.exp(-tt*34) * (1 - np.exp(-tt*500))
    return filt * env

def shaker(length=0.055):
    m = int(sr*length)
    tt = np.arange(m)/sr
    noise = sosfilt(butter(2, 4000, fs=sr, btype='highpass', output='sos'), rng.normal(0,1,m))
    return noise * np.exp(-tt*70)

def pluck(freq, length=0.30):
    m = int(sr*length)
    tt = np.arange(m)/sr
    env = np.exp(-tt*9.5)
    wave = (np.sin(2*np.pi*freq*tt) + 0.34*np.sin(2*np.pi*freq*2.01*tt) + 0.11*np.sin(2*np.pi*freq*3.97*tt))
    wave *= env * np.minimum(1, tt*220)
    return wave

def bass_note(freq, length=0.45):
    m = int(sr*length)
    tt = np.arange(m)/sr
    env = np.exp(-tt*5.0) * np.minimum(1, tt*35)
    return (np.sin(2*np.pi*freq*tt) + 0.15*np.sin(2*np.pi*freq*2*tt)) * env

bpm = 112
beat = 60.0 / bpm
bar = 4*beat
for b in range(int(np.ceil(duration/bar))):
    base = b*bar
    for j in range(4):
        st = base + j*beat
        if st >= duration: continue
        if j in (0,2): add_at(st, drum_kick(), 0.68 if j == 0 else 0.56)
        if j in (1,3): add_at(st, clap(), 0.25)
        if j in (1,3): add_at(st + 0.16, tom(f0=185 if j==1 else 155), 0.23)
        roots = [110.0, 130.81, 146.83, 98.0]
        root = roots[(b // 2) % len(roots)]
        if j in (0,2): add_at(st, bass_note(root, 0.46), 0.33)
        else: add_at(st, bass_note(root*1.5, 0.35), 0.22)
    for k in range(8):
        st = base + k*beat/2 + (0.022 if k%2 else 0.0)
        add_at(st, shaker(), 0.045 if k%2 else 0.06)
    notes = [440.00, 523.25, 587.33, 659.25, 783.99, 659.25, 587.33, 523.25]
    for k, freq in enumerate(notes):
        st = base + k*beat/2 + (0.02 if k%2 else 0.0)
        if st < duration:
            add_at(st, pluck(freq, 0.30), 0.105 if k not in (0,4) else 0.13)
    if b % 2 == 1:
        for off, freq in [(1.5*beat, 880.0), (3.5*beat, 783.99)]:
            if base+off < duration:
                add_at(base+off, pluck(freq, 0.22), 0.055)

fade_in = np.clip(t/0.45, 0, 1)
fade_out = np.clip((duration-t)/1.55, 0, 1)
mono *= fade_in * fade_out
mono = np.tanh(mono * 1.25)
peak = np.max(np.abs(mono)) or 1.0
mono *= 0.30 / peak
right = np.roll(mono, int(0.008*sr))*0.97
stereo = np.column_stack([mono, right]).astype(np.float32)
sf.write(OUT, stereo, sr, subtype='PCM_16')
print(f"Wrote {OUT} ({duration:.2f}s, {sr} Hz, stereo)")

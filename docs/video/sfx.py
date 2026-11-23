"""Synthesized UI sounds (no third-party samples): key, tick, pop, click, whoosh, rise."""
import wave
import numpy as np

SR = 44100
rng = np.random.default_rng(7)


def env(n, attack=0.002, decay=0.03):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / decay)


def sweep(f0, f1, dur):
    t = np.arange(int(dur * SR)) / SR
    f = f0 + (f1 - f0) * t / dur
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def key():
    n = int(0.035 * SR)
    x = np.diff(rng.standard_normal(n + 1)) * env(n, 0.0005, 0.006)
    x += 0.5 * np.sin(2 * np.pi * rng.uniform(180, 240) * np.arange(n) / SR) * env(n, 0.001, 0.01)
    return x * rng.uniform(0.18, 0.28)


def tick():
    n = int(0.05 * SR)
    return 0.22 * np.sin(2 * np.pi * 1700 * np.arange(n) / SR) * env(n, 0.001, 0.012)


def pop():
    x = sweep(700, 320, 0.09)
    return 0.22 * x * env(len(x), 0.003, 0.03)


def click():
    n = int(0.07 * SR)
    t = np.arange(n) / SR
    x = 0.6 * np.sin(2 * np.pi * 2300 * t) * env(n, 0.0005, 0.004) + np.sin(2 * np.pi * 950 * t) * env(n, 0.001, 0.02)
    return 0.32 * x


def whoosh():
    n = int(0.7 * SR)
    x = np.convolve(rng.standard_normal(n), np.ones(40) / 40, "same")
    t = np.arange(n) / n
    return 0.9 * x * np.sin(np.pi * t) ** 2


def rise():
    x = sweep(420, 880, 0.45)
    return 0.12 * x * np.sin(np.pi * np.arange(len(x)) / len(x))


SOUNDS = {"key": key, "tick": tick, "pop": pop, "click": click, "whoosh": whoosh, "rise": rise}


def mix(events, dur, path):
    y = np.zeros(int((dur + 1) * SR))
    for t, kind in events:
        x = SOUNDS[kind]()
        i = int(t * SR)
        if i < len(y):
            y[i:i + len(x)] += x[: len(y) - i]
    y = y[: int(dur * SR)]
    y = y / max(1.0, np.abs(y).max() / 0.9)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype("<i2").tobytes())

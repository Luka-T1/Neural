"""Spike analysis helpers."""
import numpy as np


def detect_spikes(t, V, threshold=0.0):
    """Times (ms, linearly interpolated) of upward threshold crossings."""
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    idx = np.nonzero((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    if idx.size == 0:
        return np.zeros(0)
    frac = (threshold - V[idx]) / (V[idx + 1] - V[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])

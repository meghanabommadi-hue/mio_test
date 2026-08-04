"""Post-decode audio smoothing for isolated glitches.

MioCodec's decode() runs the entire content-token sequence through one forward
pass (no internal chunking/windowing), so there is no structural "chunk boundary"
to smooth. What can happen instead is that the LLM occasionally samples an
outlier speech token, which the codec faithfully renders as a rough/sharp
transient at that specific point in the sequence -- sparse and randomly placed,
not periodic. This module detects those transients by the same discontinuity
heuristic used to score takes, then softens each one with a short crossfade
rather than re-generating the whole clip.
"""

import numpy as np


def find_glitches(data: np.ndarray, percentile: float = 99.9, min_jump: float = 0.15) -> np.ndarray:
    """Return sample indices where |x[i] - x[i-1]| is an outlier discontinuity."""
    diff = np.abs(np.diff(data))
    if diff.size == 0:
        return np.array([], dtype=int)
    threshold = max(np.percentile(diff, percentile), min_jump)
    return np.where(diff > threshold)[0] + 1  # index of the sample AFTER the jump


def smooth_glitches(
    data: np.ndarray,
    sample_rate: int,
    window_ms: float = 4.0,
    percentile: float = 99.9,
    min_jump: float = 0.15,
) -> np.ndarray:
    """Crossfade a short window around each detected glitch to remove the click
    while leaving the rest of the waveform untouched.

    For each glitch at sample i, linearly blends the surrounding +/-window_ms of
    audio into a straight line between its endpoints -- cheap, local, and doesn't
    touch anything but the flagged samples, so it won't dull unaffected audio.
    """
    data = data.copy()
    glitch_indices = find_glitches(data, percentile=percentile, min_jump=min_jump)
    if glitch_indices.size == 0:
        return data

    half_window = max(1, int(sample_rate * window_ms / 1000 / 2))
    n = len(data)

    for idx in glitch_indices:
        start = max(0, idx - half_window)
        end = min(n, idx + half_window)
        if end - start < 2:
            continue
        ramp = np.linspace(data[start], data[end - 1], end - start)
        data[start:end] = ramp

    return data

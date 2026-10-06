"""Pseudo-random (spread-spectrum) transmitter waveforms.

A maximal-length sequence (m-sequence) of order n has period N = 2^n - 1
chips, values +-1, and a two-valued periodic autocorrelation (N at lag 0,
-1 elsewhere).  Its line spectrum is flat at all harmonics k/T (T = N/f_c)
under a sinc^2 envelope with nulls at multiples of the chip rate f_c, so a
single periodic record excites ~N/2 frequencies at once with a low crest
factor: the basis of spread-spectrum induced polarization (SSIP).
"""
from __future__ import annotations

import numpy as np

# primitive polynomial taps (Fibonacci LFSR, 1-based) for orders 3..20
_TAPS = {
    3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5),
    10: (10, 7), 11: (11, 9), 12: (12, 11, 10, 4), 13: (13, 12, 11, 8), 14: (14, 13, 12, 2),
    15: (15, 14), 16: (16, 15, 13, 4), 17: (17, 14), 18: (18, 11), 19: (19, 18, 17, 14),
    20: (20, 17),
}


def m_sequence(order: int, seed: int = 1) -> np.ndarray:
    """Binary m-sequence of length 2^order - 1, values +-1 (float)."""
    if order not in _TAPS:
        raise ValueError(f"order must be in {sorted(_TAPS)}")
    taps = _TAPS[order]
    n = 2 ** order - 1
    state = [(seed >> i) & 1 for i in range(order)]
    if not any(state):
        state[0] = 1
    out = np.empty(n)
    for i in range(n):
        out[i] = state[-1]
        fb = 0
        for t in taps:
            fb ^= state[t - 1]
        state = [fb] + state[:-1]
    return 1.0 - 2.0 * out


class PRBSWaveform:
    """Periodic m-sequence current waveform.

    Parameters
    ----------
    order : m-sequence order (N = 2^order - 1 chips per period).
    chip_rate : chips per second f_c (Hz).
    oversampling : samples per chip (sampling rate fs = chip_rate * oversampling).
    amplitude : current amplitude (A).
    """

    def __init__(self, order=9, chip_rate=64.0, oversampling=8, amplitude=1.0, seed=1):
        self.order = int(order)
        self.chip_rate = float(chip_rate)
        self.oversampling = int(oversampling)
        self.amplitude = float(amplitude)
        self.chips = m_sequence(self.order, seed)

    @property
    def n_chips(self):
        return len(self.chips)

    @property
    def fs(self):
        return self.chip_rate * self.oversampling

    @property
    def period(self):
        return self.n_chips / self.chip_rate

    @property
    def samples_per_period(self):
        return self.n_chips * self.oversampling

    @property
    def f0(self):
        """Fundamental frequency (harmonic spacing) 1/T."""
        return 1.0 / self.period

    def one_period(self):
        return self.amplitude * np.repeat(self.chips, self.oversampling)

    def harmonics(self):
        """Frequencies of the rfft bins of one period."""
        return np.fft.rfftfreq(self.samples_per_period, 1.0 / self.fs)

    def usable_band(self, rel_power=0.1):
        """(fmin, fmax) where the line power exceeds ``rel_power`` of its maximum."""
        f = self.harmonics()[1:]
        p = np.abs(np.fft.rfft(self.one_period()))[1:] ** 2
        ok = p > rel_power * p.max()
        last = np.argmax(~ok) if not ok.all() else len(ok)
        return f[0], f[max(last - 1, 0)]

    def describe(self):
        fmin, fmax = self.usable_band()
        return (f"m-sequence order {self.order} ({self.n_chips} chips), chip rate {self.chip_rate:g} Hz, "
                f"fs {self.fs:g} Hz, period {self.period:g} s, usable band {fmin:.3g}-{fmax:.3g} Hz")

"""Spread-spectrum IP processing: from raw current/voltage records to spectra.

For every channel the steps are

1. detrend (removes self-potential drift),
2. cut into whole periods of the transmitted m-sequence (the first,
   transient period is discarded),
3. per period p, cross-spectral transfer-function estimate on each
   analysis band b (log-spaced groups of harmonics, excluding
   powerline-contaminated lines and sinc nulls)::

        Z_p(b) = sum_k V_p(f_k) I_p(f_k)^* / sum_k |I_p(f_k)|^2

   This is the least-squares deconvolution of V by I, equivalent to
   cross-correlating with the m-sequence,
4. robust stack over periods in complex-log space (median, MAD), giving
   empirical, data-driven standard errors of ln|Z| and phase separately.

These errors feed the inversion's data weights directly, which is the key to
a reproducible, statistically consistent workflow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend

from .waveform import PRBSWaveform


@dataclass
class SpectralData:
    """Processed multi-frequency complex impedances for a survey."""

    freqs: np.ndarray  # (n_f,)
    Z: np.ndarray  # (n_f, n_data) complex transfer impedance (ohm)
    err_amp: np.ndarray  # (n_f, n_data) std error of ln|Z|
    err_phase: np.ndarray  # (n_f, n_data) std error of phase (rad)
    n_periods: int = 0

    def save(self, path):
        np.savez(path, freqs=self.freqs, Z=self.Z, err_amp=self.err_amp,
                 err_phase=self.err_phase, n_periods=self.n_periods)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["freqs"], d["Z"], d["err_amp"], d["err_phase"], int(d["n_periods"]))

    def to_csv(self, path, survey):
        rows = []
        for k, f in enumerate(self.freqs):
            for i, (a, b, m, n) in enumerate(survey.abmn):
                z = self.Z[k, i]
                rows.append([a, b, m, n, f, abs(z), np.angle(z * np.sign(z.real)) * 1e3,
                             self.err_amp[k, i], self.err_phase[k, i] * 1e3])
        np.savetxt(path, np.array(rows), delimiter=",", fmt="%.8g",
                   header="a,b,m,n,freq_hz,amp_ohm,phase_mrad,err_lnamp,err_phase_mrad", comments="")

    @classmethod
    def from_csv(cls, path, survey):
        """Read a CSV in the format written by :meth:`to_csv` (sign taken from the survey K)."""
        t = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        freqs = np.unique(t[:, 4])
        nd = survey.n_data
        Z = np.zeros((len(freqs), nd), complex)
        ea, ep = np.zeros_like(Z, float), np.zeros_like(Z, float)
        key = {tuple(r): i for i, r in enumerate(survey.abmn)}
        sgn = np.sign(survey.geometric_factor())
        for r in t:
            k = np.searchsorted(freqs, r[4])
            i = key[tuple(r[:4].astype(int))]
            Z[k, i] = sgn[i] * r[5] * np.exp(1j * r[6] * 1e-3)
            ea[k, i], ep[k, i] = r[7], r[8] * 1e-3
        return cls(freqs, Z, ea, ep)

    def apply_error_floor(self, amp_floor=0.01, phase_floor=1e-3, amp_rel=0.0):
        """Add floors in quadrature (ln-amplitude floor ~ relative error, phase in rad)."""
        ea = np.sqrt(self.err_amp ** 2 + amp_floor ** 2 + amp_rel ** 2)
        ep = np.sqrt(self.err_phase ** 2 + phase_floor ** 2)
        return SpectralData(self.freqs, self.Z, ea, ep, self.n_periods)

    def select(self, keep_freq=None, keep_data=None):
        kf = slice(None) if keep_freq is None else keep_freq
        kd = slice(None) if keep_data is None else keep_data
        return SpectralData(self.freqs[kf], self.Z[kf][:, kd], self.err_amp[kf][:, kd],
                            self.err_phase[kf][:, kd], self.n_periods)


def analysis_bands(wave: PRBSWaveform, n_bands=8, fmin=None, fmax=None, notch=(50.0,), notch_width=0.6,
                   rel_power=0.1):
    """Group usable harmonics into log-spaced bands.

    Returns (band_freqs, list of harmonic index arrays). Harmonics with line
    power below ``rel_power`` of the maximum (sinc nulls) and within
    ``notch_width`` Hz of a powerline harmonic are excluded.
    """
    f = wave.harmonics()
    P = np.abs(np.fft.rfft(wave.one_period())) ** 2
    lo, hi = wave.usable_band(rel_power)
    fmin = lo if fmin is None else max(fmin, lo)
    fmax = hi if fmax is None else min(fmax, hi)
    ok = (f >= fmin * 0.999) & (f <= fmax * 1.001) & (P > rel_power * P[1:].max())
    for f_pl in notch or ():
        for h in np.arange(f_pl, f.max() + f_pl, f_pl):
            ok &= np.abs(f - h) > notch_width
    edges = np.geomspace(fmin * 0.999, fmax * 1.001, n_bands + 1)
    bands, centers = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(ok & (f >= lo_) & (f < hi_))
        if len(idx):
            bands.append(idx)
            centers.append(np.exp(np.mean(np.log(f[idx]))))
    return np.array(centers), bands


def _periods(x, spp, skip=1):
    x = detrend(np.asarray(x, float), type="linear")
    n = len(x) // spp
    return x[: n * spp].reshape(n, spp)[skip:]


def transfer_function(current, voltage, wave: PRBSWaveform, bands, skip_periods=1, polarity=None):
    """Robust per-band transfer function for one channel.

    Returns (Z (n_b,), err_lnamp (n_b,), err_phase (n_b,), n_periods).
    """
    spp = wave.samples_per_period
    I = np.fft.rfft(_periods(current, spp, skip_periods), axis=1)
    V = np.fft.rfft(_periods(voltage, spp, skip_periods), axis=1)
    P = len(I)
    if P < 2:
        raise ValueError("need at least 2 full periods after skipping the transient")
    Zp = np.empty((P, len(bands)), complex)
    for b, idx in enumerate(bands):
        Zp[:, b] = (V[:, idx] * I[:, idx].conj()).sum(1) / (np.abs(I[:, idx]) ** 2).sum(1)
    s = np.sign(np.median(Zp.real)) if polarity is None else polarity
    s = 1.0 if s == 0 else s
    L = np.log(s * Zp)  # complex log per period: ln|Z| + i phase
    med = np.median(L.real, 0) + 1j * np.median(L.imag, 0)
    mad_r = 1.4826 * np.median(np.abs(L.real - med.real), 0)
    mad_i = 1.4826 * np.median(np.abs(L.imag - med.imag), 0)
    # standard error of the median ~ 1.2533 * sigma / sqrt(P)
    se_r = 1.2533 * mad_r / np.sqrt(P)
    se_i = 1.2533 * mad_i / np.sqrt(P)
    return s * np.exp(med), se_r, se_i, P


def process_dataset(current, voltage, tx_index, wave: PRBSWaveform, n_bands=8, fmin=None, fmax=None,
                    notch=(50.0,), skip_periods=1):
    """Process a multi-channel SSIP dataset.

    current  : (n_tx, n_samples) measured current records (A)
    voltage  : (n_data, n_samples) measured MN voltages (V)
    tx_index : (n_data,) index of the current record of each datum
    """
    freqs, bands = analysis_bands(wave, n_bands, fmin, fmax, notch)
    nd = len(voltage)
    Z = np.empty((len(freqs), nd), complex)
    ea, ep = np.empty_like(Z, float), np.empty_like(Z, float)
    P = 0
    for i in range(nd):
        z, a, p, P = transfer_function(current[tx_index[i]], voltage[i], wave, bands, skip_periods)
        Z[:, i], ea[:, i], ep[:, i] = z, a, p
    return SpectralData(freqs, Z, ea, ep, P)


def impulse_response(current, voltage, wave: PRBSWaveform, skip_periods=1):
    """System impulse response h(t) by periodic deconvolution (stacked over periods).

    For an m-sequence this equals the normalised cross-correlation of the
    received voltage with the transmitted code.
    """
    spp = wave.samples_per_period
    I = np.fft.rfft(_periods(current, spp, skip_periods), axis=1).mean(0)
    V = np.fft.rfft(_periods(voltage, spp, skip_periods), axis=1).mean(0)
    P = np.abs(I) ** 2
    H = np.where(P > 1e-6 * P.max(), V * I.conj() / np.maximum(P, 1e-30), 0)
    h = np.fft.irfft(H, n=spp) * wave.fs
    t = np.arange(spp) / wave.fs
    return t, h

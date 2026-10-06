"""Synthetic SSIP surveys: 3D spectral forward modelling and raw time series.

Raw records are generated physically: the 3D earth response is computed at a
set of log-spaced frequencies, interpolated (smooth in log-frequency) to every
harmonic of the m-sequence, and convolved with the transmitted current over
several periods.  Realistic noise is added: Gaussian sensor noise, current
noise, powerline interference with harmonics and self-potential drift.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import CubicSpline

from .forward import Simulation
from .models import ColeColeModel
from .waveform import PRBSWaveform

log = logging.getLogger("ssip3d")


def _one_freq(args):
    sim, model, f = args
    return sim.predict(model.sigma(f))


def _one_freq_worker(args):
    # one BLAS thread per worker process: avoids oversubscription (busy-waiting threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        return _one_freq(args)
    with threadpool_limits(1):
        return _one_freq(args)


def simulate_spectra(sim: Simulation, model: ColeColeModel, freqs, correction=None, n_jobs=1):
    """Complex transfer impedances (n_f, n_data) at ``freqs`` for a Cole-Cole earth."""
    freqs = np.atleast_1d(freqs)
    if n_jobs > 1 and len(freqs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            Z = list(ex.map(_one_freq_worker, [(sim, model, f) for f in freqs]))
    else:
        Z = []
        for f in freqs:
            log.info(f"  forward at {f:.4g} Hz")
            Z.append(_one_freq((sim, model, f)))
    Z = np.array(Z)
    if correction is not None:
        Z = Z * correction[None, :]
    return Z


def interpolate_spectra(f_model, Z_model, f_new):
    """Interpolate complex spectra smoothly in log-frequency (per datum).

    Uses a cubic spline of ln(s Z) vs ln f; outside the modelled range the
    end values are held constant.
    """
    s = np.sign(Z_model[0].real)
    s[s == 0] = 1
    L = np.log(s[None] * Z_model)
    lf = np.log(f_model)
    x = np.log(np.clip(f_new, f_model.min(), f_model.max()))
    cs_r = CubicSpline(lf, L.real, axis=0)
    cs_i = CubicSpline(lf, L.imag, axis=0)
    return s[None] * np.exp(cs_r(x) + 1j * cs_i(x))


def synthesize_record(Zh, wave: PRBSWaveform, n_periods, rng, noise_rel=0.0, noise_abs=1e-5,
                      current_noise=1e-3, powerline_amp=0.0, powerline_freq=50.0, drift=0.0):
    """Time series (current, voltage) for one channel.

    Zh : transfer impedance at every rfft harmonic of one period (incl. DC).
    One extra leading period is generated (transient), which processing skips.
    """
    spp = wave.samples_per_period
    i1 = wave.one_period()
    v1 = np.fft.irfft(Zh * np.fft.rfft(i1), n=spp)
    n_tot = n_periods + 1
    t = np.arange(n_tot * spp) / wave.fs
    I = np.tile(i1, n_tot)
    V = np.tile(v1, n_tot)
    rms = np.sqrt(np.mean(v1 ** 2))
    V = V + rng.normal(0, noise_rel * rms + noise_abs, V.shape)
    if powerline_amp > 0:
        for h, a in ((1, 1.0), (3, 0.3), (5, 0.1)):
            V += powerline_amp * a * np.sin(2 * np.pi * h * powerline_freq * t + rng.uniform(0, 2 * np.pi))
    if drift:
        V += drift * rng.normal() * t / t[-1] + 0.2 * drift * np.sin(2 * np.pi * t / (t[-1] * 1.7))
    I = I + rng.normal(0, current_noise * wave.amplitude, I.shape)
    return I, V


def simulate_and_process(Z_model, f_model, wave: PRBSWaveform, bands, n_periods=16, seed=0,
                         save_records=(), **noise):
    """Simulate raw SSIP records for every datum and process them.

    Records of the datum indices listed in ``save_records`` are returned for
    inspection/plotting (all records are never held in memory at once).
    """
    from .processing import SpectralData, transfer_function

    rng = np.random.default_rng(seed)
    fh = wave.harmonics()
    Zh = interpolate_spectra(np.asarray(f_model), Z_model, np.where(fh == 0, np.min(f_model), fh))
    nd = Z_model.shape[1]
    nb = len(bands)
    Z = np.empty((nb, nd), complex)
    ea, ep = np.empty((nb, nd)), np.empty((nb, nd))
    saved = {}
    P = 0
    for i in range(nd):
        I, V = synthesize_record(Zh[:, i], wave, n_periods, rng, **noise)
        Z[:, i], ea[:, i], ep[:, i], P = transfer_function(I, V, wave, bands)
        if i in save_records:
            saved[i] = (I, V)
    centers = np.array([np.exp(np.mean(np.log(fh[b]))) for b in bands])
    return SpectralData(centers, Z, ea, ep, P), saved, Zh

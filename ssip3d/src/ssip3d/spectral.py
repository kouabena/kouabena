"""Cell-by-cell spectral analysis of the multi-frequency inversion results.

Given the recovered complex conductivity sigma*(f) of every model cell, two
spectral models are fitted:

* **Debye decomposition** (linear, robust, non-negative):
  rho*(w) = rho0 - sum_l a_l * i w tau_l / (1 + i w tau_l),  a_l >= 0
  -> total chargeability m = sum a_l / rho0, mean relaxation time
     tau_mean = exp(sum m_l ln tau_l / m), normalised chargeability m / rho0.
* **Pelton Cole-Cole** (nonlinear, optional), initialised from the Debye result:
  rho0, m, tau, c.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, nnls

from .models import cole_cole_resistivity


def debye_decomposition(freqs, rho, amp_err=0.01, phase_err=1e-3, n_tau_per_decade=6, lam=1.0,
                        tau_range=None):
    """Debye decomposition of complex resistivity spectra.

    freqs : (n_f,), rho : (n_f, n_cells) complex resistivity.
    Returns dict with rho0, m, tau_mean, mn (normalised chargeability, S/m), tau grid, weights.
    """
    freqs = np.asarray(freqs, float)
    w = 2 * np.pi * freqs
    if tau_range is None:
        tau_range = (0.1 / w.max(), 10.0 / w.min())
    nt = max(int(np.ceil(np.log10(tau_range[1] / tau_range[0]) * n_tau_per_decade)), 3)
    tau = np.geomspace(*tau_range, nt)
    K = 1j * np.outer(w, tau) / (1 + 1j * np.outer(w, tau))  # (n_f, nt)
    A = np.concatenate([np.ones((len(w), 1)), -K], axis=1)  # complex (n_f, 1+nt)
    L = np.diff(np.eye(nt), 2, axis=0) if nt > 2 else np.zeros((0, nt))
    Lfull = np.concatenate([np.zeros((len(L), 1)), L], axis=1)
    rho = np.atleast_2d(rho)
    n = rho.shape[1]
    out = dict(rho0=np.zeros(n), m=np.zeros(n), tau_mean=np.zeros(n), mn=np.zeros(n),
               misfit=np.zeros(n), tau=tau, weights=np.zeros((nt, n)))
    for j in range(n):
        r = rho[:, j]
        scale = np.abs(r[0])
        rr = r / scale
        wr = 1.0 / (amp_err * np.abs(rr))
        wi = 1.0 / (phase_err * np.abs(rr))
        M = np.concatenate([wr[:, None] * A.real, wi[:, None] * A.imag, np.sqrt(lam) * Lfull])
        b = np.concatenate([wr * rr.real, wi * rr.imag, np.zeros(len(Lfull))])
        x, res = nnls(M, b, maxiter=50 * M.shape[1])
        r0 = x[0]
        a = x[1:]
        mt = a.sum() / max(r0, 1e-12)
        out["rho0"][j] = r0 * scale
        out["m"][j] = mt
        out["weights"][:, j] = a / max(r0, 1e-12)
        out["tau_mean"][j] = np.exp(np.sum(a * np.log(tau)) / a.sum()) if a.sum() > 0 else np.nan
        out["mn"][j] = mt / out["rho0"][j]
        out["misfit"][j] = res ** 2 / (2 * len(w))
    return out


def fit_cole_cole(freqs, rho, init=None, amp_err=0.01, phase_err=1e-3, c_bounds=(0.1, 1.0)):
    """Per-cell Pelton Cole-Cole fit. Returns dict rho0, m, tau, c, misfit."""
    freqs = np.asarray(freqs, float)
    rho = np.atleast_2d(rho)
    n = rho.shape[1]
    out = {k: np.full(n, np.nan) for k in ("rho0", "m", "tau", "c", "misfit")}
    if init is None:
        init = debye_decomposition(freqs, rho, amp_err, phase_err)
    for j in range(n):
        r = rho[:, j]
        lr = np.log(r * np.sign(r.real))

        def resid(p):
            r0, m, tau, c = np.exp(p[0]), 1 / (1 + np.exp(-p[1])), np.exp(p[2]), p[3]
            lm = np.log(cole_cole_resistivity(freqs, r0, m, tau, c))
            return np.r_[(lm.real - lr.real) / amp_err, (lm.imag - lr.imag) / phase_err]

        m0 = np.clip(init["m"][j], 1e-4, 0.95)
        t0 = init["tau_mean"][j] if np.isfinite(init["tau_mean"][j]) else 1 / (2 * np.pi * np.sqrt(freqs.min() * freqs.max()))
        p0 = [np.log(max(init["rho0"][j], 1e-6)), np.log(m0 / (1 - m0)), np.log(t0), 0.5]
        lb = [-np.inf, -12, np.log(1e-6), c_bounds[0]]
        ub = [np.inf, 6, np.log(1e4), c_bounds[1]]
        p0 = np.clip(p0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
        try:
            sol = least_squares(resid, p0, bounds=(lb, ub), x_scale="jac", max_nfev=200)
        except Exception:
            continue
        p = sol.x
        out["rho0"][j] = np.exp(p[0])
        out["m"][j] = 1 / (1 + np.exp(-p[1]))
        out["tau"][j] = np.exp(p[2])
        out["c"][j] = p[3]
        out["misfit"][j] = np.mean(sol.fun ** 2)
    # tau outside the resolvable band (one decade beyond the measured frequencies) is not constrained
    w = 2 * np.pi * freqs
    bad = (out["tau"] < 0.1 / w.max()) | (out["tau"] > 10 / w.min())
    for k in ("tau", "c"):
        out[k][bad] = np.nan
    return out

"""Gauss-Newton complex-resistivity inversion with decoupled amplitude/phase control.

Objective at one frequency (model m = ln sigma*, data d = ln(s Z))::

    phi = || W_re Re(d_obs - F(m)) ||^2 + || W_im Im(d_obs - F(m)) ||^2
        + beta_re * phi_m(Re m) + beta_im * phi_m(Im m)

    phi_m(x) = alpha_s || w (x - x_ref) ||^2 + sum_{x,y,z} alpha_i || w_i D_i x ||^2

Because F is holomorphic, the real-equivalent Jacobian is
[[Re J, -Im J], [Im J, Re J]], and all products are evaluated in complex
arithmetic without forming it.  The amplitude (Re) and phase (Im) parts have
their own trade-off parameters, each cooled independently until its own data
misfit reaches the discrepancy target  chi^2 = N.  This removes the classic
problem of joint complex inversions where the phase image is over- or
under-regularised relative to the amplitude image (cf. Kemna 2000's separate
phase-refinement stage), in a single pass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .forward import LogImpedanceProblem, ModelMap

log = logging.getLogger("ssip3d")


# --------------------------------------------------------------------------- regularisation
class Regularization:
    """Smallness + first-order smoothness on the active core cells."""

    def __init__(self, model_map: ModelMap, alpha_s=1e-3, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
                 cell_weights=None):
        mesh = model_map.mesh
        (i0, _), (j0, _), (k0, _) = mesh.core
        nxc, nyc, nzc = mesh.core_shape
        i, j, k = mesh.cell_ijk()
        cells = model_map.model_cells
        ci, cj, ck = i[cells] - i0, j[cells] - j0, k[cells] - k0
        lut = -np.ones((nxc, nyc, nzc), int)
        lut[ci, cj, ck] = np.arange(len(cells))
        n = len(cells)
        w = np.ones(n) if cell_weights is None else np.asarray(cell_weights, float)
        self.n = n
        blocks = [np.sqrt(alpha_s) * sp.diags(w)]
        self.smooth = []
        for ax, alpha in enumerate((alpha_x, alpha_y, alpha_z)):
            if alpha <= 0:
                continue
            a = lut
            if ax == 0:
                p, q = a[:-1], a[1:]
            elif ax == 1:
                p, q = a[:, :-1], a[:, 1:]
            else:
                p, q = a[:, :, :-1], a[:, :, 1:]
            p, q = p.ravel(), q.ravel()
            ok = (p >= 0) & (q >= 0)
            p, q = p[ok], q[ok]
            r = np.arange(len(p))
            wf = 0.5 * (w[p] + w[q])
            D = sp.csr_matrix((np.r_[-wf, wf], (np.r_[r, r], np.r_[p, q])), shape=(len(p), n))
            self.smooth.append(np.sqrt(alpha) * D)
        self.Ws = blocks[0].tocsr()
        self.Wd = sp.vstack(self.smooth).tocsr() if self.smooth else sp.csr_matrix((0, n))
        self.WtW = (self.Ws.T @ self.Ws + self.Wd.T @ self.Wd).tocsr()
        self.WsTWs = (self.Ws.T @ self.Ws).tocsr()

    def phi(self, x, x_ref):
        """x real."""
        return float(np.sum((self.Ws @ (x - x_ref)) ** 2) + np.sum((self.Wd @ x) ** 2))

    def grad(self, x, x_ref):
        return self.WsTWs @ (x - x_ref) + (self.Wd.T @ (self.Wd @ x))


# --------------------------------------------------------------------------- CG on R^{2n}
def _pcg(apply_H, b, precond, maxiter=50, rtol=1e-3):
    """Preconditioned CG for a real-linear SPD operator acting on complex vectors.

    Inner product <x, y> = Re(x^H y), i.e. the Euclidean product on R^{2n}.
    ``precond`` is a complex vector (Re part preconditions Re, Im part Im).
    """
    x = np.zeros_like(b)
    r = b.copy()
    z = r.real / precond.real + 1j * (r.imag / precond.imag)
    p = z.copy()
    rz = np.real(np.vdot(r, z))
    bn = np.sqrt(np.real(np.vdot(b, b)))
    for it in range(maxiter):
        Hp = apply_H(p)
        alpha = rz / np.real(np.vdot(p, Hp))
        x += alpha * p
        r -= alpha * Hp
        if np.sqrt(np.real(np.vdot(r, r))) < rtol * bn:
            return x, it + 1
        z = r.real / precond.real + 1j * (r.imag / precond.imag)
        rz_new = np.real(np.vdot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxiter


# --------------------------------------------------------------------------- GN driver
@dataclass
class InversionOptions:
    max_iter: int = 10
    max_iter_first: int = 15  # iterations allowed for the first (lowest) frequency
    chi_target: float = 1.0  # target misfit per datum (each of amplitude / phase)
    beta_ratio: float = 1.0  # initial beta = ratio * tr(J^T W J) / tr(R^T R)
    beta_cooling: float = 2.0
    beta_im_scale: float = 1.0  # initial beta_im = beta_im_scale * estimate
    cg_maxiter: int = 100
    cg_rtol: float = 1e-3
    max_step_halvings: int = 6
    lnsig_bounds: tuple = (np.log(1e-5), np.log(10.0))  # bounds of ln|sigma| (S/m)
    phase_bounds: tuple = (-0.2, 0.8)  # bounds of the conductivity phase (rad)
    alpha_s: float = 1e-3
    alpha_x: float = 1.0
    alpha_y: float = 1.0
    alpha_z: float = 1.0
    sensitivity_weighting: bool = True
    sensitivity_floor: float = 0.05


@dataclass
class InversionResult:
    m: np.ndarray
    d_pred: np.ndarray
    history: list = field(default_factory=list)
    converged: bool = False
    beta_re: float = 0.0
    beta_im: float = 0.0
    J: np.ndarray | None = None

    volume: np.ndarray | None = None  # mesh volume represented by each model cell

    @property
    def coverage(self):
        """Volume-normalised cumulative sensitivity per model cell (max = 1), a resolution proxy."""
        if self.J is None:
            return None
        s = np.sqrt(np.sum(np.abs(self.J) ** 2, axis=0))
        if self.volume is not None:
            s = s / self.volume
        return s / s.max()


def sensitivity_weights(J, wr, wi, floor=0.05):
    """Cell weights for the regulariser that counteract the decay of sensitivity with depth."""
    s = np.sqrt(np.sum((wr[:, None] ** 2 + wi[:, None] ** 2) * np.abs(J) ** 2, axis=0))
    s = s / s.max()
    return np.sqrt(np.maximum(s, floor))


def gauss_newton(problem: LogImpedanceProblem, d_obs, err_re, err_im, m0, m_ref=None,
                 opts: InversionOptions | None = None, reg: Regularization | None = None,
                 beta=None, first=None, label="", max_iter=None):
    """Run Gauss-Newton for one frequency. Returns :class:`InversionResult`."""
    opts = opts or InversionOptions()
    max_iter = opts.max_iter if max_iter is None else max_iter
    wr, wi = 1.0 / np.asarray(err_re), 1.0 / np.asarray(err_im)
    N = len(d_obs)
    target = opts.chi_target * N
    m = np.array(m0, dtype=complex)
    m_ref = m.copy() if m_ref is None else np.asarray(m_ref, complex)
    d, J = first if first is not None else problem.forward(m)

    if reg is None:
        cw = sensitivity_weights(J, wr, wi, opts.sensitivity_floor) if opts.sensitivity_weighting else None
        reg = Regularization(problem.map, opts.alpha_s, opts.alpha_x, opts.alpha_y, opts.alpha_z, cw)

    def misfits(dd):
        r = d_obs - dd
        return float(np.sum((wr * r.real) ** 2)), float(np.sum((wi * r.imag) ** 2))

    diagR = reg.WtW.diagonal()
    def jdiag(J):
        a, b = np.abs(J.real) ** 2, np.abs(J.imag) ** 2
        return (wr[:, None] ** 2 * a + wi[:, None] ** 2 * b).sum(0), (wr[:, None] ** 2 * b + wi[:, None] ** 2 * a).sum(0)

    if beta is None:
        dre, dim_ = jdiag(J)
        beta_re = opts.beta_ratio * dre.sum() / diagR.sum()
        beta_im = opts.beta_im_scale * opts.beta_ratio * dim_.sum() / diagR.sum()
    else:
        beta_re, beta_im = beta

    def objective(mm, dd):
        fre, fim = misfits(dd)
        return fre + fim + beta_re * reg.phi(mm.real, m_ref.real) + beta_im * reg.phi(mm.imag, m_ref.imag)

    hist = []
    converged = False
    for it in range(max_iter + 1):
        fre, fim = misfits(d)
        phim_re, phim_im = reg.phi(m.real, m_ref.real), reg.phi(m.imag, m_ref.imag)
        hist.append(dict(iter=it, chi2_amp=fre / N, chi2_phase=fim / N, beta_amp=beta_re,
                         beta_phase=beta_im, phim_amp=phim_re, phim_phase=phim_im))
        log.info(f"{label} it {it:2d}  chi2 amp {fre / N:9.3f}  phase {fim / N:9.3f}  "
                 f"beta {beta_re:.2e}/{beta_im:.2e}")
        # accept when each misfit lies in the discrepancy window [0.4, 1.1] x target;
        # a part that over-fits is also accepted once warming beta no longer changes it
        prev = hist[-2] if len(hist) > 1 else None

        def ok(phi, key):
            if phi > 1.1 * target:
                return False
            if phi >= 0.4 * target:
                return True
            return prev is not None and abs(phi / N - prev[key]) < 0.1 * prev[key]

        ok_re, ok_im = ok(fre, "chi2_amp"), ok(fim, "chi2_phase")
        if ok_re and ok_im:
            converged = True
            break
        if it == max_iter:
            break
        r = d_obs - d
        wres = wr ** 2 * r.real + 1j * (wi ** 2 * r.imag)
        g = -(J.conj().T @ wres)
        g += beta_re * reg.grad(m.real, m_ref.real) + 1j * beta_im * reg.grad(m.imag, m_ref.imag)

        def H(v):
            Jv = J @ v
            out = J.conj().T @ (wr ** 2 * Jv.real + 1j * (wi ** 2 * Jv.imag))
            return out + beta_re * (reg.WtW @ v.real) + 1j * beta_im * (reg.WtW @ v.imag)

        dre, dim_ = jdiag(J)
        pc = (dre + beta_re * diagR) + 1j * (dim_ + beta_im * diagR)
        step, ncg = _pcg(H, -g, pc, opts.cg_maxiter, opts.cg_rtol)

        f0 = objective(m, d)
        alpha = 1.0
        for _ in range(opts.max_step_halvings + 1):
            m_try = m + alpha * step
            m_try = np.clip(m_try.real, *opts.lnsig_bounds) + 1j * np.clip(m_try.imag, *opts.phase_bounds)
            d_try, J_try = problem.forward(m_try)
            if objective(m_try, d_try) < f0:
                break
            alpha *= 0.5
        else:
            log.warning(f"{label} line search failed; stopping")
            break
        m, d, J = m_try, d_try, J_try
        # independent cooling: each part is cooled only while its misfit is above target
        fre, fim = misfits(d)
        beta_re = _update_beta(beta_re, fre, target, opts.beta_cooling)
        beta_im = _update_beta(beta_im, fim, target, opts.beta_cooling)
    vol = problem.map.P.T @ problem.sim.mesh.cell_volumes
    return InversionResult(m=m, d_pred=d, history=hist, converged=converged,
                           beta_re=beta_re, beta_im=beta_im, J=J, volume=vol)


def _update_beta(beta, phi, target, cooling):
    """Cool while above target (faster when far, gently when close); warm up when over-fitting.

    Updates are damped near the target so that beta does not ping-pong
    around the discrepancy value.
    """
    if phi > 1.1 * target:
        return beta / (cooling if phi > 2 * target else np.sqrt(cooling))
    if phi < 0.4 * target:
        return beta * min(np.sqrt(target / max(phi, 1e-12)), cooling)
    return beta


# --------------------------------------------------------------------------- multi-frequency
def invert_multifrequency(problem: LogImpedanceProblem, freqs, d_obs, err_re, err_im, m_start,
                          opts: InversionOptions | None = None, freq_coupling: float = 0.0,
                          callback=None):
    """Sequential multi-frequency inversion with frequency continuation.

    Frequencies are inverted from low to high. Each frequency starts from the
    previous result; if ``freq_coupling > 0`` the previous model is also used
    as the reference model with smallness weight ``freq_coupling``, which
    enforces smooth spectral behaviour of every cell.

    d_obs, err_re, err_im : (n_freq, n_data) arrays.
    Returns list of InversionResult in the order of ``freqs``.
    """
    opts = opts or InversionOptions()
    order = np.argsort(freqs)
    results = [None] * len(freqs)
    m_prev = np.asarray(m_start, complex)
    # one regulariser (with sensitivity weights from the starting model at the
    # lowest frequency) shared by all frequencies, for consistent images
    k0 = order[0]
    first = problem.forward(m_prev)
    cw = (sensitivity_weights(first[1], 1 / err_re[k0], 1 / err_im[k0], opts.sensitivity_floor)
          if opts.sensitivity_weighting else None)
    reg = Regularization(problem.map, opts.alpha_s, opts.alpha_x, opts.alpha_y, opts.alpha_z, cw)
    reg_c = (Regularization(problem.map, freq_coupling, opts.alpha_x, opts.alpha_y, opts.alpha_z, cw)
             if freq_coupling > 0 else reg)
    prev = None
    for rank, k in enumerate(order):
        label = f"[f={freqs[k]:.4g} Hz]"
        if prev is None:
            res = gauss_newton(problem, d_obs[k], err_re[k], err_im[k], m_prev, m_ref=m_prev,
                               opts=opts, reg=reg, first=first, label=label, max_iter=opts.max_iter_first)
        else:
            res = gauss_newton(problem, d_obs[k], err_re[k], err_im[k], m_prev, m_ref=m_prev,
                               opts=opts, reg=reg_c, beta=(prev.beta_re, prev.beta_im), label=label)
        results[k] = res
        prev = res
        m_prev = res.m
        if callback:
            callback(k, freqs[k], res)
    return results

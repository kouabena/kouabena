"""Data-space Occam inversion with eigen-decomposed trade-off search.

Objective (m = ln sigma of earth cells):

    phi(m) = || Wd (d - F(m)) ||^2 + lambda || m - m_ref ||^2_{Cm^-1}

    Cm^-1 = alpha_s I + R^T R        (R: first differences in x, y, z)

Each iteration linearises F about m_k and computes the *full* new model in
data space (Siripunvaraporn et al. 2005 style):

    m(lambda) = m_ref + Cm J^T U (s + lambda)^-1 U^T d_hat,   G = J Cm J^T = U s U^T

Because G is diagonalised once per iteration, the linearised misfit for any
lambda costs O(Nd); lambda is picked automatically to hit a target misfit
(Occam phase 1) and, once the target is reached, to give the smoothest model
that still fits (Occam phase 2). Only one forward solve (+Jacobian) is
needed per iteration unless a step has to be damped.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .data import determinant_average
from .forward import MU0


class Smoothness:
    """Index-space smoothing model covariance Cm = (alpha_s I + R^T R)^-1."""

    def __init__(self, mesh, alpha_s=1e-2, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0):
        nx, ny, nz = mesh.earth_shape

        def d(n):
            return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))

        I = sp.identity
        Dx = sp.kron(I(nz), sp.kron(I(ny), d(nx)))
        Dy = sp.kron(I(nz), sp.kron(d(ny), I(nx)))
        Dz = sp.kron(d(nz), sp.kron(I(ny), I(nx)))
        self.R = sp.vstack([alpha_x * Dx, alpha_y * Dy, alpha_z * Dz]).tocsr()
        n = nx * ny * nz
        self.Wm = (alpha_s * sp.identity(n) + self.R.T @ self.R).tocsc()
        self._lu = spla.splu(self.Wm)

    def cov(self, X):
        """Apply Cm to the columns of X."""
        return self._lu.solve(np.asarray(X, float))

    def roughness(self, dm):
        return float(dm @ (self.Wm @ dm))


def bostick_start_model(data, mesh, rho_min=0.1, rho_max=1e5):
    """1D Niblett-Bostick transform of the station-averaged determinant
    impedance, used as a data-driven starting/reference model (ln sigma)."""
    rho, ph = determinant_average(data)
    w = 2 * np.pi * data.freqs
    depth = np.sqrt(rho / (w * MU0))
    phr = np.radians(np.clip(ph, 10, 80))
    rb = np.clip(rho * (np.pi / (2 * phr) - 1.0), rho_min, rho_max)
    order = np.argsort(depth)
    zc = mesh.zc[mesh.nair:]
    logr = np.interp(np.log(np.maximum(zc, 1.0)), np.log(depth[order]), np.log(rb[order]))
    # light smoothing in depth
    k = np.array([0.25, 0.5, 0.25])
    logr = np.convolve(np.pad(logr, 1, mode="edge"), k, mode="valid")
    nx, ny, nz = mesh.earth_shape
    return np.tile(-logr[None, None, :], (nx, ny, 1)).ravel(order="F")


def rms(r_w):
    return float(np.sqrt(np.mean(r_w ** 2)))


def invert(sim, data, m0=None, m_ref=None, reg=None, target_rms=1.0, max_iter=12,
           step_ratio=0.5, sigma_bounds=(1e-5, 10.0), verbose=True, callback=None):
    """Run the inversion. Returns (m, predicted ImpedanceData, history)."""
    t_start = time.time()
    d = data.vector()
    err = data.error_vector()
    Wd = np.where(np.isfinite(err) & (err > 0), 1.0 / np.where(err > 0, err, 1.0), 0.0)
    good = Wd > 0
    ndata = int(good.sum())
    reg = reg or Smoothness(sim.mesh)
    m0 = bostick_start_model(data, sim.mesh) if m0 is None else np.asarray(m0, float)
    m_ref = m0.copy() if m_ref is None else np.asarray(m_ref, float)
    lo, hi = np.log(sigma_bounds[0]), np.log(sigma_bounds[1])

    def forward(m):
        Z, J = sim.predict(m, jacobian=True)
        f = np.r_[Z.real.ravel(), Z.imag.ravel()]
        Jr = J.reshape(-1, sim.nm)
        Jr = np.vstack([Jr.real, Jr.imag])
        r = Wd * (d - f)
        return Z, f, Jr, r

    def rms_of(r):
        return float(np.sqrt(np.sum(r ** 2) / ndata))

    m = m0.copy()
    Z, f, J, r = forward(m)
    cur = rms_of(r)
    lam = None
    history = [dict(iter=0, rms=cur, lam=np.nan, rough=reg.roughness(m - m_ref), time=time.time() - t_start)]
    if verbose:
        print(f"iter  0  rms {cur:7.3f}")
    if callback:
        callback(0, m, Z, history)
    for it in range(1, max_iter + 1):
        Jw = Wd[:, None] * J
        dhat = r + Jw @ (m - m_ref)
        B = reg.cov(Jw.T)                       # Cm J^T      (nm x nd)
        G = Jw @ B
        s, U = np.linalg.eigh(0.5 * (G + G.T))
        s = np.maximum(s, 0.0)
        c = U.T @ dhat

        def lin_rms(l):
            return float(np.sqrt(np.sum((l * c / (s + l)) ** 2) / ndata))

        def model(l):
            return np.clip(m_ref + B @ (U @ (c / (s + l))), lo, hi)

        # pick lambda: linearised misfit = goal (bisection in log lambda)
        goal = max(target_rms, step_ratio * cur) if cur > target_rms else target_rms
        l_lo, l_hi = s.max() * 1e-10 + 1e-12, s.max() * 1e4 + 1.0
        if lin_rms(l_lo) > goal:
            lam_try = l_lo * 10
        else:
            for _ in range(60):
                mid = np.sqrt(l_lo * l_hi)
                if lin_rms(mid) > goal:
                    l_hi = mid
                else:
                    l_lo = mid
            lam_try = l_lo
        accepted = False
        for damp in range(5):
            m_new = model(lam_try)
            Z_n, f_n, J_n, r_n = forward(m_new)
            new = rms_of(r_n)
            if verbose:
                print(f"iter {it:2d}  lambda {lam_try:9.3e}  lin-rms {lin_rms(lam_try):7.3f}  rms {new:7.3f}")
            fits = new < cur or (cur <= target_rms and new <= 1.02 * target_rms)
            if fits:
                accepted = True
                break
            lam_try *= 10.0
        if not accepted:
            if verbose:
                print("no further improvement - stopping")
            break
        rough_old = reg.roughness(m - m_ref)
        m, Z, f, J, r, cur, lam = m_new, Z_n, f_n, J_n, r_n, new, lam_try
        history.append(dict(iter=it, rms=cur, lam=lam, rough=reg.roughness(m - m_ref),
                            time=time.time() - t_start))
        if callback:
            callback(it, m, Z, history)
        rough_new = history[-1]["rough"]
        if cur <= target_rms and abs(rough_new - rough_old) <= 0.01 * max(rough_old, 1e-12):
            if verbose:
                print("target reached and model stable - done")
            break
    if verbose:
        print(f"total time {time.time() - t_start:.0f} s")
    pred = data.copy(Z=Z)
    return m, pred, history

"""Run with:  python -m pytest tests -q   (from the amt3d-inversion folder)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amt3d.forward import MU0, Simulation  # noqa: E402
from amt3d.mesh import Mesh, padded_spacing  # noqa: E402


def small_mesh():
    h = padded_spacing(100.0, 6, 5, 1.8)
    hz = np.r_[25.0 * 1.3 ** np.arange(10), 500 * 1.8 ** np.arange(1, 5)]
    hz_air = 20.0 * 4 ** np.arange(5)
    return Mesh(h, h, hz, hz_air)


def test_halfspace_matches_analytic():
    M = small_mesh()
    st = np.array([[0.0, 0.0], [100.0, -100.0]])
    freqs = np.array([100.0, 10.0])
    sim = Simulation(M, st, freqs)
    Z, _ = sim.predict(np.full(sim.nm, np.log(0.01)))
    w = 2 * np.pi * freqs[:, None]
    rho_xy = np.abs(Z[:, :, 0, 1]) ** 2 / (w * MU0)
    ph_xy = np.degrees(np.angle(Z[:, :, 0, 1]))
    assert np.allclose(rho_xy, 100.0, rtol=0.06)
    assert np.allclose(ph_xy, 45.0, atol=2.0)
    assert np.allclose(Z[:, :, 1, 0], -Z[:, :, 0, 1], rtol=1e-6)
    assert np.abs(Z[:, :, 0, 0]).max() < 1e-8 * np.abs(Z[:, :, 0, 1]).max()


def test_jacobian_matches_finite_difference():
    M = small_mesh()
    st = np.array([[30.0, -40.0], [-120.0, 60.0]])
    sim = Simulation(M, st, [30.0])
    rng = np.random.default_rng(1)
    m = np.log(0.01) + 0.5 * rng.standard_normal(sim.nm)
    Z, J = sim.predict(m, jacobian=True)
    # The 1D boundary condition depends on the outer ring of (padding) cells;
    # that dependence is neglected in J, so the exact check excludes them.
    interior = ~sim.ops.ring[sim.earth]
    for mask, tol in [(interior, 1e-4), (np.ones(sim.nm, bool), 3e-3)]:
        dm = 0.01 * rng.standard_normal(sim.nm) * mask
        fd = (sim.predict(m + dm)[0] - sim.predict(m - dm)[0]) / 2
        err = np.abs(fd - J @ dm).max() / np.abs(fd).max()
        assert err < tol, err

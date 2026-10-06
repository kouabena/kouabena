"""Synthetic test: conductive sulphide halo + resistive core under a 5x5 AMT grid.

    python examples/synthetic_porphyry.py          (writes results to ./results)
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import amt3d  # noqa: E402
from amt3d import plot  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out, exist_ok=True)

# ---- survey ---------------------------------------------------------------
sx, sy = np.meshgrid(np.arange(-200, 201, 100.0), np.arange(-200, 201, 100.0), indexing="ij")
stations = np.c_[sx.ravel(), sy.ravel()]
freqs = np.array([2000.0, 600.0, 200.0, 60.0, 20.0, 6.0])

mesh = amt3d.Mesh.for_stations(stations[:, 0], stations[:, 1], cell=50, n_margin=3, n_pad=5,
                               pad_factor=1.8, z_first=20, z_factor=1.3, z_core=1500, z_pad=4,
                               n_air=6, air_factor=4, max_period=1 / freqs.min(), rho_bg=100)
print(mesh)
sim = amt3d.Simulation(mesh, stations, freqs)
print(f"{sim.nm} model cells, solver: {'MKL Pardiso' if amt3d.HAVE_PARDISO else 'SuperLU'}")

# ---- true model -----------------------------------------------------------
rho = np.full(mesh.earth_shape, 100.0)
X, Y, Z = np.meshgrid(mesh.xc, mesh.yc, mesh.zc[mesh.nair:], indexing="ij")
halo = (np.abs(X + 50) < 125) & (np.abs(Y + 75) < 125) & (Z > 100) & (Z < 400)
core = (np.abs(X - 75) < 100) & (np.abs(Y - 150) < 75) & (Z > 150) & (Z < 600)
rho[halo] = 5.0
rho[core] = 1000.0
m_true = -np.log(rho).ravel(order="F")

t = time.time()
Z_true, _ = sim.predict(m_true)
print(f"forward modelling: {time.time() - t:.1f} s")
obs = amt3d.ImpedanceData(freqs, stations, Z_true).add_noise(0.03, seed=1)
obs.err = np.zeros(obs.Z.shape)
obs.set_error_floor(0.03)

# round-trip through the ModEM format to exercise the reader
obs.write_modem(os.path.join(out, "synthetic_data.dat"))
obs = amt3d.ImpedanceData.read_modem(os.path.join(out, "synthetic_data.dat"))

# ---- inversion --------------------------------------------------------------
reg = amt3d.Smoothness(mesh, alpha_s=1e-2, alpha_x=1, alpha_y=1, alpha_z=1)
m0 = amt3d.bostick_start_model(obs, mesh)
m_inv, pred, hist = amt3d.invert(sim, obs, m0=m0, reg=reg, target_rms=1.0, max_iter=10)
np.save(os.path.join(out, "model_inverted.npy"), m_inv)
plot.export_vtk(mesh, m_inv, os.path.join(out, "model_inverted.vtk"))
plot.export_vtk(mesh, m_true, os.path.join(out, "model_true.vtk"))

# ---- figures ----------------------------------------------------------------
ref = 2.0  # log10(100 Ohm.m): colour-scale midpoint
cs = dict(ref=ref, vmin=0.7, vmax=3.0)  # same colour range for every model plot
figs = {
    "sections_x.png": plot.plot_sections(mesh, [m_true, m0, m_inv], stations, along="x", at=-100,
                                         zmax=1000, titles=["True", "Start (Bostick)", "Inverted"], **cs),
    "sections_y.png": plot.plot_sections(mesh, [m_true, m_inv], stations, along="y", at=0,
                                         zmax=1000, titles=["True", "Inverted"], **cs),
    "depth_slices.png": plot.plot_depth_slices(mesh, m_inv, [50, 150, 250, 350, 500, 700], stations,
                                               title="Inverted model", **cs),
    "depth_slices_true.png": plot.plot_depth_slices(mesh, m_true, [50, 150, 250, 350, 500, 700],
                                                    stations, title="True model", **cs),
    "soundings.png": plot.plot_soundings(obs, pred, stations=[0, 6, 12, 18, 24]),
    "phase_tensors.png": plot.plot_phase_tensors(obs, freq_index=3),
    "convergence.png": plot.plot_convergence(hist),
    "station_rms.png": plot.plot_rms_map(obs, pred),
    "conductors_3d.png": plot.plot_3d_conductors(mesh, m_inv, stations, threshold_ohmm=30, zmax=800, **cs),
}
for name, fig in figs.items():
    fig.savefig(os.path.join(out, name), dpi=130, bbox_inches="tight")
print("figures written to", os.path.abspath(out))

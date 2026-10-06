"""Invert a ModEM Full_Impedance data file.

    python examples/invert_modem_file.py my_data.dat --cell 50 --floor 0.05 --iters 10
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import amt3d  # noqa: E402
from amt3d import plot  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("datafile")
p.add_argument("--cell", type=float, default=50.0, help="core cell size (m)")
p.add_argument("--floor", type=float, default=0.05, help="error floor, fraction of sqrt|Zxy Zyx|")
p.add_argument("--diag-floor", type=float, default=None, help="separate floor for Zxx, Zyy")
p.add_argument("--iters", type=int, default=10)
p.add_argument("--target", type=float, default=1.0)
p.add_argument("--zmax", type=float, default=1500.0, help="depth of fine core mesh (m)")
p.add_argument("--out", default="results_field")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

obs = amt3d.ImpedanceData.read_modem(a.datafile).set_error_floor(a.floor, a.diag_floor)
rho, _ = amt3d.determinant_average(obs)
st = obs.stations
mesh = amt3d.Mesh.for_stations(st[:, 0], st[:, 1], cell=a.cell, n_margin=3, n_pad=5, pad_factor=1.8,
                               z_first=a.cell / 2.5, z_factor=1.3, z_core=a.zmax, z_pad=4, n_air=6,
                               air_factor=4, max_period=obs.periods.max(), rho_bg=float(np.median(rho)))
print(mesh, f"\n{len(st)} stations, {len(obs.freqs)} frequencies")
sim = amt3d.Simulation(mesh, st, obs.freqs)
m, pred, hist = amt3d.invert(sim, obs, target_rms=a.target, max_iter=a.iters)

np.save(os.path.join(a.out, "model.npy"), m)
plot.export_vtk(mesh, m, os.path.join(a.out, "model.vtk"))
pred.write_modem(os.path.join(a.out, "predicted.dat"))
zs = [z for z in (25, 50, 100, 200, 300, 500, 750, 1000, 1500) if z < a.zmax]
plot.plot_depth_slices(mesh, m, zs, st).savefig(os.path.join(a.out, "depth_slices.png"), dpi=130)
plot.plot_sections(mesh, m, st, along="x", at=float(np.median(st[:, 1])), zmax=a.zmax).savefig(
    os.path.join(a.out, "section_x.png"), dpi=130)
plot.plot_sections(mesh, m, st, along="y", at=float(np.median(st[:, 0])), zmax=a.zmax).savefig(
    os.path.join(a.out, "section_y.png"), dpi=130)
plot.plot_soundings(obs, pred, stations=list(range(min(6, len(st))))).savefig(
    os.path.join(a.out, "soundings.png"), dpi=130)
plot.plot_convergence(hist, a.target).savefig(os.path.join(a.out, "convergence.png"), dpi=130)
plot.plot_rms_map(obs, pred).savefig(os.path.join(a.out, "station_rms.png"), dpi=130)
print("results in", os.path.abspath(a.out))

"""Step 4: export the 3D models (VTK, CSV) and make the thesis figures.

Outputs in results/:
  ssip3d_model.vtk        all cell properties, open in ParaView
  ssip3d_model_cells.csv  x y z depth resistivity phase_F1..F4 coverage
  fig_depth_slices_*.png  plan views at fixed depths below surface
  fig_sections_*.png      vertical sections along the three lines
  fig_phase_spectrum.png  phase sections of the middle line at F1..F4
  fig_data_fit.png        observed vs. predicted data, misfit histograms
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import pygimli as pg
from pygimli.physics import ert

import config as C
from meshing import topography

CMAP_RES = "viridis"      # log10 resistivity
CMAP_PHASE = "inferno"    # phase (mrad)
DEPTHS = [25, 75, 150, 250, 400]            # m below surface
X_LIM = (-100, 2500)                         # plotted x range (m)
Y_LIM = (-100, 500)


def load():
    R = C.RESULTS_DIR
    mesh = pg.load(str(R / "paraDomain.bms"))
    m = dict(res=np.load(R / "res.npy"), coverage=np.load(R / "coverage.npy"))
    for fk in C.FREQS:
        f = R / f"phase_{fk}.npy"
        if f.exists():
            m[f"phase_{fk}"] = np.load(f)
    return mesh, m


def coverage_mask(cov):
    """Cells considered resolved: coverage above a data-driven threshold."""
    thr = np.percentile(cov[np.isfinite(cov)], 25)
    return np.isfinite(cov) & (cov >= thr), thr


def interpolators(centers, values):
    lin = LinearNDInterpolator(centers, values)
    near = NearestNDInterpolator(centers, values)

    def f(p):
        v = lin(p)
        bad = ~np.isfinite(v)
        v[bad] = near(p[bad])
        return v
    return f


def main():
    mesh, m = load()
    data = ert.load(str(C.WORK_DIR / "ssip3d.dat"))
    sensors = np.array(data.sensors())
    topo = topography(sensors)
    cc = np.array(mesh.cellCenters())
    depth = topo(cc[:, 0], cc[:, 1]) - cc[:, 2]
    ok, thr = coverage_mask(m["coverage"])
    print(f"coverage threshold (log10) {thr:.2f}: {ok.mean()*100:.0f}% cells kept")

    # ---------------- VTK + CSV ----------------------------------------
    out = pg.Mesh(mesh)
    out["resistivity"] = m["res"]
    out["log10_resistivity"] = np.log10(m["res"])
    out["coverage_log10"] = m["coverage"]
    out["resolved"] = ok.astype(float)
    out["depth"] = depth
    cols = ["x", "y", "z", "depth", "resistivity"]
    table = [cc[:, 0], cc[:, 1], cc[:, 2], depth, m["res"]]
    for fk in C.FREQS:
        if f"phase_{fk}" in m:
            out[f"phase_{fk}_mrad"] = m[f"phase_{fk}"]
            cols.append(f"phase_{fk}_mrad")
            table.append(m[f"phase_{fk}"])
    cols += ["coverage_log10", "resolved"]
    table += [m["coverage"], ok.astype(int)]
    out.exportVTK(str(C.RESULTS_DIR / "ssip3d_model.vtk"))
    np.savetxt(C.RESULTS_DIR / "ssip3d_model_cells.csv", np.c_[tuple(table)],
               delimiter=",", header=",".join(cols), comments="", fmt="%.4f")

    props = [("res", "Resistivity (Ω·m)", CMAP_RES, True)]
    if "phase_F1" in m:
        props.append(("phase_F1", f"Phase F1 {C.FREQS['F1']:.3f} Hz (mrad)",
                      CMAP_PHASE, False))

    # common colour limits from resolved cells
    def limits(key, log):
        v = m[key][ok]
        lo, hi = np.percentile(v, [2, 98])
        return (lo, hi)

    # ---------------- depth slices -------------------------------------
    xg = np.arange(X_LIM[0], X_LIM[1] + 1, 10.0)
    yg = np.arange(Y_LIM[0], Y_LIM[1] + 1, 10.0)
    X, Y = np.meshgrid(xg, yg)
    Zs = topo(X.ravel(), Y.ravel()).reshape(X.shape)
    cov_f = interpolators(cc, m["coverage"])
    for key, label, cmap, log in props:
        f = interpolators(cc, np.log10(m[key]) if log else m[key])
        lo, hi = limits(key, log)
        fig, axs = plt.subplots(len(DEPTHS), 1, figsize=(10, 2.1 * len(DEPTHS)),
                                sharex=True, constrained_layout=True)
        for ax, d in zip(axs, DEPTHS):
            P = np.c_[X.ravel(), Y.ravel(), (Zs - d).ravel()]
            V = f(P).reshape(X.shape)
            V = 10 ** V if log else V
            V = np.ma.masked_where(cov_f(P).reshape(X.shape) < thr, V)
            im = ax.pcolormesh(X, Y, V, cmap=cmap, shading="auto",
                               norm=LogNorm(lo, hi) if log else None,
                               vmin=None if log else lo,
                               vmax=None if log else hi)
            ax.plot(sensors[:, 0], sensors[:, 1], ".", color="0.2", ms=1.5)
            for name, (_, y) in C.LINES.items():
                ax.text(X_LIM[0] + 15, y + 12, name, fontsize=7, color="0.15")
            ax.set_aspect("equal")
            ax.set_ylabel("y (m)")
            ax.set_title(f"{d} m below surface", fontsize=9, loc="left")
            ax.set_xlim(*X_LIM)
            ax.set_ylim(*Y_LIM)
        axs[-1].set_xlabel("x, chainage along lines (m)")
        fig.colorbar(im, ax=axs, shrink=0.6, label=label)
        fig.savefig(C.RESULTS_DIR / f"fig_depth_slices_{key}.png", dpi=200)
        plt.close(fig)

    # ---------------- vertical sections --------------------------------
    zg = np.arange(-450, 220, 5.0)
    XS, ZS = np.meshgrid(xg, zg)

    def section(f, y, log):
        P = np.c_[XS.ravel(), np.full(XS.size, y), ZS.ravel()]
        V = f(P).reshape(XS.shape)
        V = 10 ** V if log else V
        above = ZS > topo(XS.ravel(), np.full(XS.size, y)).reshape(XS.shape)
        low = cov_f(P).reshape(XS.shape) < thr
        return np.ma.masked_where(above | low, V)

    ys = [y for _, y in C.LINES.values()]
    y_all = sorted(set(ys + [(a + b) / 2 for a, b in zip(ys[:-1], ys[1:])]))
    for key, label, cmap, log in props:
        f = interpolators(cc, np.log10(m[key]) if log else m[key])
        lo, hi = limits(key, log)
        fig, axs = plt.subplots(len(y_all), 1, figsize=(10, 2.3 * len(y_all)),
                                sharex=True, constrained_layout=True)
        for ax, y in zip(axs, y_all):
            im = ax.pcolormesh(XS, ZS, section(f, y, log), cmap=cmap,
                               shading="auto",
                               norm=LogNorm(lo, hi) if log else None,
                               vmin=None if log else lo,
                               vmax=None if log else hi)
            ax.plot(xg, topo(xg, np.full_like(xg, y)), color="0.2", lw=0.8)
            name = [n for n, (_, yy) in C.LINES.items() if yy == y]
            ax.set_title((name[0] if name else "between lines") +
                         f"  (y = {y:.0f} m)", fontsize=9, loc="left")
            ax.set_ylabel("Elevation (m)")
            ax.set_aspect("equal")
            ax.set_xlim(*X_LIM)
        axs[-1].set_xlabel("x, chainage (m)")
        fig.colorbar(im, ax=axs, shrink=0.6, label=label)
        fig.savefig(C.RESULTS_DIR / f"fig_sections_{key}.png", dpi=200)
        plt.close(fig)

    # ---------------- phase at all frequencies, middle line --------------
    fks = [fk for fk in C.FREQS if f"phase_{fk}" in m]
    if fks:
        y_mid = ys[len(ys) // 2]
        lo = min(limits(f"phase_{fk}", False)[0] for fk in fks)
        hi = max(limits(f"phase_{fk}", False)[1] for fk in fks)
        fig, axs = plt.subplots(len(fks), 1, figsize=(10, 2.3 * len(fks)),
                                sharex=True, constrained_layout=True)
        for ax, fk in zip(np.atleast_1d(axs), fks):
            f = interpolators(cc, m[f"phase_{fk}"])
            im = ax.pcolormesh(XS, ZS, section(f, y_mid, False), cmap=CMAP_PHASE,
                               shading="auto", vmin=lo, vmax=hi)
            ax.plot(xg, topo(xg, np.full_like(xg, y_mid)), color="0.2", lw=0.8)
            ax.set_title(f"{fk} = {C.FREQS[fk]:.3f} Hz", fontsize=9, loc="left")
            ax.set_ylabel("Elevation (m)")
            ax.set_aspect("equal")
            ax.set_xlim(*X_LIM)
        np.atleast_1d(axs)[-1].set_xlabel("x, chainage (m)")
        fig.colorbar(im, ax=axs, shrink=0.6, label="Phase (mrad)")
        fig.suptitle(f"Middle line (y = {y_mid:.0f} m): phase vs. frequency",
                     fontsize=10)
        fig.savefig(C.RESULTS_DIR / "fig_phase_spectrum.png", dpi=200)
        plt.close(fig)

    # ---------------- data fit -----------------------------------------
    R = C.RESULTS_DIR
    obs, pre = np.load(R / "rhoa_data.npy"), np.load(R / "rhoa_response.npy")
    ncol = 1 + len(fks)
    fig, axs = plt.subplots(2, ncol, figsize=(3.4 * ncol, 6.4),
                            constrained_layout=True)
    axs = np.array(axs).reshape(2, ncol)
    ax = axs[0, 0]
    ax.loglog(obs, pre, ".", ms=2, alpha=0.4, color="#2a6f97")
    lim = [min(obs.min(), pre.min()), max(obs.max(), pre.max())]
    ax.plot(lim, lim, color="0.3", lw=1)
    ax.set_xlabel("observed ρa (Ω·m)")
    ax.set_ylabel("predicted ρa (Ω·m)")
    ax.set_title("Resistivity", fontsize=9, loc="left")
    err = np.array(data["err"])
    axs[1, 0].hist(np.log(pre / obs) / err, bins=80, range=(-6, 6),
                   color="#2a6f97")
    axs[1, 0].set_xlabel("normalised misfit")
    for j, fk in enumerate(fks, start=1):
        phi = np.array(data[f"ip{fk}"])
        used = np.load(R / f"phase_used_{fk}.npy")
        resp = np.load(R / f"phase_response_{fk}.npy")
        ax = axs[0, j]
        ax.plot(phi[used], resp[used], ".", ms=2, alpha=0.4, color="#2a6f97")
        lim = np.percentile(np.r_[phi[used], resp[used]], [0.5, 99.5])
        ax.plot(lim, lim, color="0.3", lw=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("observed phase (mrad)")
        ax.set_ylabel("predicted phase (mrad)")
        ax.set_title(f"Phase {fk}", fontsize=9, loc="left")
        axs[1, j].hist(resp[used] - phi[used], bins=80, range=(-30, 30),
                       color="#2a6f97")
        axs[1, j].set_xlabel("predicted − observed (mrad)")
    fig.savefig(R / "fig_data_fit.png", dpi=200)
    plt.close(fig)
    print("written to", R)


if __name__ == "__main__":
    main()

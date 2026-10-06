"""Visualisation: model slices/sections, 3D view, data fit, phase tensors,
convergence, plus VTK export for ParaView."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Ellipse

from .data import phase_tensor

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
XY_COLOR, YX_COLOR = "#2a78d6", "#eb6834"      # categorical slots 1, 2

# Diverging: conductive (red) <- neutral grey -> resistive (blue)
RES_CMAP = LinearSegmentedColormap.from_list(
    "amt_res", ["#7a1f1e", "#e34948", "#f0efec", "#2a78d6", "#104281"])
BETA_CMAP = LinearSegmentedColormap.from_list(
    "amt_beta", ["#e34948", "#f0efec", "#2a78d6"])


def _style(ax):
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _log_rho(mesh, m):
    return mesh.to_grid(-m / np.log(10))   # log10(resistivity)


def _norm(logr, ref=None):
    lo, hi = np.percentile(logr, [1, 99])
    ref = np.median(logr) if ref is None else ref
    lo, hi = min(lo, ref - 0.3), max(hi, ref + 0.3)
    return TwoSlopeNorm(vcenter=ref, vmin=lo, vmax=hi)


def _core(mesh, stations, margin):
    sx, sy = stations[:, 0], stations[:, 1]
    return (sx.min() - margin, sx.max() + margin, sy.min() - margin, sy.max() + margin)


def plot_depth_slices(mesh, m, depths, stations, margin=200.0, ref=None, title="", ncols=3):
    """Plan-view slices of log10 resistivity at the given depths (m)."""
    logr = _log_rho(mesh, m)
    norm = _norm(logr, ref)
    zc = mesh.zc[mesh.nair:]
    x0, x1, y0, y1 = _core(mesh, stations, margin)
    nrows = int(np.ceil(len(depths) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False,
                            constrained_layout=True)
    for ax, z in zip(axs.flat, depths):
        k = int(np.argmin(np.abs(zc - z)))
        pc = ax.pcolormesh(mesh.yn, mesh.xn, logr[:, :, k], cmap=RES_CMAP, norm=norm, shading="flat")
        ax.plot(stations[:, 1], stations[:, 0], "v", ms=4, mfc="white", mec=INK, mew=0.6)
        ax.set_xlim(y0, y1); ax.set_ylim(x0, x1); ax.set_aspect("equal")
        ax.set_title(f"depth {zc[k]:.0f} m", fontsize=9)
        ax.set_xlabel("East y (m)"); ax.set_ylabel("North x (m)")
        _style(ax)
    for ax in list(axs.flat)[len(depths):]:
        ax.axis("off")
    cb = fig.colorbar(pc, ax=axs, shrink=0.8, label="log$_{10}$ resistivity (Ω·m)")
    cb.ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, color=INK)
    return fig


def plot_sections(mesh, models, stations, along="x", at=0.0, zmax=1500.0, margin=200.0,
                  titles=None, ref=None):
    """Vertical sections through one or several models (e.g. true vs inverted)."""
    models = models if isinstance(models, (list, tuple)) else [models]
    grids = [_log_rho(mesh, m) for m in models]
    norm = _norm(grids[0], ref)
    zn = mesh.zn[mesh.nair:]
    x0, x1, y0, y1 = _core(mesh, stations, margin)
    fig, axs = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6), squeeze=False,
                            constrained_layout=True)
    for n, (ax, g) in enumerate(zip(axs.flat, grids)):
        if along == "x":          # section along North at y = at
            j = int(np.argmin(np.abs(mesh.yc - at)))
            pc = ax.pcolormesh(mesh.xn, zn, g[:, j, :].T, cmap=RES_CMAP, norm=norm)
            ax.set_xlim(x0, x1); ax.set_xlabel("North x (m)")
            on = np.abs(stations[:, 1] - mesh.yc[j]) < 1e-6 + 0.5 * mesh.hy[j]
            ax.plot(stations[on, 0], np.zeros(on.sum()), "v", ms=6, mfc="white", mec=INK, clip_on=False)
        else:
            i = int(np.argmin(np.abs(mesh.xc - at)))
            pc = ax.pcolormesh(mesh.yn, zn, g[i, :, :].T, cmap=RES_CMAP, norm=norm)
            ax.set_xlim(y0, y1); ax.set_xlabel("East y (m)")
            on = np.abs(stations[:, 0] - mesh.xc[i]) < 1e-6 + 0.5 * mesh.hx[i]
            ax.plot(stations[on, 1], np.zeros(on.sum()), "v", ms=6, mfc="white", mec=INK, clip_on=False)
        ax.set_ylim(zmax, 0); ax.set_ylabel("Depth (m)")
        if titles:
            ax.set_title(titles[n], fontsize=10)
        _style(ax)
    fig.colorbar(pc, ax=axs, shrink=0.9, label="log$_{10}$ resistivity (Ω·m)")
    return fig


def plot_3d_conductors(mesh, m, stations, threshold_ohmm, zmax=1500.0, margin=200.0, ax=None):
    """3D voxel view of all core cells less resistive than `threshold_ohmm`."""
    logr = _log_rho(mesh, m)
    x0, x1, y0, y1 = _core(mesh, stations, margin)
    ix = np.flatnonzero((mesh.xc > x0) & (mesh.xc < x1))
    iy = np.flatnonzero((mesh.yc > y0) & (mesh.yc < y1))
    zc = mesh.zc[mesh.nair:]
    iz = np.flatnonzero(zc < zmax)
    sub = logr[np.ix_(ix, iy, iz)]
    filled = sub < np.log10(threshold_ohmm)
    X, Y, Z = np.meshgrid(mesh.xn[ix[0]:ix[-1] + 2], mesh.yn[iy[0]:iy[-1] + 2],
                          mesh.zn[mesh.nair:][iz[0]:iz[-1] + 2], indexing="ij")
    norm = _norm(logr)
    colors = RES_CMAP(norm(sub))
    colors[..., 3] = 0.85
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
    ax.voxels(X, Y, -Z, filled, facecolors=colors, edgecolor=(1, 1, 1, 0.15), linewidth=0.2)
    ax.scatter(stations[:, 0], stations[:, 1], np.zeros(len(stations)), marker="v", c=INK, s=12)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.set_zlabel("Elevation (m)", fontsize=8)
    ax.set_title(f"Cells < {threshold_ohmm:g} Ω·m", fontsize=10, color=INK)
    ax.tick_params(labelsize=7)
    return ax.figure


def plot_soundings(obs, pred=None, stations=(0,), ncols=3):
    """Apparent resistivity and phase (xy, yx) with error bars; prediction as lines."""
    rho_o, ph_o = obs.rho_a(), obs.phase()
    rho_e, ph_e = obs.rho_a_err(), obs.phase_err()
    T = obs.periods
    ncols = min(ncols, len(stations))
    nrows = int(np.ceil(len(stations) / ncols))
    fig, axs = plt.subplots(2 * nrows, ncols, figsize=(3.6 * ncols, 4.6 * nrows), squeeze=False,
                            sharex=True, gridspec_kw=dict(height_ratios=[2, 1] * nrows), constrained_layout=True)
    for n, s in enumerate(stations):
        ar, ap = axs[2 * (n // ncols), n % ncols], axs[2 * (n // ncols) + 1, n % ncols]
        for (a, b), col, lab in [((0, 1), XY_COLOR, "xy"), ((1, 0), YX_COLOR, "yx")]:
            ar.errorbar(T, rho_o[:, s, a, b], yerr=rho_e[:, s, a, b], fmt="o", ms=5, color=col,
                        mfc="white", mew=1.5, elinewidth=1, capsize=0, label=f"{lab} observed")
            ap.errorbar(T, np.mod(ph_o[:, s, a, b], 180) if a == 1 else ph_o[:, s, a, b],
                        yerr=ph_e[:, s, a, b], fmt="o", ms=5, color=col, mfc="white", mew=1.5,
                        elinewidth=1, capsize=0)
            if pred is not None:
                pr, pp = pred.rho_a(), pred.phase()
                ar.plot(T, pr[:, s, a, b], "-", lw=2, color=col, label=f"{lab} predicted")
                ap.plot(T, np.mod(pp[:, s, a, b], 180) if a == 1 else pp[:, s, a, b], "-", lw=2, color=col)
        ar.set_xscale("log"); ar.set_yscale("log"); ap.set_xscale("log")
        ar.set_title(obs.names[s], fontsize=10)
        ar.set_ylabel("ρa (Ω·m)"); ap.set_ylabel("Phase (°)"); ap.set_xlabel("Period (s)")
        ap.set_ylim(0, 90)
        for ax in (ar, ap):
            ax.grid(True, color=GRID, lw=0.6); _style(ax)
        if n == 0:
            ar.legend(fontsize=7, frameon=False)
    for k in range(len(stations), nrows * ncols):
        axs[2 * (k // ncols), k % ncols].axis("off"); axs[2 * (k // ncols) + 1, k % ncols].axis("off")
    return fig


def plot_phase_tensors(data, freq_index=0, ax=None, scale=None):
    """Phase-tensor ellipses (Caldwell et al., 2004), coloured by skew beta.
    |beta| > 3 deg suggests 3D structure."""
    pmax, pmin, alpha, beta = phase_tensor(data.Z[freq_index])
    st = data.stations
    if scale is None:
        d = np.diff(np.unique(st[:, 0]))
        scale = 0.8 * (d.min() if len(d) else 100.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4))
    norm = Normalize(-6, 6)
    for s in range(len(st)):
        w = scale
        h = scale * max(pmin[s], 0.5) / max(pmax[s], 1e-3)
        # x = North (plot vertical), y = East (plot horizontal); angle from North
        ang = 90.0 - (alpha[s] - beta[s])
        e = Ellipse((st[s, 1], st[s, 0]), w, h, angle=ang, facecolor=BETA_CMAP(norm(beta[s])),
                    edgecolor=INK2, lw=0.6)
        ax.add_patch(e)
    ax.set_xlim(st[:, 1].min() - scale, st[:, 1].max() + scale)
    ax.set_ylim(st[:, 0].min() - scale, st[:, 0].max() + scale)
    ax.set_aspect("equal")
    ax.set_title(f"Phase tensors, {data.freqs[freq_index]:g} Hz", fontsize=10)
    ax.set_xlabel("East y (m)"); ax.set_ylabel("North x (m)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=BETA_CMAP)
    ax.figure.colorbar(sm, ax=ax, label="skew β (°)", shrink=0.8)
    _style(ax)
    return ax.figure


def plot_convergence(history, target=1.0):
    it = [h["iter"] for h in history]
    r = [h["rms"] for h in history]
    fig, ax = plt.subplots(figsize=(4.4, 3), constrained_layout=True)
    ax.plot(it, r, "-o", color=XY_COLOR, lw=2, ms=6, mfc="white", mew=1.5)
    ax.axhline(target, color=INK2, lw=1, ls="--")
    ax.text(it[-1], target, " target", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Normalised RMS")
    ax.set_title("Data misfit", fontsize=10)
    ax.set_ylim(0, max(r) * 1.1)
    ax.grid(True, color=GRID, lw=0.6); _style(ax)
    return fig


def plot_rms_map(obs, pred, ax=None):
    """Per-station normalised RMS (all frequencies and components)."""
    r = (obs.Z - pred.Z) / obs.err
    w = np.isfinite(obs.err)
    rr = np.where(w, np.abs(r) ** 2 / 2, 0).sum(axis=(0, 2, 3)) / w.sum(axis=(0, 2, 3))
    st = obs.stations
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4))
    sc = ax.scatter(st[:, 1], st[:, 0], c=np.sqrt(rr), s=90, cmap=LinearSegmentedColormap.from_list(
        "seq", ["#cde2fb", "#2a78d6", "#0d366b"]), edgecolor="white", linewidth=1.5)
    ax.set_aspect("equal"); ax.set_xlabel("East y (m)"); ax.set_ylabel("North x (m)")
    ax.set_title("Station RMS", fontsize=10)
    ax.figure.colorbar(sc, ax=ax, label="RMS", shrink=0.8)
    _style(ax)
    return ax.figure


def export_vtk(mesh, m, path):
    """Write the model as a legacy VTK rectilinear grid (open in ParaView).
    z is written as elevation (positive up)."""
    nx, ny, nz = mesh.earth_shape
    zn = -mesh.zn[mesh.nair:]
    logr = -m / np.log(10)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\namt3d model\nASCII\nDATASET RECTILINEAR_GRID\n")
        fh.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n")
        for name, arr in (("X", mesh.xn), ("Y", mesh.yn), ("Z", zn)):
            fh.write(f"{name}_COORDINATES {len(arr)} double\n" + " ".join(f"{v:.3f}" for v in arr) + "\n")
        fh.write(f"CELL_DATA {nx * ny * nz}\nSCALARS log10_resistivity double 1\nLOOKUP_TABLE default\n")
        fh.write("\n".join(f"{v:.5f}" for v in logr) + "\n")

"""Publication-ready figures for the SSIP workflow."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from .io import core_grid  # noqa: E402
from .processing import impulse_response  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 9})


def plot_survey(survey, mesh, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    (i0, i1), (j0, j1), _ = mesh.core
    ax.add_patch(plt.Rectangle((mesh.nodes_x[i0], mesh.nodes_y[j0]), mesh.nodes_x[i1] - mesh.nodes_x[i0],
                               mesh.nodes_y[j1] - mesh.nodes_y[j0], fill=False, ls="--", color="0.5",
                               label="inversion domain"))
    e = survey.electrodes
    ax.plot(e[:, 0], e[:, 1], "k.", ms=4, label="electrodes")
    mp = survey.midpoints()
    ax.scatter(mp[:, 0], mp[:, 1], c=survey.pseudo_depth(), s=6, cmap="viridis", label="data midpoints")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Survey: {survey.n_electrodes} electrodes, {survey.n_data} configurations")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeseries(I, V, wave, path, title=""):
    fig, axs = plt.subplots(3, 1, figsize=(7, 7))
    n = min(len(I), 2 * wave.samples_per_period)
    t = np.arange(n) / wave.fs
    axs[0].plot(t, I[:n], lw=0.6, color="C0")
    ax2 = axs[0].twinx()
    ax2.plot(t, V[:n] * 1e3, lw=0.6, color="C3")
    axs[0].set_xlabel("time (s)")
    axs[0].set_ylabel("current (A)", color="C0")
    ax2.set_ylabel("voltage (mV)", color="C3")
    axs[0].set_title(f"Raw records (first two periods) {title}")
    f = wave.harmonics()[1:]
    spp = wave.samples_per_period
    Ii = np.abs(np.fft.rfft(I[spp:2 * spp]))[1:]
    Vv = np.abs(np.fft.rfft(V[spp:2 * spp]))[1:]
    axs[1].loglog(f, Ii / Ii.max(), lw=0.6, label="current")
    axs[1].loglog(f, Vv / Vv.max(), lw=0.6, label="voltage", alpha=0.7)
    axs[1].axvline(wave.chip_rate, color="0.5", ls=":")
    axs[1].set_xlabel("frequency (Hz)")
    axs[1].set_ylabel("normalised amplitude")
    axs[1].legend(fontsize=7)
    axs[1].set_title("Line spectra: the m-sequence excites all harmonics at once")
    tt, h = impulse_response(I, V, wave)
    k = int(0.5 * len(tt))
    step = np.cumsum(h) / wave.fs  # step response (ohm)
    axs[2].semilogx(tt[1:k] * 1e3, step[1:k] / step[k], lw=1)
    axs[2].set_xlabel("time after current switch-on (ms)")
    axs[2].set_ylabel("normalised step response")
    axs[2].set_title("Step response from m-sequence deconvolution (slow rise = IP charging)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectra_fit(freqs, spec, Zpred, idx, path, Ztrue=None):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for c, i in enumerate(idx):
        s = np.sign(spec.Z[0, i].real)
        col = f"C{c % 10}"
        amp = np.abs(spec.Z[:, i])
        ph = np.angle(s * spec.Z[:, i]) * 1e3
        axs[0].errorbar(freqs, amp, yerr=amp * spec.err_amp[:, i], fmt="o", ms=3, color=col, label=f"#{i}")
        axs[1].errorbar(freqs, -ph, yerr=spec.err_phase[:, i] * 1e3, fmt="o", ms=3, color=col)
        if Zpred is not None:
            axs[0].plot(freqs, np.abs(Zpred[:, i]), "-", color=col, lw=1)
            axs[1].plot(freqs, -np.angle(s * Zpred[:, i]) * 1e3, "-", color=col, lw=1)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_xscale("log")
    axs[0].set_ylabel("|Z| (ohm)")
    axs[1].set_ylabel("-phase (mrad)")
    for a in axs:
        a.set_xlabel("frequency (Hz)")
    axs[0].legend(fontsize=6, ncol=2)
    axs[0].set_title("Observed (dots, 1 s.e.) vs predicted (lines)")
    axs[1].set_title("Phase spectra")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_misfit_maps(freqs, spec, Zpred, path):
    """Cross-plots of observed vs predicted amplitude and phase, all frequencies."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    for k, f in enumerate(freqs):
        s = np.sign(spec.Z[0].real)
        axs[0].loglog(np.abs(spec.Z[k]), np.abs(Zpred[k]), ".", ms=2, label=f"{f:.3g} Hz")
        axs[1].plot(-np.angle(s * spec.Z[k]) * 1e3, -np.angle(s * Zpred[k]) * 1e3, ".", ms=2)
    for a, lab in zip(axs, ("|Z| (ohm)", "-phase (mrad)")):
        lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
        a.plot(lim, lim, "k-", lw=0.6)
        a.set_xlabel(f"observed {lab}")
        a.set_ylabel(f"predicted {lab}")
    axs[0].legend(fontsize=6, markerscale=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(freqs, histories, path):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for f, h in zip(freqs, histories):
        it = [x["iter"] for x in h]
        axs[0].semilogy(it, [x["chi2_amp"] for x in h], "-o", ms=2, label=f"{f:.3g} Hz")
        axs[1].semilogy(it, [x["chi2_phase"] for x in h], "-o", ms=2)
    for a, t in zip(axs, ("amplitude", "phase")):
        a.axhline(1, color="k", ls=":")
        a.set_xlabel("Gauss-Newton iteration")
        a.set_ylabel(f"chi^2 / N ({t})")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _section(mesh, values, y):
    x, yn, z, arr = core_grid(mesh, values)
    yc = 0.5 * (yn[1:] + yn[:-1])
    j = int(np.argmin(np.abs(yc - y)))
    return x, z, arr[:, j, :].T


def _slice(mesh, values, zval):
    x, y, z, arr = core_grid(mesh, values)
    zc = 0.5 * (z[1:] + z[:-1])
    k = int(np.argmin(np.abs(zc - zval)))
    return x, y, arr[:, :, k].T, zc[k]


def plot_model_panels(mesh, panels, path, y_sections=(), z_slices=(), electrodes=None, mask=None,
                      suptitle=""):
    """panels: list of (title, values_full, dict(cmap=, log=, vmin=, vmax=, label=, mask=))."""
    ncol = len(y_sections) + len(z_slices)
    nrow = len(panels)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol + 0.8, 2.5 * nrow + 0.6), squeeze=False,
                            layout="constrained")
    for r, (title, vals, st) in enumerate(panels):
        v = np.asarray(vals, float).copy()
        if mask is not None:
            v[~mask] = np.nan
        if st.get("mask") is not None:
            v[~st["mask"]] = np.nan
        fin = v[np.isfinite(v)]
        vmin = st.get("vmin", np.nanpercentile(fin, 2) if fin.size else 0)
        vmax = st.get("vmax", np.nanpercentile(fin, 98) if fin.size else 1)
        norm = LogNorm(vmin, vmax) if st.get("log") else plt.Normalize(vmin, vmax)
        cmap = st.get("cmap", "viridis")
        im = None
        c = 0
        for y in y_sections:
            x, z, a = _section(mesh, v, y)
            ax = axs[r, c]
            im = ax.pcolormesh(x, z, a, norm=norm, cmap=cmap, shading="flat")
            ax.set_title(f"{title} | section y = {y:g} m")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            if electrodes is not None:
                sel = np.abs(electrodes[:, 1] - y) < 1e-6
                ax.plot(electrodes[sel, 0], electrodes[sel, 2], "kv", ms=2)
            c += 1
        for zv in z_slices:
            x, yy, a, zr = _slice(mesh, v, zv)
            ax = axs[r, c]
            im = ax.pcolormesh(x, yy, a, norm=norm, cmap=cmap, shading="flat")
            ax.set_title(f"{title} | z = {zr:.0f} m")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal")
            if electrodes is not None:
                ax.plot(electrodes[:, 0], electrodes[:, 1], "k.", ms=1)
            c += 1
        cb = fig.colorbar(im, ax=axs[r, :].tolist(), shrink=0.9, pad=0.01, aspect=12)
        cb.set_label(st.get("label", title))
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(path)
    plt.close(fig)

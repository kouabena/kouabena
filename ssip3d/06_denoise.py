"""Step 6: denoise and QC the full SSIP data (all dipoles, 4 frequencies).

For one injection the recorded dipoles are differences of electrode
potentials, V(MN) = phi(M) - phi(N). With all receiver pairs recorded, the
potential profile is heavily over-determined:

1. Consistency / reconstruction. For every line, injection, side of the
   injection and frequency, the complex electrode potentials are estimated by
   robust (Huber-IRLS) least squares from all dipoles, weighted by relative
   error. Each dipole is replaced by the reconstructed potential difference
   (denoised value). Dipoles that disagree with the reconstruction are
   flagged, and injections whose reconstruction is poor overall are
   flagged as bad (typically wrong positions).
2. Error estimation. F1..F4 come from different harmonics of the
   transmitted signal, so their noise is largely independent while the true
   amplitude and phase vary smoothly with log-frequency. The scatter about a
   linear trend in log f gives a noise estimate, smoothed as a function of
   signal level per line (single-datum estimates from 4 points are noisy).
3. Phase QC: phases outside [MIN_PHASE, MAX_ABS_PHASE] at any frequency
   (EM coupling / noise) are flagged.

Writes work/denoised_all.csv.gz (every dipole with original and denoised
values, errors and flags) and results/denoise/ QC figure and summary.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from ssip_io import read_res2dinv_ip, pole_dipole_k

KEY = ["c1x", "p1x", "p2x"]
FK = list(C.FREQS)
LOGF = np.log(np.array(list(C.FREQS.values())))


def load_line(name, prefix, y):
    """All dipoles of one line, 4 frequencies side by side, signed complex."""
    merged = None
    for fk in FK:
        df, info = read_res2dinv_ip(C.RAW_DIR / f"{prefix}_{fk}.dat")
        df = df.drop_duplicates(KEY)
        cols = {"val": f"val_{fk}", "err": f"err_{fk}",
                "phase": f"phase_{fk}", "phase_err": f"phase_err_{fk}"}
        df = df.rename(columns=cols)
        merged = df if merged is None else merged.merge(
            df[KEY + list(cols.values())], on=KEY, how="inner")
        is_res = info["is_resistance"]
    df = merged
    n = len(df)
    Y = np.full(n, y)
    c = np.c_[df.c1x, Y, df.c1z]
    m = np.c_[df.p1x, Y, df.p1z]
    nn = np.c_[df.p2x, Y, df.p2z]
    df["k"] = pole_dipole_k(c, m, nn)            # true 3D (with topography)
    # L22 is stored as apparent resistivity. The potential-consistency test
    # (README) shows it was computed with horizontal distances, so that k is
    # used to convert it back to resistance.
    zero = np.zeros(n)
    k_h = pole_dipole_k(np.c_[df.c1x, Y, zero], np.c_[df.p1x, Y, zero],
                        np.c_[df.p2x, Y, zero])
    sign = np.sign(df.k.values)
    for fk in FK:
        amp = np.abs(df[f"val_{fk}"].values)
        err = np.abs(df[f"err_{fk}"].values)
        if not is_res:
            amp, err = amp / np.abs(k_h), err / np.abs(k_h)
        df[f"R_{fk}"] = amp                       # |transfer resistance| (Ohm)
        df[f"Rerr_rep_{fk}"] = err
        # signed complex transfer impedance; phase in mrad, lagging positive
        df[f"z_{fk}"] = sign * amp * np.exp(-1j * df[f"phase_{fk}"] / 1000)
    df["line"] = name
    df["y"] = y
    df["straddle"] = (df.p1x < df.c1x) & (df.p2x > df.c1x)
    df["side"] = np.where(df.p1x > df.c1x, 1, -1)
    df["stored_as"] = "resistance" if is_res else "app_resistivity"
    return df


def reconstruct(d, z):
    """Robust complex potential reconstruction for one injection/side.

    Returns fitted dipole values and the robust relative residual scale.
    """
    nodes = np.unique(np.r_[d.p1x.values, d.p2x.values])
    idx = {x: i for i, x in enumerate(nodes)}
    A = np.zeros((len(d), len(nodes)))
    rows = np.arange(len(d))
    A[rows, [idx[x] for x in d.p1x]] = 1.0
    A[rows, [idx[x] for x in d.p2x]] = -1.0
    w = 1.0 / np.abs(z)
    for _ in range(8):                           # Huber IRLS
        phi = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
        fit = A @ phi
        r = np.abs(z - fit) / np.abs(z)
        s = 1.4826 * np.median(r) + 1e-9
        w = np.minimum(1.0, 1.5 * s / (r + 1e-12)) / np.abs(z)
    return fit, r, s


def denoise_line(df):
    for fk in FK:
        df[f"zfit_{fk}"] = np.nan + 0j
        df[f"cons_res_{fk}"] = np.nan
    tx_rows = []
    groups = df[~df.straddle].groupby(["c1x", "side"]).indices
    for (c, side), ii in groups.items():
        ii = df.index[~df.straddle][ii]
        if len(ii) < 6:
            continue
        d = df.loc[ii]
        scales = []
        for fk in FK:
            fit, r, s = reconstruct(d, d[f"z_{fk}"].values)
            df.loc[ii, f"zfit_{fk}"] = fit
            df.loc[ii, f"cons_res_{fk}"] = r
            scales.append(s)
        tx_rows.append(dict(line=df.line.iloc[0], c1x=c, side=side,
                            n=len(ii), cons_scale=np.median(scales),
                            cons_med=np.median(df.loc[ii, "cons_res_F1"])))
    return df, pd.DataFrame(tx_rows)


def phase_of(z, k):
    """Phase (mrad, lagging positive) of a signed complex transfer impedance.

    The sign of the geometric factor is removed first; otherwise dipoles on
    the negative side of the injection come out shifted by pi.
    """
    return -np.angle(np.asarray(z) * np.sign(np.asarray(k))) * 1000


def frequency_noise(df):
    """Noise estimate from scatter across F1..F4 about a linear log-f trend."""
    lnR = np.log(np.abs(np.c_[tuple(df[f"zfit_{fk}"] for fk in FK)]))
    ph = np.c_[tuple(phase_of(df[f"zfit_{fk}"], df.k) for fk in FK)]
    G = np.c_[np.ones(4), LOGF - LOGF.mean()]
    P = G @ np.linalg.pinv(G)                    # hat matrix, 4 x 4
    rl = lnR - lnR @ P.T
    rp = ph - ph @ P.T
    dof = 4 - 2
    df["freq_rel_raw"] = np.sqrt((rl ** 2).sum(1) / dof)     # relative
    df["freq_phase_raw"] = np.sqrt((rp ** 2).sum(1) / dof)   # mrad
    # smooth: median of the single-datum estimates in bins of signal level
    lvl = np.log10(np.abs(df["zfit_F1"]).values)
    ok = np.isfinite(lvl) & np.isfinite(df.freq_rel_raw)
    edges = np.quantile(lvl[ok], np.linspace(0, 1, 31))
    b = np.clip(np.searchsorted(edges, lvl, side="right") - 1, 0, 29)
    for col, out in (("freq_rel_raw", "freq_rel"),
                     ("freq_phase_raw", "freq_phase")):
        med = pd.Series(df[col].values[ok]).groupby(b[ok]).median()
        # a median of single-datum RMS (chi with 2 dof) underestimates the
        # sigma by sqrt(2 ln 2 / 2) = 0.833
        df[out] = med.reindex(b).values / 0.833
    return df


def main():
    out = C.RESULTS_DIR / "denoise"
    out.mkdir(parents=True, exist_ok=True)
    frames, txs = [], []
    for name, (prefix, y) in C.LINES.items():
        df = load_line(name, prefix, y)
        df, tx = denoise_line(df)
        df = frequency_noise(df)
        frames.append(df)
        txs.append(tx)
        print(f"{name}: {len(df)} dipoles, consistency residual (F1) "
              f"median {np.nanmedian(df.cons_res_F1) * 100:.3f}%, p99 "
              f"{np.nanpercentile(df.cons_res_F1, 99) * 100:.2f}%")
    df = pd.concat(frames, ignore_index=True)
    tx = pd.concat(txs, ignore_index=True)

    # ---------------- flags --------------------------------------------
    tx["bad_tx"] = tx.cons_med > C.DN_BAD_TX_RES
    bad_tx = set(map(tuple, tx.loc[tx.bad_tx, ["line", "c1x", "side"]].values))
    df["flag_bad_tx"] = [(l, c, s) in bad_tx for l, c, s in
                         zip(df.line, df.c1x, df.side)]
    cons_max = np.nanmax(np.c_[tuple(df[f"cons_res_{fk}"] for fk in FK)], 1)
    df["flag_inconsistent"] = cons_max > C.DN_MAX_CONS_RES
    df["flag_straddle"] = df.straddle
    ph = np.c_[tuple(phase_of(df[f"zfit_{fk}"], df.k) for fk in FK)]
    df["flag_phase"] = ((ph < C.MIN_PHASE) | (ph > C.MAX_ABS_PHASE)).any(1)
    df["dc_ok"] = ~(df.flag_bad_tx | df.flag_inconsistent | df.flag_straddle
                    | df.zfit_F1.isna())
    df["ip_ok"] = df.dc_ok & ~df.flag_phase

    # ---------------- denoised values and errors ------------------------
    for fk in FK:
        df[f"R_dn_{fk}"] = np.abs(df[f"zfit_{fk}"])
        df[f"phase_dn_{fk}"] = phase_of(df[f"zfit_{fk}"], df.k)
    # relative error of the resistance: frequency-scatter noise, residual
    # inconsistency and a small floor, added in quadrature
    df["R_relerr"] = np.sqrt(df.freq_rel ** 2 + df.cons_res_F1 ** 2 +
                             C.DN_REL_FLOOR ** 2)
    for fk in FK:
        df[f"phase_err_dn_{fk}"] = np.sqrt(
            df.freq_phase ** 2 + (df[f"cons_res_{fk}"] * 1000) ** 2 +
            C.DN_PHASE_FLOOR ** 2)

    keep = ([c for c in df.columns if not c.startswith(("z_", "zfit_"))])
    df[keep].to_csv(C.WORK_DIR / "denoised_all.csv.gz", index=False,
                    float_format="%.6g")
    tx.to_csv(out / "injection_consistency.csv", index=False,
              float_format="%.5g")

    # ---------------- summary + QC figure --------------------------------
    lines = ["Denoising summary (all dipoles, before selection)", ""]
    for name, g in df.groupby("line"):
        lines.append(
            f"{name}: {len(g)} dipoles | straddling {g.flag_straddle.sum()} | "
            f"bad injection {g.flag_bad_tx.sum()} | inconsistent "
            f"{g.flag_inconsistent.sum()} | DC ok {g.dc_ok.sum()} | "
            f"phase flagged {(g.dc_ok & g.flag_phase).sum()} | IP ok "
            f"{g.ip_ok.sum()}")
        gg = g[g.dc_ok]
        lines.append(
            f"      rel. error of R: median {gg.R_relerr.median() * 100:.2f}%"
            f" (p10-p90 {gg.R_relerr.quantile(.1) * 100:.2f}-"
            f"{gg.R_relerr.quantile(.9) * 100:.2f}%) | phase error F1: "
            f"median {gg.phase_err_dn_F1.median():.2f} mrad | reported "
            f"phase error F1: median {gg.phase_err_F1.median():.2f} mrad")
    badl = tx[tx.bad_tx]
    lines += ["", "Injections flagged as bad (median consistency residual "
              f"> {C.DN_BAD_TX_RES * 100:.1f}%):"]
    lines += [f"  {r.line} C1 x = {r.c1x:.0f} m, side {'+' if r.side > 0 else '-'}"
              f": {r.cons_med * 100:.2f}% ({r.n} dipoles)"
              for r in badl.itertuples()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    colors = {"L22": "#2a6f97", "L24": "#c8553d", "L26": "#5a8f29"}
    for name, g in df[~df.straddle].groupby("line"):
        r = g.cons_res_F1.dropna().clip(lower=1e-6)
        axs[0].hist(np.log10(r), bins=80, histtype="step", lw=1.4,
                    color=colors[name], label=name)
        gg = g[g.dc_ok].sort_values("R_dn_F1")
        axs[1].plot(gg.R_dn_F1, gg.freq_rel * 100, color=colors[name], lw=1.4,
                    label=name)
        axs[2].plot(gg.R_dn_F1, gg.freq_phase, color=colors[name], lw=1.4,
                    label=name)
    axs[0].axvline(np.log10(C.DN_MAX_CONS_RES), color="0.3", ls="--", lw=1)
    axs[0].set_xlabel("log10 consistency residual (relative, F1)")
    axs[0].set_ylabel("dipoles")
    axs[0].set_title("Potential-reconstruction residual", fontsize=10,
                     loc="left")
    for ax, lab in ((axs[1], "relative noise of R (%)"),
                    (axs[2], "phase noise (mrad)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|transfer resistance| (Ω)")
        ax.set_ylabel(lab)
    axs[1].set_title("Noise from F1–F4 scatter", fontsize=10, loc="left")
    axs[2].set_title("Phase noise from F1–F4 scatter", fontsize=10,
                     loc="left")
    for ax in axs:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "fig_denoise_qc.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

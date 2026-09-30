"""Step 7: write the denoised data as ZondRes3D input files.

Reads work/denoised_all.csv.gz (06_denoise.py) and writes zond/:

  <line>_F1.dat .. <line>_F4.dat   Res2DInv general array with IP (resistance,
                                   phase in mrad, absolute errors), identical
                                   geometry in all 4 files -> ZondRes3D
                                   "Collect from 2D" + "Collect IP"
  <line>_res.dat                   resistance only, all data passing DC QC
                                   (more data than the IP files)
  <line>_F1.z2d ... / <line>_res.z2d   the same in ZondRes2D format, with
                                   weights derived from the error estimates
  ssip3d_denoised_3d.csv           every selected datum with full x, y, z of
                                   A, M, N (B remote) for other software

Selection: the same non-redundant dipole subset as the pyGIMLi inversion
(01_prepare_data.select_dipoles), MIN_OFFSET and EXCLUDE_TX from config, the
pass-1 misfit outliers (config.OUTLIER_FILE), plus the injections flagged by
06_denoise.py.
"""
import importlib
import numpy as np
import pandas as pd

import config as C
from ssip_io import REMOTE

prep = importlib.import_module("01_prepare_data")
FK = list(C.FREQS)
OUT = C.ROOT / "zond"


def fmt_res2dinv_row(r, fk=None):
    base = (f"3  {r.c1x:<12.6f}  {r.c1z:<12.6f}  {r.p1x:<12.6f}  "
            f"{r.p1z:<12.6f}  {r.p2x:<12.6f}  {r.p2z:<12.6f}  ")
    R, err = r.R_dn_F1 if fk is None else getattr(r, f"R_dn_{fk}"), None
    err = R * r.R_relerr
    s = base + f"{R:.5E}  {err:.5E}"
    if fk is not None:
        s += (f"  {getattr(r, f'phase_dn_{fk}'):.6f}  "
              f"{getattr(r, f'phase_err_dn_{fk}'):.6f}")
    return s


def write_res2dinv(path, d, title, fk=None):
    hdr = [title, "10.000000", "11", "0",
           "Type of measurement (0=app. resistivity,1=resistance)", "1",
           f"{len(d):<20d}", "1"]
    if fk is None:
        hdr += ["0"]
    else:
        hdr += ["1", "Phase Angle", "mrad", f"0,{C.FREQS[fk]:g}"]
    hdr += ["Error estimate for data present",
            "Type of error estimate (0=same unit as data)", "0"]
    rows = [fmt_res2dinv_row(r, fk) for r in d.itertuples()]
    path.write_text("\r\n".join(hdr + rows) + "\r\n")


def zond_weight(d):
    """min(1, 3 % / relative error) times the user's ZondRes2D weight."""
    return (np.minimum(1.0, C.ZOND_W_REF_ERR / d.R_relerr.values)
            * d.zond_user_weight.values)


def eta_from_phase(phase_mrad):
    """ZondRes2D converts phase to eta_a (%) as 100 * tan(phase)."""
    return 100.0 * np.tan(np.asarray(phase_mrad) / 1000.0)


def write_z2d(path, d, fk=None):
    """ZondRes2D text format, laid out like the .z2d files of this survey.

    weight: see zond_weight(). eta_a: 100 * tan(phase), ZondRes2D's own
    conversion found in the original .z2d files (8.0 mrad -> 0.80).
    """
    def e(v):
        """Number as in the original .z2d files: ' 1.23450000000000E+0003'."""
        mant, exp = f"{v: .14E}".split("E")
        return f"{mant}E{exp[0]}{abs(int(exp)):04d}"
    w = zond_weight(d)
    R = d.R_dn_F1.values if fk is None else d[f"R_dn_{fk}"].values
    head = "c1 p1 c2 p2 res weight" + ("" if fk is None else " eta_a") + " "
    lines = [head]
    for i, r in enumerate(d.itertuples()):
        row = [e(r.c1x), e(r.p1x), e(REMOTE), e(r.p2x), e(R[i]), e(w[i])]
        if fk is not None:
            row.append(e(eta_from_phase(getattr(r, f"phase_dn_{fk}"))))
        lines.append(" ".join(row) + " ")
    topo = pd.concat([d[["c1x", "c1z"]].set_axis(["x", "z"], axis=1),
                      d[["p1x", "p1z"]].set_axis(["x", "z"], axis=1),
                      d[["p2x", "p2z"]].set_axis(["x", "z"], axis=1)])
    topo = topo.groupby("x", as_index=False).z.mean().sort_values("x")
    lines.append("topo#")
    lines.append(f"{e(0.0)} {e(0.0)}")
    lines += [f"{e(x)} {e(z)}" for x, z in zip(topo.x, topo.z)]
    path.write_text("\r\n".join(lines) + "\r\n")


def write_z3d(path, t):
    """Native ZondRes3D text data file (manual, "Z3D file format").

    One file for all lines (prof = 0, 1, 2), pole-dipole (C2 columns omitted),
    resistance (recommended with topography), 4-frequency IP as multichannel
    frequency-domain data: mod1..4 = |R| and pha1..4 = phase in mrad with
    Zond's sign convention (phase shift negative for a lagging voltage).
    Electrodes are on the surface (z = 0); elevations are in the Topo block.
    Rows whose phase failed QC get weightip = 0.
    """
    freqs = " ".join(f"{C.FREQS[fk]:g}" for fk in FK)
    head = ("prof c1x p1x p2x c1y p1y p2y c1z p1z p2z weight res "
            + " ".join(f"mod{i + 1}" for i in range(len(FK))) + " "
            + " ".join(f"pha{i + 1}" for i in range(len(FK))) + " weightip")
    prof = {n: i for i, n in enumerate(C.LINES)}
    w = zond_weight(t)
    perr = np.sqrt(np.mean([t[f"phase_err_dn_{fk}"].values ** 2
                            for fk in FK], axis=0))
    wip = np.where(t.ip_ok.values,
                   np.minimum(1.0, C.ZOND_W_REF_PHASE / perr)
                   * t.zond_user_weight.values, 0.0)
    lines = [f"time_#chann {freqs}", head]
    for i, r in enumerate(t.itertuples()):
        vals = [f"{prof[r.line]}", f"{r.c1x:.3f}", f"{r.p1x:.3f}",
                f"{r.p2x:.3f}", f"{r.y:.3f}", f"{r.y:.3f}", f"{r.y:.3f}",
                "0", "0", "0", f"{w[i]:.4f}", f"{r.R_dn_F1:.6e}"]
        vals += [f"{getattr(r, f'R_dn_{fk}'):.6e}" for fk in FK]
        vals += [f"{-getattr(r, f'phase_dn_{fk}'):.4f}" for fk in FK]
        vals.append(f"{wip[i]:.4f}")
        lines.append(" ".join(vals))
    topo = pd.concat([
        t[["c1x", "y", "c1z"]].set_axis(["x", "y", "z"], axis=1),
        t[["p1x", "y", "p1z"]].set_axis(["x", "y", "z"], axis=1),
        t[["p2x", "y", "p2z"]].set_axis(["x", "y", "z"], axis=1)])
    topo = topo.groupby(["x", "y"], as_index=False).z.mean()
    lines.append("Topo")
    lines += [f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in
              zip(topo.x, topo.y, topo.z)]
    path.write_text("\r\n".join(lines) + "\r\n")
    return len(topo)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(C.WORK_DIR / "denoised_all.csv.gz")
    offset = np.minimum(np.abs(df.p1x - df.c1x), np.abs(df.p2x - df.c1x))
    manual = np.zeros(len(df), bool)
    for line, xs in C.EXCLUDE_TX.items():
        manual |= (df.line == line) & df.c1x.isin(xs)
    base = df.dc_ok & ~manual & (offset >= C.MIN_OFFSET)
    summary = []
    table = []
    for name, (_, y) in C.LINES.items():
        g = df[base & (df.line == name)]
        sel = prep.select_dipoles(g)                      # non-redundant
        # same misfit-outlier removal as the original pass 2 (01_prepare_data)
        if C.OUTLIER_FILE.exists():
            out = pd.read_csv(C.OUTLIER_FILE)
            out = out[out.line == name][["c1x", "p1x", "p2x"]].assign(_o=1)
            sel = sel.merge(out, on=["c1x", "p1x", "p2x"], how="left")
            n_out = int(sel._o.notna().sum())
            sel = sel[sel._o.isna()].drop(columns="_o")
        else:
            n_out = 0
        ip = sel[sel.ip_ok]
        write_res2dinv(OUT / f"{name}_res.dat", sel,
                       f"{name} SSIP denoised resistance")
        write_z2d(OUT / f"{name}_res.z2d", sel)
        for fk in FK:
            write_res2dinv(OUT / f"{name}_{fk}.dat", ip,
                           f"{name} SSIP denoised {fk} {C.FREQS[fk]:g} Hz",
                           fk)
            write_z2d(OUT / f"{name}_{fk}.z2d", ip, fk)
        summary.append(f"{name}: pass-1 misfit outliers removed: {n_out}")
        summary.append(f"{name}: y = {y:.0f} m | resistance {len(sel)} data "
                       f"| IP {len(ip)} data | injections "
                       f"{sel.c1x.nunique()}")
        t = sel.assign(ip_ok=sel.ip_ok)
        table.append(t)
    t = pd.concat(table, ignore_index=True)
    k3d = t.k.values
    out = pd.DataFrame({
        "line": t.line,
        "Ax": t.c1x, "Ay": t.y, "Az": t.c1z,
        "Mx": t.p1x, "My": t.y, "Mz": t.p1z,
        "Nx": t.p2x, "Ny": t.y, "Nz": t.p2z,
        "R_ohm": np.sign(k3d) * t.R_dn_F1, "k_3d": k3d,
        "rhoa_ohmm": np.abs(k3d) * t.R_dn_F1, "R_relerr": t.R_relerr,
        "zond_user_weight": t.zond_user_weight,
        "ip_ok": t.ip_ok.astype(int)})
    for fk in FK:
        out[f"phase_{fk}_mrad"] = t[f"phase_dn_{fk}"]
        out[f"phase_err_{fk}_mrad"] = t[f"phase_err_dn_{fk}"]
    out.to_csv(OUT / "ssip3d_denoised_3d.csv", index=False,
               float_format="%.6g")
    n_el = write_z3d(OUT / "ssip3d_denoised.z3d", t)
    summary.append(f"total: resistance {len(out)} | IP {out.ip_ok.sum()} | "
                   f"unique electrode positions {n_el}")
    for name in C.LINES:
        g = t[t.line == name]
        xs = np.r_[g.c1x, g.p1x, g.p2x]
        summary.append(f"{name}: x range {xs.min():.0f} .. {xs.max():.0f} m "
                       "(start/end for Collect from 2D)")
    (OUT / "export_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

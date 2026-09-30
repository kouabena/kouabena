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
(01_prepare_data.select_dipoles), MIN_OFFSET and EXCLUDE_TX from config, plus
the injections flagged by 06_denoise.py.
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


def write_z2d(path, d, fk=None):
    """ZondRes2D text format, laid out like the .z2d files of this survey.

    weight: C.ZOND_W_REF_ERR / relative error, capped at 1 (so data at the
    reference error level or better get full weight). eta_a: phase / 10, the
    convention of the original .z2d files (8.0 mrad -> 0.80).
    """
    def e(v):
        """Number as in the original .z2d files: ' 1.23450000000000E+0003'."""
        mant, exp = f"{v: .14E}".split("E")
        return f"{mant}E{exp[0]}{abs(int(exp)):04d}"
    w = np.minimum(1.0, C.ZOND_W_REF_ERR / d.R_relerr.values)
    R = d.R_dn_F1.values if fk is None else d[f"R_dn_{fk}"].values
    head = "c1 p1 c2 p2 res weight" + ("" if fk is None else " eta_a") + " "
    lines = [head]
    for i, r in enumerate(d.itertuples()):
        row = [e(r.c1x), e(r.p1x), e(REMOTE), e(r.p2x), e(R[i]), e(w[i])]
        if fk is not None:
            row.append(e(getattr(r, f"phase_dn_{fk}") / 10.0))
        lines.append(" ".join(row) + " ")
    topo = pd.concat([d[["c1x", "c1z"]].set_axis(["x", "z"], axis=1),
                      d[["p1x", "p1z"]].set_axis(["x", "z"], axis=1),
                      d[["p2x", "p2z"]].set_axis(["x", "z"], axis=1)])
    topo = topo.groupby("x", as_index=False).z.mean().sort_values("x")
    lines.append("topo#")
    lines.append(f"{e(0.0)} {e(0.0)}")
    lines += [f"{e(x)} {e(z)}" for x, z in zip(topo.x, topo.z)]
    path.write_text("\r\n".join(lines) + "\r\n")


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
        ip = sel[sel.ip_ok]
        write_res2dinv(OUT / f"{name}_res.dat", sel,
                       f"{name} SSIP denoised resistance")
        write_z2d(OUT / f"{name}_res.z2d", sel)
        for fk in FK:
            write_res2dinv(OUT / f"{name}_{fk}.dat", ip,
                           f"{name} SSIP denoised {fk} {C.FREQS[fk]:g} Hz",
                           fk)
            write_z2d(OUT / f"{name}_{fk}.z2d", ip, fk)
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
        "ip_ok": t.ip_ok.astype(int)})
    for fk in FK:
        out[f"phase_{fk}_mrad"] = t[f"phase_dn_{fk}"]
        out[f"phase_err_{fk}_mrad"] = t[f"phase_err_dn_{fk}"]
    out.to_csv(OUT / "ssip3d_denoised_3d.csv", index=False,
               float_format="%.6g")
    summary.append(f"total: resistance {len(out)} | IP {out.ip_ok.sum()}")
    (OUT / "export_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

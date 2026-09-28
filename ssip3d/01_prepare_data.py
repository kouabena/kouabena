"""Step 1: merge the three SSIP lines (4 frequencies each) into one 3D dataset.

* places each line at its y offset (config.LINES),
* converts everything to transfer resistance and recomputes the pole-dipole
  geometric factor from the true 3D electrode positions (topography included),
* selects a non-redundant subset of the all-pairs dipoles (see README),
* writes work/ssip3d.dat (pyGIMLi unified data format) with rhoa, err and the
  phase + phase error of every frequency, and a CSV of the same table.
"""
import numpy as np
import pandas as pd
import pygimli as pg
from pygimli.physics import ert

import config as C
from ssip_io import read_res2dinv_ip, pole_dipole_k

KEY = ["c1x", "p1x", "p2x"]


def load_line(prefix):
    """Merge F1..F4 of one line on the electrode geometry."""
    merged = None
    for fk in C.FREQS:
        df, info = read_res2dinv_ip(C.RAW_DIR / f"{prefix}_{fk}.dat")
        assert abs(info["freq"] - C.FREQS[fk]) < 1e-6, (prefix, fk, info)
        df = df.drop_duplicates(KEY)
        df = df.rename(columns={"val": f"val_{fk}", "err": f"err_{fk}",
                                "phase": f"phase_{fk}",
                                "phase_err": f"phase_err_{fk}"})
        if merged is None:
            merged = df
            is_res = info["is_resistance"]
        else:
            merged = merged.merge(df[KEY + [c for c in df if c.endswith(fk)]],
                                  on=KEY, how="inner")
    return merged, is_res


def select_dipoles(df):
    """Keep one dipole per Tx and receiver electrode, sized by offset.

    With a distributed all-pairs recording every dipole of one Tx is a sum of
    the adjacent (40 m) dipoles, so most of the ~55k values per line are
    redundant. For each Tx and each receiver electrode P (at least MIN_OFFSET
    away) we keep the dipole P -> P + L pointing away from the Tx, with
    L = 40 * 2^j the shortest length >= offset / MAX_N (capped at MAX_DIPOLE),
    i.e. pole-dipole n <= MAX_N with long dipoles only where the signal is
    small. If that dipole was removed in QC, the next shorter one is used.
    """
    keep = []
    have = set(zip(df.c1x, df.p1x, df.p2x))
    elec = np.unique(np.r_[df.p1x, df.p2x])

    def length_for(offset):
        L = C.RX_SPACING
        while L < offset / C.MAX_N and L < C.MAX_DIPOLE:
            L *= 2
        return L

    for c in np.unique(df.c1x):
        for side in (+1, -1):
            for p in elec[(side * (elec - c)) >= C.MIN_OFFSET]:
                L = length_for(abs(p - c))
                while L >= C.RX_SPACING:
                    q = p + side * L
                    a, b = min(p, q), max(p, q)
                    if (c, a, b) in have:
                        keep.append((c, a, b))
                        break
                    L /= 2
    sel = pd.DataFrame(keep, columns=KEY).drop_duplicates()
    return df.merge(sel, on=KEY, how="inner")


def main():
    C.WORK_DIR.mkdir(exist_ok=True)
    tables = []
    for name, (prefix, y) in C.LINES.items():
        df, is_res = load_line(prefix)
        n_all = len(df)
        c = np.c_[df.c1x, np.full(len(df), y), df.c1z]
        m = np.c_[df.p1x, np.full(len(df), y), df.p1z]
        n = np.c_[df.p2x, np.full(len(df), y), df.p2z]
        k = pole_dipole_k(c, m, n)
        # Values are stored as magnitudes. Convert all lines to |R| (Ohm).
        fk = C.DC_FREQ
        if is_res:
            r_abs = np.abs(df[f"val_{fk}"].values)
        else:  # apparent resistivity; assume it was computed with the
            #    half-space pole-dipole k of the same electrode positions
            r_abs = np.abs(df[f"val_{fk}"].values) / np.abs(k)
        rel_err = np.abs(df[f"err_{fk}"].values / df[f"val_{fk}"].values)
        df["y"] = y
        df["line"] = name
        df["k"] = k
        df["r"] = np.sign(k) * r_abs          # signed transfer resistance
        df["rhoa"] = np.abs(k) * r_abs
        df["rel_err_rep"] = rel_err

        # --- quality control -------------------------------------------
        ok = (r_abs > 0) & np.isfinite(k) & (rel_err <= C.MAX_REL_ERR_DC)
        # dipoles straddling the Tx carry no reliable sign/magnitude
        ok &= ~((df.p1x < df.c1x) & (df.p2x > df.c1x))
        df = df[ok]
        df = select_dipoles(df)
        print(f"{name}: {n_all} data -> {ok.sum()} after QC -> "
              f"{len(df)} selected")
        tables.append(df)

    tab = pd.concat(tables, ignore_index=True)

    # --- build the pyGIMLi data container ---------------------------------
    pos = np.unique(np.round(np.r_[
        tab[["c1x", "y", "c1z"]].values,
        tab[["p1x", "y", "p1z"]].values,
        tab[["p2x", "y", "p2z"]].values], 3), axis=0)
    # one sensor per (x, y); a Tx and Rx never share a location here
    pos = pd.DataFrame(pos, columns=["x", "y", "z"]).groupby(
        ["x", "y"], as_index=False).z.mean().values
    idx = {(x, y): i for i, (x, y, _) in enumerate(pos)}

    data = ert.DataContainer()
    for p in pos:
        data.createSensor(p)
    data.resize(len(tab))
    data["a"] = [idx[(x, y)] for x, y in zip(tab.c1x, tab.y)]
    data["b"] = np.full(len(tab), -1)
    data["m"] = [idx[(x, y)] for x, y in zip(tab.p1x, tab.y)]
    data["n"] = [idx[(x, y)] for x, y in zip(tab.p2x, tab.y)]
    data["k"] = tab.k.values
    data["r"] = tab.r.values
    data["rhoa"] = tab.rhoa.values
    data["err"] = np.maximum(tab.rel_err_rep.values, C.ERR_FLOOR_DC)
    data["valid"] = 1
    for fk in C.FREQS:
        data[f"ip{fk}"] = tab[f"phase_{fk}"].values
        data[f"iperr{fk}"] = tab[f"phase_err_{fk}"].values
    data.markValid(data["rhoa"] > 0)
    data.removeInvalid()
    data.save(str(C.WORK_DIR / "ssip3d.dat"),
              "a b m n k r rhoa err " + " ".join(
                  f"ip{f} iperr{f}" for f in C.FREQS))
    tab.to_csv(C.WORK_DIR / "ssip3d_selected.csv", index=False)
    print(data)
    print("rhoa range", np.min(data["rhoa"]), np.max(data["rhoa"]),
          "median", np.median(data["rhoa"]))


if __name__ == "__main__":
    main()

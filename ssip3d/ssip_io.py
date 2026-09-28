"""Readers for the RES2DINV 'General array with IP' files exported for SSIP."""
import numpy as np
import pandas as pd

REMOTE = 1.23456789e8  # ZondRes2D marker for an electrode at infinity


def read_res2dinv_ip(fname):
    """Read one RES2DINV general-array IP file (pole-dipole, with topography).

    Returns a DataFrame with columns c1x c1z p1x p1z p2x p2z val err phase
    phase_err and the header info (freq in Hz, is_resistance).
    """
    with open(fname) as fh:
        lines = fh.read().splitlines()
    is_resistance = int(lines[5].strip()) == 1   # 0 = app. res, 1 = resistance
    n = int(lines[6].strip())
    freq = float(lines[11].split(",")[1])
    rows = [ln.split() for ln in lines[15:15 + n]]
    a = np.array(rows, dtype=float)
    if a.shape[1] != 11 or int(a[0, 0]) != 3:
        raise ValueError(f"{fname}: expected 3-electrode rows with 11 columns")
    df = pd.DataFrame(a[:, 1:], columns=["c1x", "c1z", "p1x", "p1z", "p2x",
                                         "p2z", "val", "err", "phase",
                                         "phase_err"])
    # keep P1 < P2 (the files already do, but be safe)
    swap = df.p1x > df.p2x
    if swap.any():
        df.loc[swap, ["p1x", "p1z", "p2x", "p2z"]] = \
            df.loc[swap, ["p2x", "p2z", "p1x", "p1z"]].values
    return df, dict(freq=freq, is_resistance=is_resistance)


def pole_dipole_k(c, m, n):
    """Half-space geometric factor for pole-dipole C1-P1-P2 (C2 at infinity).

    c, m, n: (N, 3) arrays of electrode coordinates.
    """
    rcm = np.linalg.norm(c - m, axis=1)
    rcn = np.linalg.norm(c - n, axis=1)
    return 2.0 * np.pi / (1.0 / rcm - 1.0 / rcn)

"""Step 8: pyGIMLi dataset from exactly the data exported for ZondRes3D.

Reads zond/ssip3d_denoised_3d.csv (07_export_zond.py) and writes
work_denoised/ssip3d.dat, so pyGIMLi and ZondRes3D invert identical data:
same selection, denoised values, the same per-datum error estimates and the
user's ZondRes2D weights (as error inflation 1/w).

Then run the usual steps on it:
    SSIP_VARIANT=denoised python 02_make_mesh.py
    SSIP_VARIANT=denoised python 03_invert.py
    SSIP_VARIANT=denoised python 04_export_results.py
"""
import os
import numpy as np
import pandas as pd

os.environ["SSIP_VARIANT"] = "denoised"
import config as C                                      # noqa: E402
from pygimli.physics import ert                         # noqa: E402

FK = list(C.FREQS)


def main():
    C.WORK_DIR.mkdir(exist_ok=True)
    t = pd.read_csv(C.ROOT / "zond" / "ssip3d_denoised_3d.csv")
    pos = np.unique(np.round(np.r_[t[["Ax", "Ay", "Az"]].values,
                                   t[["Mx", "My", "Mz"]].values,
                                   t[["Nx", "Ny", "Nz"]].values], 3), axis=0)
    pos = pd.DataFrame(pos, columns=["x", "y", "z"]).groupby(
        ["x", "y"], as_index=False).z.mean().values
    idx = {(x, y): i for i, (x, y, _) in enumerate(pos)}
    data = ert.DataContainer()
    for p in pos:
        data.createSensor(p)
    data.resize(len(t))
    data["a"] = [idx[(x, y)] for x, y in zip(t.Ax, t.Ay)]
    data["b"] = np.full(len(t), -1)
    data["m"] = [idx[(x, y)] for x, y in zip(t.Mx, t.My)]
    data["n"] = [idx[(x, y)] for x, y in zip(t.Nx, t.Ny)]
    data["k"] = t.k_3d.values
    data["r"] = t.R_ohm.values
    data["rhoa"] = t.rhoa_ohmm.values
    # measurement error (denoising) plus the modelling-error floor used in
    # the original inversion, inflated by the user's ZondRes2D weight
    data["err"] = (np.sqrt(t.R_relerr ** 2 + C.ERR_FLOOR_DC ** 2)
                   / t.zond_user_weight).values
    for fk in FK:
        data[f"ip{fk}"] = t[f"phase_{fk}_mrad"].values
        data[f"iperr{fk}"] = (t[f"phase_err_{fk}_mrad"]
                              / t.zond_user_weight).values
    data["ipok"] = t.ip_ok.values
    data["valid"] = 1
    data.save(str(C.WORK_DIR / "ssip3d.dat"),
              "a b m n k r rhoa err ipok " + " ".join(
                  f"ip{f} iperr{f}" for f in FK))
    print(data)
    print(f"err: median {np.median(data['err']) * 100:.1f}%, "
          f"user-weighted rows {(t.zond_user_weight < 1).sum()}, "
          f"IP rows {int(t.ip_ok.sum())}")


if __name__ == "__main__":
    main()

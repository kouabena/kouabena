"""List data misfit by more than OUTLIER_SIGMA after an inversion run.

Run after 03_invert.py (pass 1); writes config.OUTLIER_FILE, which
01_prepare_data.py then uses to remove these data for pass 2.
"""
import numpy as np
import pandas as pd

import config as C


def main():
    tab = pd.read_csv(C.WORK_DIR / "ssip3d_selected.csv")
    obs = np.load(C.RESULTS_DIR / "rhoa_data.npy")
    pre = np.load(C.RESULTS_DIR / "rhoa_response.npy")
    assert len(tab) == len(obs) and np.allclose(tab.rhoa, obs, rtol=1e-4)
    err = np.maximum(tab.rel_err_rep.values, C.ERR_FLOOR_DC)
    nres = np.log(pre / obs) / err
    bad = np.abs(nres) > C.OUTLIER_SIGMA
    out = tab.loc[bad, ["line", "c1x", "p1x", "p2x", "rhoa"]].copy()
    out["normalised_misfit"] = nres[bad]
    out.to_csv(C.OUTLIER_FILE, index=False)
    print(f"{bad.sum()} of {len(bad)} data flagged -> {C.OUTLIER_FILE.name}")


if __name__ == "__main__":
    main()

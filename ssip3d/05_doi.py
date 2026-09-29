"""Step 5: depth-of-investigation (DOI) test (Oldenburg & Li 1999).

The data are inverted twice more, with homogeneous reference models
m_ref1 = 0.1 * m0 and m_ref2 = 10 * m0, and a regularization that includes a
small smallness term: ||C_s (m - m_ref)||^2 + alpha_s^2 ||m - m_ref||^2 (log
parameters). The DOI index per cell is

    R = (log m1 - log m2) / (log m_ref1 - log m_ref2)

R ~ 0 where the data determine the model, R -> 1 where the model just returns
to the reference. Cells with R < DOI_CUTOFF are considered resolved.

Usage:  python 05_doi.py dc     two resistivity inversions (~35 min each)
        python 05_doi.py ip     two phase inversions at F1 (~25 min each),
                                using the Jacobian of the final model
        python 05_doi.py index  compute R and save results/doi_*.npy
        python 05_doi.py ... --test   quick API check on 300 data
"""
import importlib
import sys
import time
import numpy as np
import pygimli as pg
from pygimli.physics import ert

import config as C

inv3 = importlib.import_module("03_invert")
TEST = "--test" in sys.argv
OUT = C.RESULTS_DIR / ("test" if TEST else "")
FACTORS = {"lo": 0.1, "hi": 10.0}


def scale_smallness(inv, n_cells, cell_volumes):
    """Scale the smallness (identity) rows of a cType=10 regularization.

    pyGIMLi's cType=10 stacks first-order smoothness rows (zWeight-weighted)
    with an identity block of weight 1. As in Oldenburg & Li (1999), where
    the smallness term is a volume integral of (m - m_ref)^2, each identity
    row is weighted by DOI_ALPHA_S * sqrt(V_i / median V). Without volume
    weighting the thousands of small, data-constrained near-surface cells
    and the few large, insensitive deep cells count the same, and no single
    weight works (0.03 left deep cells unaffected, 1 destroyed the fit).
    Call after the inversion has been set up (zero-iteration run).
    """
    w = np.array(inv.inv.cWeight(), dtype=float)
    assert len(w) == inv.fop.constraints().rows() > n_cells
    vol = np.asarray(cell_volumes, dtype=float)
    assert len(vol) == n_cells
    w[-n_cells:] *= C.DOI_ALPHA_S * np.sqrt(vol / np.median(vol))
    inv.setConstraintWeights(pg.Vector(w))
    print(f"constraints: {len(w) - n_cells} smoothness + {n_cells} "
          f"volume-weighted smallness rows (alpha_s={C.DOI_ALPHA_S}, "
          f"weights {w[-n_cells:].min():.3g} - {w[-n_cells:].max():.3g})")
    return w


def set_reference(inv, ref):
    """Reference model for the smallness term, independent of the start."""
    inv.inv.setReferenceModel(pg.Vector(ref))


def check_weights(inv, w):
    """Fail loudly if pyGIMLi replaced the constraint weights."""
    used = np.array(inv.inv.cWeight())
    assert len(used) == len(w) and np.allclose(used, w), \
        f"constraint weights not kept ({len(used)} vs {len(w)})"


def load_data():
    data = ert.load(str(C.WORK_DIR / "ssip3d.dat"))
    if TEST:
        idx = np.random.default_rng(0).choice(data.size(), 300, replace=False)
        valid = np.zeros(data.size())
        valid[idx] = 1
        data.markValid(valid > 0)
        data.markInvalid(valid == 0)
        data.removeInvalid()
    return data


def run_dc():
    pg.setThreadCount(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    mesh = pg.load(str(C.WORK_DIR / "mesh.bms"))
    pd_mesh = pg.load(str(C.RESULTS_DIR / "paraDomain.bms"))
    rho0 = float(np.median(data["rhoa"]))
    start = np.load(C.RESULTS_DIR / "res.npy")
    vol = [c.size() for c in pd_mesh.cells()]
    if TEST:
        start = np.full(pd_mesh.cellCount(), rho0)
    for tag, fac in FACTORS.items():
        t0 = time.time()
        mgr = inv3.make_manager(data, mesh)
        ref = np.full(pd_mesh.cellCount(), rho0 * fac)
        # start from the final (data-fitting) model; the reference enters
        # only through the regularization. Starting at the reference itself
        # (10x off the background) converged far too slowly.
        kw = dict(startModel=start, lam=C.LAM_DC, verbose=True)
        mgr.invert(cType=10, zWeight=C.ZWEIGHT, maxIter=0, **kw)   # set-up
        w = scale_smallness(mgr.inv, pd_mesh.cellCount(), vol)
        set_reference(mgr.inv, ref)
        # dPhi=0: the default "<2 % improvement" stop is unreliable with
        # robust reweighting (it stopped a run whose chi2 fell 37 %/iter)
        # fixed number of iterations: no dPhi stop and no stop at chi2 < 1,
        # so the model reaches the minimum of the regularized objective and
        # insensitive cells relax to the reference (stopping at chi2 < 1
        # after 3 iterations left R ~ 0.001 everywhere)
        mgr.inv.run(mgr.inv.dataVals, mgr.inv.errorVals, robustData=True,
                    dPhi=0.0, stopAtChi1=False,
                    maxIter=1 if TEST else C.DOI_MAX_ITER, **kw)
        check_weights(mgr.inv, w)
        assert mgr.paraDomain.cellCount() == pd_mesh.cellCount()
        np.save(OUT / f"doi_res_{tag}.npy", np.array(mgr.model))
        with open(OUT / "doi_log.txt", "a") as fh:
            fh.write(f"dc {tag} ref {rho0 * fac:.1f} ohmm chi2 "
                     f"{mgr.inv.chi2():.3f} history " + " ".join(
                         f"{c:.2f}" for c in mgr.inv.chi2History) +
                     f" time {(time.time() - t0) / 60:.1f} min\n")
        print(f"DOI dc {tag}: chi2={mgr.inv.chi2():.2f} "
              f"({(time.time() - t0) / 60:.1f} min)")
        del mgr


def run_ip(fk="F1"):
    pg.setThreadCount(4)
    data = load_data()
    mesh = pg.load(str(C.WORK_DIR / "mesh.bms"))
    res = np.load(C.RESULTS_DIR / "res.npy")
    mgr = inv3.make_manager(data, mesh)
    if TEST:
        mgr.invert(maxIter=0, lam=C.LAM_DC, verbose=True,
                   startModel=np.median(res))
    else:   # Jacobian of the final resistivity model, as in 03_invert.py
        mgr.invert(startModel=res, maxIter=0, lam=C.LAM_DC, verbose=True)
    mgr.fop.createJacobian(mgr.inv.model)
    pd_mesh = pg.Mesh(mgr.paraDomain)
    fop = inv3.LinearIPModelling(pd_mesh, mgr.fop.jacobian(),
                                 mgr.inv.response, mgr.inv.model)
    phi, err, ok = inv3.phase_errors(data, fk)
    phi0 = max(float(np.median(phi[ok])), 1.0)
    vol = [c.size() for c in pd_mesh.cells()]
    start = (np.full(pd_mesh.cellCount(), phi0) if TEST
             else np.load(C.RESULTS_DIR / f"phase_{fk}.npy"))
    for tag, fac in FACTORS.items():
        t0 = time.time()
        inv = pg.Inversion(fop=fop, verbose=True)
        inv.setRegularization(zWeight=C.ZWEIGHT)
        inv.dataTrans = pg.trans.Trans()
        inv.modelTrans = pg.trans.TransLogLU(0.01, 500.0)
        ref = np.full(pd_mesh.cellCount(), phi0 * fac)
        kw = dict(absoluteError=err, relativeError=0.0, startModel=start,
                  lam=C.LAM_IP)
        inv.run(phi, cType=10, maxIter=0, **kw)                     # set-up
        w = scale_smallness(inv, pd_mesh.cellCount(), vol)
        set_reference(inv, ref)
        model = inv.run(phi, robustData=True, dPhi=0.0, stopAtChi1=False,
                        maxIter=2 if TEST else C.DOI_MAX_ITER_IP, **kw)
        check_weights(inv, w)
        np.save(OUT / f"doi_phase_{fk}_{tag}.npy", np.array(model))
        resp = np.array(inv.response)
        chi2 = np.mean(((resp[ok] - phi[ok]) / err[ok]) ** 2)
        with open(OUT / "doi_log.txt", "a") as fh:
            fh.write(f"ip {fk} {tag} ref {phi0 * fac:.2f} mrad chi2 "
                     f"{chi2:.3f} time {(time.time() - t0) / 60:.1f} min\n")
        print(f"DOI ip {fk} {tag}: chi2={chi2:.2f}")


def doi_index(m_lo, m_hi, ref_lo, ref_hi):
    """Raw and normalized DOI index.

    The absolute scale of R depends on the smallness weighting, so, as in
    Oldenburg & Li (1999) and Marescot et al. (2003), the cutoff is applied
    to R normalized by its maximum. The 99.9th percentile is used as the
    maximum because single near-surface edge cells otherwise set it.
    """
    R = np.abs((np.log(m_lo) - np.log(m_hi)) /
               (np.log(ref_lo) - np.log(ref_hi)))
    return R, R / np.percentile(R, 99.9)


def summarize(name, Rn, depth):
    lines = [f"{name}: normalized DOI index, cutoff {C.DOI_CUTOFF}"]
    for a, b in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
                 (500, 600), (600, 800)]:
        s = (depth >= a) & (depth < b)
        if s.any():
            lines.append(f"  depth {a}-{b} m: median {np.median(Rn[s]):.3f}, "
                         f"{(Rn[s] < C.DOI_CUTOFF).mean() * 100:.0f}% cells "
                         f"< {C.DOI_CUTOFF}, "
                         f"{(Rn[s] < 0.1).mean() * 100:.0f}% < 0.1")
    return lines


def run_index():
    from meshing import topography
    data = load_data()
    pd_mesh = pg.load(str(C.RESULTS_DIR / "paraDomain.bms"))
    cc = np.array(pd_mesh.cellCenters())
    topo = topography(np.array(data.sensors()))
    depth = topo(cc[:, 0], cc[:, 1]) - cc[:, 2]
    s = np.array(data.sensors())
    inside = ((cc[:, 0] >= s[:, 0].min()) & (cc[:, 0] <= s[:, 0].max()) &
              (cc[:, 1] >= s[:, 1].min() - 50) &
              (cc[:, 1] <= s[:, 1].max() + 50))
    rho0 = float(np.median(data["rhoa"]))
    lo, hi = (np.load(OUT / f"doi_res_{t}.npy") for t in FACTORS)
    R, Rn = doi_index(lo, hi, rho0 * FACTORS["lo"], rho0 * FACTORS["hi"])
    np.save(OUT / "doi_res_raw.npy", R)
    np.save(OUT / "doi_res.npy", Rn)
    msg = summarize("resistivity", Rn[inside], depth[inside])
    f_lo = OUT / "doi_phase_F1_lo.npy"
    if f_lo.exists():
        _, err, ok = inv3.phase_errors(data, "F1")
        phi0 = max(float(np.median(np.array(data["ipF1"])[ok])), 1.0)
        plo, phi_ = np.load(f_lo), np.load(OUT / "doi_phase_F1_hi.npy")
        Rp, Rpn = doi_index(plo, phi_, phi0 * FACTORS["lo"],
                            phi0 * FACTORS["hi"])
        np.save(OUT / "doi_phase_F1_raw.npy", Rp)
        np.save(OUT / "doi_phase_F1.npy", Rpn)
        msg += summarize("phase F1", Rpn[inside], depth[inside])
    with open(OUT / "doi_log.txt", "a") as fh:
        fh.write("\n".join(msg) + "\n")
    print("\n".join(msg))


if __name__ == "__main__":
    {"dc": run_dc, "ip": run_ip, "index": run_index}[sys.argv[1]]()

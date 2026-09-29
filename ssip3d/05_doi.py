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


def scale_smallness(inv, n_cells):
    """Scale the smallness (identity) rows of a cType=10 regularization.

    pyGIMLi's cType=10 stacks first-order smoothness rows (zWeight-weighted)
    with an identity block of weight 1. The identity rows are scaled to
    DOI_ALPHA_S. Must be called after the inversion has been set up (a
    zero-iteration run) and before the real run.
    """
    w = np.array(inv.inv.cWeight(), dtype=float)
    assert len(w) == inv.fop.constraints().rows() > n_cells
    w[-n_cells:] *= C.DOI_ALPHA_S
    inv.setConstraintWeights(pg.Vector(w))
    print(f"constraints: {len(w) - n_cells} smoothness + {n_cells} "
          f"smallness rows (alpha_s={C.DOI_ALPHA_S})")
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
        w = scale_smallness(mgr.inv, pd_mesh.cellCount())
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
        w = scale_smallness(inv, pd_mesh.cellCount())
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
    return np.abs((np.log(m_lo) - np.log(m_hi)) /
                  (np.log(ref_lo) - np.log(ref_hi)))


def run_index():
    data = load_data()
    rho0 = float(np.median(data["rhoa"]))
    lo, hi = (np.load(OUT / f"doi_res_{t}.npy") for t in FACTORS)
    R = doi_index(lo, hi, rho0 * FACTORS["lo"], rho0 * FACTORS["hi"])
    np.save(OUT / "doi_res.npy", R)
    msg = [f"resistivity DOI: median R {np.median(R):.3f}, "
           f"{(R < C.DOI_CUTOFF).mean() * 100:.0f}% cells R < {C.DOI_CUTOFF}"]
    f_lo = OUT / "doi_phase_F1_lo.npy"
    if f_lo.exists():
        _, err, ok = inv3.phase_errors(data, "F1")
        phi0 = max(float(np.median(np.array(data["ipF1"])[ok])), 1.0)
        plo, phi_ = np.load(f_lo), np.load(OUT / "doi_phase_F1_hi.npy")
        Rp = doi_index(plo, phi_, phi0 * FACTORS["lo"], phi0 * FACTORS["hi"])
        np.save(OUT / "doi_phase_F1.npy", Rp)
        msg.append(f"phase F1 DOI: median R {np.median(Rp):.3f}, "
                   f"{(Rp < C.DOI_CUTOFF).mean() * 100:.0f}% cells "
                   f"R < {C.DOI_CUTOFF}")
    with open(OUT / "doi_log.txt", "a") as fh:
        fh.write("\n".join(msg) + "\n")
    print("\n".join(msg))


if __name__ == "__main__":
    {"dc": run_dc, "ip": run_ip, "index": run_index}[sys.argv[1]]()

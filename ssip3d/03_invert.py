"""Step 3: 3D resistivity inversion, then 3D phase (IP) inversion at F1..F4.

Resistivity: Gauss-Newton smoothness-constrained inversion of log(rhoa)
(pyGIMLi ERTManager) with robust (L1-type) data weighting.

Phase: for small phases, the apparent phase is a sensitivity-weighted average
of the intrinsic phases, phi_a = J_log . phi, with J_log = d ln(rhoa) /
d ln(rho) of the final resistivity model (Oldenburg & Li 1994; pyGIMLi
DCIPSeigelModelling). This linear problem is solved for each frequency with
the same Jacobian, so the 4 frequencies cost little extra.

Usage:  python 03_invert.py            full run
        python 03_invert.py --ip-only  phase inversions only, reusing the
                                       saved resistivity model (results/)
        python 03_invert.py --test     quick API check on 300 data
"""
import sys
import time
import numpy as np
import pygimli as pg
from pygimli.physics import ert

import config as C

TEST = "--test" in sys.argv
IP_ONLY = "--ip-only" in sys.argv


class LinearIPModelling(pg.frameworks.MeshModelling):
    """phi_a = J_log . phi with J_log = diag(1/rhoa) J diag(rho).

    Same as pyGIMLi's DCIPSeigelModelling, but on a single-region copy of the
    parameter mesh (the original has one marker per cell, which makes the
    region manager treat every cell as its own region).
    """

    def __init__(self, para_domain, jacobian, response, model):
        super().__init__()
        self._refineH2 = False            # no forward mesh needed
        mesh = pg.Mesh(para_domain)
        mesh.setCellMarkers(np.zeros(mesh.cellCount(), dtype=int))
        self.setMesh(mesh)
        self.J = pg.matrix.MultLeftRightMatrix(
            jacobian, 1.0 / pg.Vector(response), pg.Vector(model))
        self.setJacobian(self.J)

    def response(self, model):
        return self.J.dot(model)

    def createJacobian(self, model):
        pass                               # linear problem: J is fixed


def phase_errors(data, fk):
    """Absolute phase error (mrad) and a validity mask for frequency fk."""
    phi = np.array(data[f"ip{fk}"])
    rep = np.abs(np.array(data[f"iperr{fk}"]))
    if data.haveData("ipok"):
        # denoised data: IP QC already done in 06_denoise.py, and the errors
        # are already conservative estimates (with floor), so they are used
        # as they are; adding the usual floor again double-counts noise and
        # stopped the inversion after one iteration with an over-smooth model
        ok = np.isfinite(phi) & (np.array(data["ipok"]) > 0)
        err = rep.copy()
    else:
        ok = (np.isfinite(phi) & (phi <= C.MAX_ABS_PHASE)
              & (phi >= C.MIN_PHASE) & (rep <= C.MAX_PHASE_ERR))
        err = rep + C.PHASE_ERR_FLOOR + C.PHASE_ERR_REL * np.abs(phi)
    err[~ok] = 1e4                  # rejected data get ~zero weight
    return phi, err, ok


def invert_phases(out, data, para_domain, jacobian, response, model):
    """Linear phase inversion at every frequency with a fixed Jacobian."""
    fopIP = LinearIPModelling(para_domain, jacobian, response, model)
    (out / "ip_log.txt").unlink(missing_ok=True)
    for fk in C.FREQS:
        phi, err, ok = phase_errors(data, fk)
        print(f"{fk}: {ok.sum()} of {ok.size} phase data used")
        inv = pg.Inversion(fop=fopIP, verbose=True)
        inv.setRegularization(zWeight=C.ZWEIGHT)
        inv.dataTrans = pg.trans.Trans()           # linear phase data
        inv.modelTrans = pg.trans.TransLogLU(0.01, 500.0)  # 0 < phi < 500
        start = max(np.median(phi[ok]), 1.0)
        phase = inv.run(phi, absoluteError=err, relativeError=0.0,
                        lam=C.LAM_IP, robustData=True,
                        startModel=start, maxIter=2 if TEST else C.MAX_ITER_IP)
        resp = np.array(inv.response)
        rms = np.sqrt(np.mean((resp[ok] - phi[ok]) ** 2))
        chi2 = np.mean(((resp[ok] - phi[ok]) / err[ok]) ** 2)
        med = np.median(np.abs(resp[ok] - phi[ok]))
        np.save(out / f"phase_{fk}.npy", np.array(phase))
        np.save(out / f"phase_response_{fk}.npy", resp)
        np.save(out / f"phase_used_{fk}.npy", ok)
        with open(out / "ip_log.txt", "a") as fh:
            fh.write(f"{fk} ndata_used {ok.sum()} chi2 {chi2:.3f} "
                     f"rms {rms:.3f} mrad median|res| {med:.2f} mrad\n")
        print(f"{fk}: chi2={chi2:.2f} rms={rms:.2f} mrad "
              f"median|res|={med:.2f} mrad (used data)")


def make_manager(data, mesh):
    mgr = ert.ERTManager(data, verbose=True)
    # quadratic (P2) shape functions instead of the default H2 mesh
    # refinement: same accuracy, ~8x fewer cells and a fraction of the memory
    mgr.fop._refineH2 = False
    mgr.fop._refineP2 = True
    mgr.setMesh(mesh)
    return mgr


def main():
    pg.setThreadCount(4)
    out = C.RESULTS_DIR / ("test" if TEST else "")
    out.mkdir(parents=True, exist_ok=True)
    data = ert.load(str(C.WORK_DIR / "ssip3d.dat"))
    if TEST:
        idx = np.random.default_rng(0).choice(data.size(), 300, replace=False)
        valid = np.zeros(data.size())
        valid[idx] = 1
        data.markValid(valid > 0)
        data.markInvalid(valid == 0)
        data.removeInvalid()
    mesh = pg.load(str(C.WORK_DIR / "mesh.bms"))
    mgr = make_manager(data, mesh)

    if IP_ONLY:
        # rebuild the forward operator at the saved resistivity model
        res = np.load(out / "res.npy")
        t0 = time.time()
        # zero-iteration run: same operator setup as a normal inversion,
        # evaluates the response of the saved model without updating it
        mgr.invert(startModel=res, maxIter=0, lam=C.LAM_DC,
                   zWeight=C.ZWEIGHT, verbose=True)
        pd_mesh = mgr.paraDomain
        assert np.array_equal(np.array(pd_mesh.cellMarkers()),
                              np.arange(pd_mesh.cellCount())), \
            "parameter order differs from cell order"
        model = mgr.inv.model
        assert np.allclose(np.array(model), res), "model changed"
        response = mgr.inv.response
        saved = np.load(out / "rhoa_response.npy")
        print("response vs. saved: max rel. diff "
              f"{np.max(np.abs(np.array(response) / saved - 1)):.2e}")
        mgr.fop.createJacobian(model)
        print(f"Jacobian {(time.time() - t0) / 60:.1f} min")
        invert_phases(out, data, pd_mesh, mgr.fop.jacobian(), response, model)
        return

    # ---------------- resistivity ---------------------------------------
    t0 = time.time()
    mgr.invert(lam=C.LAM_DC, zWeight=C.ZWEIGHT, robustData=True,
               maxIter=1 if TEST else C.MAX_ITER_DC, verbose=True)
    print(f"DC inversion done in {(time.time() - t0) / 60:.1f} min, "
          f"chi2={mgr.inv.chi2():.2f} rrms={mgr.inv.relrms():.2f}%")
    res = np.array(mgr.model)
    pd_mesh = pg.Mesh(mgr.paraDomain)
    cov = np.array(mgr.coverage())                 # log10 already
    np.save(out / "res.npy", res)
    np.save(out / "coverage.npy", cov)
    pd_mesh.save(str(out / "paraDomain.bms"))
    np.save(out / "rhoa_response.npy", np.array(mgr.inv.response))
    np.save(out / "rhoa_data.npy", np.array(data["rhoa"]))
    with open(out / "dc_log.txt", "w") as fh:
        fh.write(f"ndata {data.size()} ncells {pd_mesh.cellCount()}\n")
        fh.write(f"chi2 {mgr.inv.chi2():.3f} rrms {mgr.inv.relrms():.3f}\n")
        fh.write("chi2 history " + " ".join(
            f"{c:.2f}" for c in mgr.inv.chi2History) + "\n")

    # Jacobian at the final model (the stored one is from the last update)
    t0 = time.time()
    mgr.fop.createJacobian(mgr.inv.model)
    print(f"final Jacobian {(time.time() - t0) / 60:.1f} min")
    invert_phases(out, data, mgr.paraDomain, mgr.fop.jacobian(),
                  mgr.inv.response, mgr.inv.model)


if __name__ == "__main__":
    main()

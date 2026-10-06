import numpy as np
import pytest

from ssip3d.forward import LogImpedanceProblem, ModelMap, Simulation
from ssip3d.mesh import build_mesh
from ssip3d.models import build_cole_cole_model, cole_cole_resistivity
from ssip3d.processing import analysis_bands, transfer_function
from ssip3d.spectral import debye_decomposition, fit_cole_cole
from ssip3d.survey import make_survey
from ssip3d.synthetic import synthesize_record
from ssip3d.waveform import PRBSWaveform, m_sequence


@pytest.fixture(scope="module")
def small():
    sv = make_survey(n_per_line=8, n_lines=2, spacing=10.0, line_spacing=20.0, n_max=3, cross_n_max=2)
    mesh = build_mesh(sv.electrodes, dx=5.0, dz=2.5, depth=20, n_pad=6, pad_factor=1.5)
    sim = Simulation(mesh, sv)
    return sv, mesh, sim


@pytest.mark.parametrize("order", [5, 7, 9, 11])
def test_m_sequence_autocorrelation(order):
    s = m_sequence(order)
    n = 2 ** order - 1
    assert len(s) == n and abs(s.sum()) == 1
    ac = np.real(np.fft.ifft(np.abs(np.fft.fft(s)) ** 2))
    assert np.isclose(ac[0], n)
    assert np.allclose(ac[1:], -1)


def test_halfspace_with_correction(small):
    sv, mesh, sim = small
    rho = 37.0
    c = sim.correction_factors(rho_ref=100.0)
    Z = sim.predict(np.full(mesh.n_cells, 1 / rho, complex)) * c
    rhoa = sv.geometric_factor() * Z.real
    assert np.allclose(rhoa, rho, rtol=1e-6)  # correction is exact for any half-space
    assert np.all(c > 0.7) and np.all(c < 1.3)


def test_halfspace_complex_phase(small):
    sv, mesh, sim = small
    sig = 0.01 * np.exp(0.02j)
    Z = sim.predict(np.full(mesh.n_cells, sig))
    ph = np.angle(Z * np.sign(Z.real))
    assert np.allclose(ph, -0.02, atol=1e-10)


def test_jacobian_finite_difference(small):
    sv, mesh, sim = small
    mm = ModelMap(mesh)
    rng = np.random.default_rng(0)
    m = np.log(0.01) + 0.3 * rng.standard_normal(mm.n_model) + 1j * 0.02 * rng.random(mm.n_model)
    Z0 = sim.predict(np.exp(mm.to_mesh(m)))
    prob = LogImpedanceProblem(sim, mm, np.sign(Z0.real))
    d, J = prob.forward(m)
    dm = (rng.standard_normal(mm.n_model) + 1j * rng.standard_normal(mm.n_model)) * 1e-3
    errs = []
    for h in (1.0, 0.5, 0.25):
        d1, _ = prob.forward(m + h * dm, need_jacobian=False)
        errs.append(np.linalg.norm(d1 - d - h * J @ dm))
    # second-order convergence of the Taylor remainder
    assert errs[1] / errs[0] < 0.3 and errs[2] / errs[1] < 0.3
    assert errs[0] < 1e-2 * np.linalg.norm(J @ dm)


def test_processing_recovers_transfer_function():
    wave = PRBSWaveform(order=7, chip_rate=64, oversampling=4)
    fh = wave.harmonics()
    Zh = 2.0 * cole_cole_resistivity(np.maximum(fh, fh[1]), 1.0, 0.2, 0.05, 0.6) / 1.0
    rng = np.random.default_rng(1)
    I, V = synthesize_record(Zh, wave, 12, rng, noise_rel=0.02, noise_abs=0.0,
                             powerline_amp=0.5, drift=1.0)
    freqs, bands = analysis_bands(wave, n_bands=6)
    Z, ea, ep, P = transfer_function(I, V, wave, bands)
    true = 2.0 * cole_cole_resistivity(freqs, 1.0, 0.2, 0.05, 0.6)
    assert P == 12
    # within ~4 standard errors (band average vs centre-frequency interpolation adds a little)
    assert np.all(np.abs(np.log(np.abs(Z / true))) < 4 * ea + 2e-3)
    assert np.all(np.abs(np.angle(Z / true)) < 4 * ep + 2e-3)


def test_debye_and_cole_cole_fit():
    f = np.geomspace(0.1, 100, 12)
    rho = cole_cole_resistivity(f, np.array([100.0, 50.0]), np.array([0.05, 0.3]),
                                np.array([0.01, 1.0]), np.array([0.5, 0.7]))
    dd = debye_decomposition(f, rho)
    assert np.allclose(dd["rho0"], [100, 50], rtol=0.05)
    assert dd["m"][1] > dd["m"][0]
    cc = fit_cole_cole(f, rho, init=dd)
    assert np.allclose(cc["m"], [0.05, 0.3], rtol=0.05)
    assert np.allclose(cc["c"], [0.5, 0.7], atol=0.03)


def test_topography_mesh_runs():
    sv = make_survey(n_per_line=8, n_lines=2, spacing=10.0, line_spacing=20.0, n_max=3, cross_n_max=2)
    el = sv.electrodes.copy()
    el[:, 2] = 0.1 * el[:, 0]  # 5.7 degree slope
    sv.electrodes = el
    mesh = build_mesh(el, dx=5.0, dz=2.5, depth=20, n_pad=6, pad_factor=1.5)
    assert 0 < mesh.active.sum() < mesh.n_cells
    sim = Simulation(mesh, sv)
    model = build_cole_cole_model(mesh, dict(rho0=100, m=0.1, tau=0.1, c=0.5))
    Z = sim.predict(model.sigma(1.0))
    assert np.all(np.isfinite(Z))
    assert np.all(np.abs(sim.electrode_xyz[:, 2] - el[:, 2]) <= mesh.hz.max())


def test_pipeline_end_to_end(tmp_path):
    """Tiny synthetic run through every stage; checks outputs and reproducibility records."""
    from ssip3d.pipeline import Pipeline, load_config

    cfg = load_config({
        "output_dir": str(tmp_path / "run"), "seed": 3,
        "survey": {"n_per_line": 8, "n_lines": 2, "spacing": 10.0, "line_spacing": 20.0, "n_max": 3,
                   "cross_n_max": 2},
        "waveform": {"order": 6, "chip_rate": 32.0, "oversampling": 4},
        "synthetic": {"enabled": True, "background": {"rho0": 100.0, "m": 0.01, "tau": 0.01, "c": 0.5},
                      "bodies": [{"type": "box", "xmin": 25, "xmax": 45, "ymin": 0, "ymax": 20,
                                  "zmin": -12, "zmax": -4, "rho0": 30.0, "m": 0.2, "tau": 0.3, "c": 0.6}],
                      "n_model_freqs": 6, "n_periods": 4},
        "processing": {"n_bands": 3},
        "mesh": {"dx": 5.0, "dz": 2.5, "depth": 15, "n_pad": 5, "pad_factor": 1.5},
        "inversion": {"max_iter": 4},
        "spectral": {"cole_cole": False},
    })
    pl = Pipeline(cfg)
    pl.run()
    out = tmp_path / "run"
    for f in ("spectra.csv", "inversion.npz", "spectral.npz", "model.vtk", "model.csv", "provenance.json",
              "config_used.yaml", "figures/model_sections.png"):
        assert (out / f).exists(), f
    d = np.load(out / "inversion.npz")
    sp_ = np.load(out / "spectral.npz")
    # chargeable body is recovered: max chargeability well above background
    assert np.nanmax(sp_["dd_m"]) > 0.05
    # final misfits reasonable
    import json
    hist = json.loads((out / "inversion_history.json").read_text())
    last = [h[-1] for h in hist.values()]
    assert all(x["chi2_amp"] < 3 and x["chi2_phase"] < 3 for x in last)
    assert d["m"].shape[0] == 3

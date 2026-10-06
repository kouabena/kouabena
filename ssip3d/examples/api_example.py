"""Using ssip3d from Python: forward model, processing and inversion in a few lines."""
import numpy as np

from ssip3d import (InversionOptions, LogImpedanceProblem, ModelMap, PRBSWaveform, Simulation,
                    build_cole_cole_model, build_mesh, invert_multifrequency, make_survey)
from ssip3d.processing import analysis_bands
from ssip3d.synthetic import simulate_and_process, simulate_spectra

survey = make_survey(n_per_line=12, n_lines=3, spacing=10.0, line_spacing=20.0, n_max=5, cross_n_max=2)
mesh = build_mesh(survey.electrodes, dx=5.0, dz=2.5, depth=30, n_pad=6, pad_factor=1.5)
sim = Simulation(mesh, survey)
true = build_cole_cole_model(mesh, dict(rho0=100, m=0.01, tau=0.01, c=0.5),
                             [dict(type="box", xmin=40, xmax=70, ymin=10, ymax=30, zmin=-20, zmax=-6,
                                   rho0=30, m=0.2, tau=0.5, c=0.6)])

# synthetic m-sequence records -> spectra with empirical errors
wave = PRBSWaveform(order=7, chip_rate=32, oversampling=4)
f_model = np.geomspace(wave.f0, wave.fs / 2, 9)
corr = sim.correction_factors(100.0)
Zm = simulate_spectra(sim, true, f_model, correction=corr)
freqs, bands = analysis_bands(wave, n_bands=5)
spec, _, _ = simulate_and_process(Zm, f_model, wave, bands, n_periods=8, seed=0)
spec = spec.apply_error_floor(0.01, 1e-3)

# multi-frequency 3D inversion
mmap = ModelMap(mesh)
pol = np.sign(spec.Z[0].real)
prob = LogImpedanceProblem(sim, mmap, pol, corr)
d_obs = np.log(pol * spec.Z)
m0 = np.full(mmap.n_model, np.log(1 / 100.0) + 0j)
res = invert_multifrequency(prob, spec.freqs, d_obs, spec.err_amp, spec.err_phase, m0,
                            InversionOptions(max_iter=6))
for f, r in zip(spec.freqs, res):
    print(f"{f:7.3f} Hz  chi2 amp {r.history[-1]['chi2_amp']:.2f}  phase {r.history[-1]['chi2_phase']:.2f}  "
          f"max phase {r.m.imag.max() * 1e3:.1f} mrad")

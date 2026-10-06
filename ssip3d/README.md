# ssip3d — 3D inversion of Spread-Spectrum Induced Polarization data

`ssip3d` is an open, reproducible Python workflow for **spread-spectrum induced polarization (SSIP)**.
It takes raw m-sequence current/voltage records and produces 3D images of resistivity, chargeability
and relaxation time:

```
raw records ──► spectra + empirical errors ──► 3D complex-conductivity inversion (all frequencies)
            ──► per-cell Debye / Cole-Cole parameters ──► VTK / CSV / figures + provenance
```

What it does:

* **Spread-spectrum processing.** m-sequence deconvolution, robust period stacking in complex-log
  space, powerline notching, and *data-driven* standard errors for amplitude and phase separately.
* **3D finite-volume forward model** with complex conductivity, topography (air cells),
  remote electrodes, and numerical geometric-factor correction of the source singularity.
* **Adjoint Gauss-Newton inversion of ln σ\*.** Amplitude and phase have *independent,
  automatically tuned* regularization: each is driven to its own χ² = N.
  Sensitivity weighting is included, and frequency continuation makes the extra frequencies cheap.
* **Spectral interpretation.** Debye decomposition (total and normalized chargeability, mean τ) and
  a Cole-Cole fit (ρ₀, m, τ, c) in every cell.
* **Reproducibility.** One YAML file per experiment, explicit defaults, fixed seeds, and
  `provenance.json` (config SHA-256, git commit, library versions).
* **Verification.** Unit tests cover the m-sequence autocorrelation, an analytic half-space,
  an adjoint Jacobian Taylor test, processing accuracy, spectral fits, topography, and the
  full pipeline.

See **[docs/methodology.md](docs/methodology.md)** for the equations (ready to adapt for a paper).

## Install

```bash
cd ssip3d
pip install -e .            # needs numpy, scipy, matplotlib, pyyaml
pytest                      # optional: run the test-suite
```

## Quick start

```bash
ssip3d example my_examples                       # copy the example configs
ssip3d run my_examples/quick_test.yaml           # ~2 min synthetic test
ssip3d run my_examples/synthetic_porphyry.yaml   # porphyry Cu-Mo demo (tens of minutes)
```

Each stage can also be run on its own: `ssip3d simulate|process|invert|spectral|report config.yaml`.

Outputs, in `output_dir`:

| file | content |
|---|---|
| `config_used.yaml`, `provenance.json`, `ssip3d.log` | exact settings, environment, log |
| `spectra.csv` / `.npz` | processed data: ABMN, frequency, \|Z\|, phase, errors |
| `inversion.npz`, `inversion_history.json` | ln σ\* per frequency, predicted data, convergence |
| `spectral.npz` | Debye (`dd_*`) and Cole-Cole (`cc_*`) parameters per cell |
| `model.vtk` | all models on the core mesh (open in ParaView) |
| `model.csv` | x, y, z, every field (for GIS / Leapfrog / Python) |
| `figures/` | survey, raw records + step response, data fit, convergence, model sections |

## Using your field data

There are two entry points. Set them in the YAML (`synthetic.enabled: false`):

**A. Processed spectra.** Give the electrodes, the ABMN list and a spectra CSV:

```yaml
survey: {electrodes_csv: electrodes.csv, abmn_csv: abmn.csv}   # x,y,z  /  a,b,m,n (0-based, -1 = remote)
data:   {spectra_csv: spectra.csv}  # a,b,m,n,freq_hz,amp_ohm,phase_mrad,err_lnamp,err_phase_mrad
```

`amp_ohm` is |V/I| and `phase_mrad` is the impedance phase (negative for polarizable ground). The
sign of each impedance is taken from the geometric factor.

**B. Raw m-sequence records.** Give an `.npz` file with `current` (n_tx × n_samples, A),
`voltage` (n_data × n_samples, V) and `tx_index` (n_data,: which current record goes with each
datum). The records must start at the beginning of an m-sequence period. Describe your
transmitter in `waveform:` (order, chip_rate, oversampling = samples per chip). Then run
`ssip3d run config.yaml` and the `process` stage is included automatically.

Everything is also available from Python:

```python
from ssip3d import *
survey = make_survey(n_per_line=24, n_lines=4, spacing=25, line_spacing=50)
mesh = build_mesh(survey.electrodes, dx=12.5, depth=100)
sim = Simulation(mesh, survey)
model = build_cole_cole_model(mesh, dict(rho0=300, m=0.02, tau=0.01, c=0.5),
                              [dict(type="sphere", center=[300, 75, -40], radius=30, rho0=80, m=0.25, tau=1, c=0.5)])
Z = sim.predict(model.sigma(1.0))     # complex transfer impedances at 1 Hz
```

See `examples/api_example.py`.

## Tips

* Core cell size `dx` = a/2 is a good default; `dz` = dx/2.
* The usable band of the m-sequence is ≈ 1/T … 0.74·chip_rate. Choose `order` and `chip_rate` so
  that it covers the relaxation times you expect (τ ≈ 1/(2πf)).
* Check `figures/convergence.png`: both χ² curves should reach ≈ 1. If they don't, raise the error
  floors.
* Look at the `coverage` field before interpreting deep structure.

## Citation

If you use `ssip3d`, please cite it (see `CITATION.cff`) and give the version and the
`config_sha256` from `provenance.json` so others can reproduce your results.

MIT License.

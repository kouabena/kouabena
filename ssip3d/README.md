# SSIP 3D resistivity and phase (IP) inversion with pyGIMLi

3D inversion of the SSIP spectral IP survey (lines L22, L24, L26; pole-dipole,
4 frequencies) for the Cu-Mo project. It gives one 3D resistivity model and
one 3D phase (IP) model per frequency.

## Contents

| File | Purpose |
|---|---|
| `config.py` | survey geometry, data selection, mesh and inversion settings (edit this, not the scripts) |
| `ssip_io.py` | reader for the RES2DINV *General array with IP* files |
| `01_prepare_data.py` | merge the 3 lines × 4 frequencies, QC, select data, write `work/ssip3d.dat` |
| `02_make_mesh.py`, `meshing.py` | 3D tetrahedral mesh draped on topography |
| `03_invert.py` | resistivity inversion, then phase inversion at F1–F4 |
| `flag_outliers.py`, `outliers_pass1.csv` | data misfit by > 5σ in pass 1, removed in pass 2 |
| `04_export_results.py` | VTK/CSV export and figures |
| `work/ssip3d.dat` | the selected 3D dataset (pyGIMLi unified data format) |
| `results/` | models, figures, logs (see *Results*) |

## Data

| | L22 | L24 | L26 |
|---|---|---|---|
| Stored value | app. resistivity | resistance | resistance |
| Data per frequency | 57 057 | 62 989 | 45 730 |
| Current electrodes (C1) | 52, x = −640 … 3060 m | 52, x = −640 … 3030 m | 52, x = −640 … 3010 m |
| Receiver electrodes | 58, 0 … 2280 m | 61, 0 … 2400 m | 60, 0 … 2400 m |

* Array: pole-dipole. C2 is remote (`1.23456789E+8` in the `.z2d` files).
  Every receiver pair was recorded for every injection (distributed
  acquisition), with dipoles from 40 m to 2280 m long.
* Frequencies: F1 = 0.15625, F2 = 0.40625, F3 = 0.65625, F4 = 0.90625 Hz.
  Each file holds the amplitude, its error, the phase (mrad) and the phase error.
* The injection points lie on the lines themselves: their elevations match the
  receiver topography. There are no cross-line measurements.
* The median apparent resistivity is the same on all three lines (356, 336,
  364 Ω·m for the selected data). This confirms that the L22 apparent
  resistivities and the L24/L26 resistances were converted consistently.

## Method

### 1. Survey geometry (assumption; please verify)
The files contain only chainage along each line, so the layout is set in
`config.LINES`: the lines are straight and parallel, **200 m apart**
(L22 at y = 0, L24 at y = 200 m, L26 at y = 400 m), all start at chainage 0,
and all run in the same direction. If the true layout is different (other
spacing, bearing, or line origins), put the real coordinates there and rerun
all four steps. See *Georeferencing* below.

### 2. Data preparation (`01_prepare_data.py`)
* All values are converted to transfer resistance |R|. For L22 this uses
  ρa / k, with k the half-space pole-dipole factor from the true electrode
  positions (topography included). The sign of R follows the geometric factor.
  In the inversion k cancels (ρa,obs / ρa,pred = R_obs / R_pred), so its
  choice does not affect the model.
* QC removes: dipoles straddling the current electrode, zero values, data
  whose reported error is above 10 %, four unreliable off-end transmitters
  (`config.EXCLUDE_TX`), and the pass-1 outliers (see *Results*).
* **Removing redundancy.** With all-pairs recording, every dipole of one
  injection is a sum of adjacent 40 m dipoles, so 166k values per frequency
  contain far less independent information than their number suggests. They
  would also need an ~80 GB Jacobian. For every injection and every receiver
  electrode P, one dipole is kept: P → P + L, pointing away from the
  current electrode, with L = 40·2^j m the shortest length ≥ offset / 8 (so
  n ≤ 8, and L ≤ 320 m). This gives short dipoles near the source and long
  dipoles (better signal) far from it, and keeps full lateral and offset
  coverage. Offsets start at 60 m. Final dataset: **7 819 data** (L22 2 404,
  L24 2 800, L26 2 615), each with ρa and 4 phases.
* Error model: resistivity max(reported, 5 %). Phase: reported + 1 mrad + 5 %
  of |φ|. Phases outside −20 … 150 mrad or with a reported error > 10 mrad
  are excluded.

### 3. Mesh (`02_make_mesh.py`)
* Unstructured tetrahedra (TetGen). The parameter domain extends 100 m beyond
  the electrodes and 700 m deep. A large boundary region (×4) simulates a
  half-space.
* Extra surface nodes at ±10 m around every electrode, plus a node 15 m below
  each, for accurate potentials at the short (20 m) Tx–Rx offsets.
* The mesh is generated flat and then **draped onto the topography**: every
  node moves up by topo(x, y), with a weight decaying linearly to zero at the
  model bottom. Topography is interpolated linearly along the lines and
  between them. Electrode nodes sit exactly at the surveyed elevations. Relief
  across the area is 41 – 205 m a.s.l.
* Forward solutions use quadratic (P2) elements.
* **Accuracy check.** A homogeneous half-space was modelled on a flat mesh of
  the same design (total field, no singularity removal) and compared with the
  analytic solution. Median error is 3 %, 95th percentile 6.3 %; the largest
  errors are at 20 m offsets (≈ −9 %), which are excluded in the final
  pass. The 5 % error floor accounts for the rest.
  With topography the homogeneous response varies by ±20–40 %. This is a real
  topographic effect, and the 3D inversion models it.
* 50 717 parameter cells (final mesh).

### 4. Inversion (`03_invert.py`)
* **Resistivity:** Gauss–Newton inversion of log ρa for log ρ with first-order
  smoothness (λ = 20, vertical/horizontal weight 0.3) and robust (IRLS, L1-type)
  data weighting against outliers. Starting model: homogeneous, at the median
  ρa. At most 8 iterations (stops at χ² < 1).
* **Phase (IP), F1–F4:** for small phases, the apparent phase is the
  sensitivity-weighted average of the intrinsic phases:
  φa = J_log · φ, with J_log = ∂ ln ρa / ∂ ln ρ at the final resistivity model
  (Oldenburg & Li 1994, *Geophysics* 59, 1327; as in pyGIMLi's
  `DCIPSeigelModelling`). This linear problem is solved for each frequency
  with the same Jacobian and the same regularization (λ = 30), using a log
  model transform that keeps phases positive (0 < φ < 500 mrad) and robust
  data weighting.
* The resistivity model is from F1 (0.156 Hz, closest to DC). Median
  amplitudes drop only 1.3 % (F2), 1.9 % (F3) and 2.2 % (F4) relative to F1,
  i.e. below the 5 % error floor, so per-frequency resistivity models would
  differ mainly by noise. Median apparent phases are 17.4 / 18.5 / 18.3 /
  17.4 mrad at F1–F4.

### 5. Outputs (`04_export_results.py`)
* `results/ssip3d_model.vtk`: open in **ParaView** (threshold on `resolved`,
  slice, contour). Cell data: `resistivity`, `log10_resistivity`,
  `phase_F1_mrad` … `phase_F4_mrad`, `coverage_log10`, `resolved`, `depth`.
* `results/ssip3d_model_cells.csv`: cell centres (x, y, z, depth) with all
  properties, for Voxler, Leapfrog, Oasis montaj, Surfer, etc.
* Figures: depth slices, sections along and between the lines, phase at the 4
  frequencies, data fit.
* `resolved` = coverage (sum of absolute log sensitivities per cell volume)
  above its 25th percentile. The figures blank out cells below it. This is a
  practical depth-of-investigation guide, not a formal DOI.

## Results

### Processing history (two passes)
1. **Pass 1:** 8 640 data with 20 m minimum offset and all transmitters.
   It converged to a robust-weighted χ² = 1.1, but 8 % of the data (708) were
   misfit by > 5σ. The misfits clustered at offsets ≤ 100 m (29 % of
   those data; forward-modelling error and near-surface heterogeneity below
   the cell size) and on four off-end transmitters whose positions are
   uncertain: L22 −640 m and −160 m, L24 −160 m (100 % misfit), L26 −185 m.
2. **Pass 2 (final):** those four transmitters removed, minimum offset 60 m,
   pass-1 outliers removed (`outliers_pass1.csv`), which leaves **7 819 data**.
   Phases below −20 mrad (EM coupling or noise, increasing with frequency)
   are also excluded from the IP inversion.

### Data fit (pass 2)
| | data used | χ² (robust) | χ² (plain) | fit |
|---|---|---|---|---|
| Resistivity | 7 819 | 0.88 | 3.95 | rel. RMS 10.8 % |
| Phase F1 0.156 Hz | 6 436 | – | 4.25 | median abs. residual 3.5 mrad, RMS 8.4 mrad |
| Phase F2 0.406 Hz | 6 456 | – | 4.49 | 2.5 mrad, RMS 9.2 mrad |
| Phase F3 0.656 Hz | 6 026 | – | 4.66 | 2.7 mrad, RMS 8.6 mrad |
| Phase F4 0.906 Hz | 6 384 | – | 4.65 | 2.9 mrad, RMS 8.2 mrad |

The resistivity χ² history is 107 → 69 → 14.6 → 6.8 → 2.9 → 1.5 → 0.88
(stopped at χ² < 1). The IP misfit is mostly a long tail of noisy phases,
which robust weighting down-weights. The bulk fits within about 3 mrad
(`fig_data_fit.png`).

### Models
* Resistivity 30 – 3 150 Ω·m, median 257 Ω·m. Intrinsic phase (5 / 50 /
  95 %): about 8 / 13 / 33 mrad at all four frequencies.
* **Resistive core** (> 800 Ω·m) under the ridge at x ≈ 1300 – 1800 m, from
  about 50 m below surface to the base of resolution (≈ −350 m a.s.l.). It
  is continuous across all three lines and strongest on L24–L26.
* **Conductive zone** (≈ 60 – 150 Ω·m) at x ≈ 2000 – 2300 m, from about
  −50 to −300 m a.s.l., on L22 and L24 and weaker on L26. Low-resistivity
  cover (100 – 200 Ω·m) lies over the western half (x < 1000 m).
* **Chargeable zone** (phase > 40 – 50 mrad) at depth under x ≈ 700 – 1500 m.
  Its top rises from about −250 m a.s.l. on L22 to about −100 m on
  L24–L26 (i.e. it shallows northwards / towards L26). It sits on the
  western flank and below the resistive core. A second chargeable zone lies
  at depth under x ≈ 2000 – 2400 m, beneath the conductive zone. The
  resistive, weakly chargeable core flanked by a chargeable halo is the
  pattern expected around a porphyry system (silicified / potassic core,
  pyrite-rich phyllic shell). This is an interpretation to test against
  geology and drilling.
* The phase models at F1–F4 are very similar (`fig_phase_spectrum.png`).
  Over 0.16 – 0.9 Hz the phase spectrum is nearly flat, so this band does
  little to discriminate grain size or mineralogy. The ratios between
  frequencies are best examined in the exported models, not the sections.

### Caveats specific to these results
* The deepest high-phase values sit at the bottom edge of the resolved
  volume, where the log-transformed phase inversion can overshoot. Their
  depth extent and amplitude are uncertain; their top and lateral position
  are better constrained. A DOI test (two inversions with different
  reference models) is recommended before interpreting depth extent.
* The between-line sections (y = 100 m, 300 m) are interpolation constrained
  by smoothness, not by cross-line data.

## Reproducing

```bash
pip install -r requirements.txt        # plus the TetGen executable on PATH
# unzip "SSIP DATA INVERSION" and link or copy it to ssip3d/raw
ln -s "/path/to/SSIP DATA INVERSION" raw
python 01_prepare_data.py   # seconds
python 02_make_mesh.py      # < 1 min
python flag_outliers.py      # only when re-doing pass 1 -> pass 2
python 03_invert.py         # ≈ 30 min DC + ≈ 65 min IP on 4 cores, ≈ 7 GB RAM
python 03_invert.py --ip-only   # redo only the phase inversions
python 04_export_results.py
```

## Georeferencing
To put real coordinates in the model, either:
1. change `config.LINES` if only the spacing or order is different, or
2. transform the local (x, y) to UTM in `01_prepare_data.py` right after the
   `c`, `m`, `n` arrays are built (rotation by the line bearing plus the
   origin of L22). Keep x, y in a local frame for meshing, then export
   UTM in step 4.

## Limitations to state in the thesis
* **Three parallel 2D lines, no cross-line data.** Resolution between the
  lines comes from the 3D sensitivity of each line's own data and from the
  smoothness constraint. Features between the lines (y = 100 m, 300 m) are
  interpolated, and anything outside y ≈ −100 … 500 m is unconstrained. A
  3D inversion still beats 2D here because it handles 3D topography and
  off-line structure (no 2D assumption). Results between lines should be read
  as a smooth interpolation. If the raw SSIP files hold recordings of each
  injection on the other lines' receivers, adding them would give true
  cross-line coverage and a much better 3D model.
* Line layout assumed (see *Survey geometry*).
* The linearized phase inversion assumes small phases (≪ 1 rad), which holds
  here (median ≈ 17 mrad).

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
| `05_doi.py` | depth-of-investigation test (Oldenburg & Li) |
| `06_denoise.py` | denoising of all dipoles: potential reconstruction, bad-injection detection, error estimates, phase QC |
| `08_prepare_denoised.py`, `09_compare_variants.py`, `results_denoised/` | pyGIMLi inversion of exactly the ZondRes3D dataset (final model) and comparison with the original |
| `07_export_zond.py`, `zond/` | denoised data as ZondRes3D input (Res2DInv + .z2d + 3D CSV); run guide in `zond/README_ZondRes3D.md` |
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
* `results/ssip3d_model.vtk`: open in **ParaView** (threshold on
  `doi_res_index` / `doi_phase_F1_index` < 0.2, slice, contour). Cell data:
  `resistivity`, `log10_resistivity`, `phase_F1_mrad` … `phase_F4_mrad`,
  `doi_res_index`, `doi_phase_F1_index`, `coverage_log10`,
  `resolved_coverage`, `depth`.
* `results/ssip3d_model_cells.csv`: cell centres (x, y, z, depth) with all
  properties, for Voxler, Leapfrog, Oasis montaj, Surfer, etc.
* Figures: depth slices, sections along and between the lines, phase at the 4
  frequencies, DOI index sections, data fit. Resistivity is blanked where
  its normalized DOI index ≥ 0.2, and all phase figures are blanked where the
  F1 phase DOI index ≥ 0.2 (the F1 DOI is used for all four frequencies).
* `resolved_coverage` is the earlier, simpler guide: coverage within 2.25
  decades of the near-surface coverage. It is kept for comparison only.

### 6. Depth of investigation (`05_doi.py`)
DOI index after Oldenburg & Li (1999), for resistivity and for the F1 phase:
* Each property is inverted twice more, with homogeneous reference models at
  0.1× and 10× the background (median ρa = 392 Ω·m → 39 / 3 920 Ω·m; median
  phase 18.8 mrad → 1.88 / 188 mrad). The regularization is first-order
  smoothness plus a smallness term α_s·√(V_i / V_median)·(m − m_ref), with
  α_s = 0.03 (pyGIMLi `cType = 10` with rescaled weights). The volume
  weighting is the discrete form of Oldenburg & Li's volume integral.
* R = |ln m₁ − ln m₂| / |ln m_ref1 − ln m_ref2|, normalized by its 99.9th
  percentile (Oldenburg & Li 1999; Marescot et al. 2003, *Geophys. Prosp.*).
  Cells with R̂ < 0.2 are treated as data-controlled (0.1 is the stricter
  bound).
* Both DOI runs start from the final model and iterate until the total
  objective stops decreasing. Data fits (plain χ²): resistivity 5.3 and
  8.0 (main model 3.95); phase F1 11.0 and 10.7 (main model 4.25). The pairs
  fit comparably, so R reflects the data, not a difference in fit.
* Settings that were tried and rejected (see git history):
  - Uniform smallness weights. 0.01 and 0.03 left deep cells tied to their
    neighbours by smoothness (R ≈ 0.01 even at 700 m depth); 1.0 overrode
    the data (χ² = 119).
  - Starting from the reference itself (10× from the background).
    Convergence was too slow.

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
* **Depth of investigation (DOI test):**

  | Depth below surface | Resistivity median R̂ | Phase F1 median R̂ |
  |---|---|---|
  | 0 – 100 m | 0.08 | 0.22 |
  | 100 – 200 m | 0.08 | 0.26 |
  | 200 – 300 m | 0.13 | 0.46 |
  | 300 – 400 m | 0.14 | 0.65 |
  | 400 – 500 m | 0.22 | 0.71 |
  | 500 – 800 m | 0.43 – 0.46 | 0.78 – 0.80 |

  The resistivity model is data-controlled to about **400 – 450 m** below
  surface. The phase model only to about **150 – 250 m**, and only between
  x ≈ 300 and 2150 m (`fig_doi_sections.png`), because the phase data are
  much noisier (≈ 20 % relative error, against 5 % for resistivity).
* **Resistive core** (> 800 Ω·m) under the ridge at x ≈ 1300 – 1800 m, from
  about 50 m below surface to about −350 m a.s.l. It is resolved, continuous
  across all three lines, and strongest on L24–L26.
* **Conductive zone** (≈ 60 – 150 Ω·m) at x ≈ 2000 – 2300 m on L22 and L24,
  weaker on L26. Its top (about −50 m a.s.l.) is resolved. Its centre below
  about −150 m has R̂ > 0.2, so its depth extent is not constrained. Poor
  resolution beneath a conductor is expected, since the current channels
  through it. Low-resistivity cover (100 – 200 Ω·m) lies over the western
  half (x < 1000 m).
* **Chargeable zone** (phase > 30 mrad in the resolved volume) at
  x ≈ 1100 – 1300 m. It is strongest on L22 and between L22 and L24, with
  its top at about 0 to −50 m a.s.l. (≈ 150 m below surface). It is weaker
  on L24 and fades on L26. The higher phases below it (up to > 50 mrad at
  200 – 450 m depth) lie where the phase DOI index is 0.5 – 0.7, i.e. **not
  resolved**. Its depth extent, amplitude and any northward trend are not
  constrained by these data. A second, resolved anomaly (20 – 30 mrad) sits
  at x ≈ 1900 – 2150 m on L24, at −50 to −200 m a.s.l., next to the conductive
  zone.
* **Porphyry interpretation.** A resistive, weakly chargeable core with a
  chargeable zone on its western flank fits a porphyry pattern (silicified /
  potassic core, pyrite-rich phyllic shell). Only the top of the chargeable
  zone is resolved. Its extent at depth would need deeper-reaching IP
  (longer offsets, more stacking or lower noise) or drilling to confirm.
* The phase models at F1–F4 are very similar (`fig_phase_spectrum.png`).
  Over 0.16 – 0.9 Hz the phase spectrum is nearly flat, so this band does
  little to discriminate grain size or mineralogy. The ratios between
  frequencies are best examined in the exported models, not the sections.

### Caveats specific to these results
* The phase model is resolved only to ≈ 150 – 250 m below surface (DOI
  test). Deeper phase values are shaped by the regularization, not the
  data.
* The DOI index depends on the regularization choices (α_s, volume
  weighting, the 99.9th-percentile normalization). The 0.2 cutoff is a
  common convention, not a physical threshold. The depth trend and the
  resistivity/phase contrast are robust; exact contour depths are
  approximate.
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
python 05_doi.py dc && python 05_doi.py ip && python 05_doi.py index
                            # DOI test: ≈ 35 min + ≈ 50 min
python 04_export_results.py
```

## Denoising and ZondRes3D export
```bash
python 06_denoise.py        # < 1 min, all 166k dipoles x 4 frequencies
python 07_export_zond.py    # writes zond/
```
The method, results and the ZondRes3D run guide are in `zond/README_ZondRes3D.md`. The QC figure is `results/denoise/fig_denoise_qc.png`. The pyGIMLi results above use the earlier (not denoised) selection. The denoised set differs mainly on L22: resistance converted with horizontal-distance geometric factors, and 7 inconsistent off-end injections removed.

## Harmonization with the original ZondRes2D files and the final model
The `.z2d` files of the survey were audited against the `.dat` exports.
Details are in `zond/README_ZondRes3D.md`.
* Values and topography are identical.
* ZondRes2D dropped the last 3 rows of each line. They are dropped here too.
* ZondRes2D converts phase to eta_a as 100·tan(φ). The export uses the same
  formula.
* The user had re-weighted 935 L22 F1 data in ZondRes2D (weights
  0.11–0.50), and 852 of them are on the off-end injections the consistency
  test rejects. The user weights are carried into both the ZondRes3D files
  and pyGIMLi.

**Final pyGIMLi model (`results_denoised/`)** = inversion of exactly the
data given to ZondRes3D:
- **Data:** 7 639 resistance and 6 343 IP data, after the pass-1 misfit
  outliers are removed.
- **Fit:** resistivity plain χ² 2.96, rel. RMS 10.4 % (original run: 3.95,
  10.8 %). Phase median residual 3.4–4.6 mrad at F1–F4.
- **Agreement with the original model** in the resolved volume
  (`results_denoised/compare_variants.txt`, `fig_compare_variants.png`):
  - log-resistivity correlation 0.984, median difference 5.8 %,
  - phase correlation 0.89–0.92, median difference about 1 mrad.
  The interpretation above therefore holds for the denoised data.
* Reproduce:
  ```bash
  python 06_denoise.py && python 07_export_zond.py && python 08_prepare_denoised.py
  SSIP_VARIANT=denoised python 02_make_mesh.py
  SSIP_VARIANT=denoised python 03_invert.py
  python 09_compare_variants.py
  SSIP_VARIANT=denoised python 04_export_results.py
  ```
* The DOI index is the one from the original run, mapped onto the new mesh
  (same survey geometry).

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

# Running the denoised SSIP data in ZondRes3D

The files here come from `06_denoise.py` and `07_export_zond.py`. Menu
names, file-format rules and parameter ranges follow the ZondRes3D user
manual (English edition; page numbers below refer to it).

## Files

| File | Use |
|---|---|
| **`ssip3d_denoised.z3d`** | **Recommended.** Native ZondRes3D data file with all three lines: 7 987 measurements, resistance plus 4-frequency IP (see format below) |
| `L22_res.z2d`, `L24_res.z2d`, `L26_res.z2d` | ZondRes2D files for *Collect from 2D* (resistance only, all data passing DC QC) |
| `L*_F1.z2d` … `L*_F4.z2d` | ZondRes2D files with IP (eta_a = phase/10, as in your original .z2d projects), same geometry in F1–F4 |
| `L*_res.dat`, `L*_F*.dat` | the same data in Res2DInv general-array format (backup, other software) |
| `ssip3d_denoised_3d.csv` | all data with x, y, z of A, M, N, 3D geometric factor, ρa, errors (other software) |
| `export_summary.txt` | data counts and x range of each line |

### Format of `ssip3d_denoised.z3d` (manual pp. 28–32)
```
time_#chann 0.15625 0.40625 0.65625 0.90625
prof c1x p1x p2x c1y p1y p2y c1z p1z p2z weight res mod1..mod4 pha1..pha4 weightip
0 -20.000 40.000 80.000 0.000 0.000 0.000 0 0 0 1.0000 9.49583e-02 ... -5.88 ... 0.5295
...
Topo
x y elevation   (one row per electrode position)
```
- `prof` identifies the line: 0 = L22 (y = 0 m), 1 = L24 (y = 200 m),
  2 = L26 (y = 400 m).
- The array is pole-dipole. C2 is remote, so its columns are omitted. Only
  C1, P1 and P2 are given.
- `res` is the resistance |V/I| in Ω. The manual recommends resistance
  whenever topography is present.
- Electrodes are on the surface (`z = 0`). Their elevations are in the
  `Topo` block.
- `mod1–4` are the resistances at F1–F4. `pha1–4` are the phases in
  **mrad, written negative** (Zond convention: "phase shift … always
  negative", p. 33). A field phase of +18 mrad is written as −18.
- `weight` = min(1, 3 % / relative error of R), and `weightip` =
  min(1, 3 mrad / phase error), both from the denoising error estimates.
  `weightip = 0` marks the 1 328 rows whose phase failed QC.
- **Check after opening:** in the polarizability mode the typical apparent
  phase must be about 15–20 mrad. If ZondRes3D shows it as −18 or treats it
  as radians, open the file in a text editor and flip the sign of `pha1–4`.
  Otherwise, use the .z2d route below, which uses the same convention as
  your earlier projects.

## Line positions (assumption: please check)
Lines parallel, 200 m apart, same chainage origin: L22 at y = 0, L24 at
y = 200, L26 at y = 400 m, all running in +x. If you have GPS/UTM
coordinates, replace `y` and `x` in the .z3d file (or enter the real ends
in *Collect from 2D*).

## Step by step

### 1. Open the data
- **Option A (recommended):** *File / Open file*, type *ZondRes3D files*,
  then `ssip3d_denoised.z3d`.
- **Option B:**
  1. *File / Open file*, type *ZondRes2D files*, and select the three
     `L*_res.z2d` (or `L*_F1.z2d`) files.
  2. In the *Multi-files case* dialog choose *Collect from 2D*.
  3. Enter the start and end XY of each line. The x range of each file is
     in `export_summary.txt`: L22 (−20, 0) → (2747, 0); L24 (−640, 200) →
     (3030, 200); L26 (−640, 400) → (3010, 400).
  4. For more IP channels, use *Collect IP* with the `L*_F2–F4.z2d` files.
  5. The .z2d files carry C2 at 1.23456789E+8 m. Apply *Options / Extra /
     Remote to infinite* so C2 is treated as a true infinite electrode.

In the *Line editor* (opens automatically), check that the three lines are
200 m apart, then press the button to proceed to the inversion mode.

### 2. Mesh (*Options / Mesh constructor*, pp. 36–39)
- Mesh type: **Regular**. Axes angles 0/90, orthogonal.
- Divide type: **Automatic**. Extra div **X = 1** (20 m cells between the
  40 m electrodes), **Y = 3** (50 m cells between lines 200 m apart). Never
  use cells as wide as the line spacing.
- Z-axis:
  - Start height **10 m** (about the horizontal cell size for the shallow
    part).
  - Increment **1.12**.
  - Div number **20**, which gives a model about 700 m deep. Deeper is
    pointless: the DOI test found resistivity resolved to about 450 m and
    phase to about 250 m.
  - Topo coeff **1** (topography flattens with depth).
- Use the "check electrodes in the same cell" button. A current and a
  potential electrode in the same cell must be avoided.
- **Cell grouping** (*Program setup / Options*, p. 61): merge 2–4 cells in
  X and Y from about layer 8 downwards. The manual advises roughly as many
  inversion parameters as data, here about 8 000–15 000.

### 3. Data QC in ZondRes3D (optional, *Options / Quality control module*, p. 47)
The data are already cleaned (see *Denoising* below). Use the module only to
inspect them, not to re-weight: the weights are already set from the error
estimates.

### 4. Resistivity inversion (*Option / Program setup*, pp. 55–63)
- **Solver tab:**
  - Solver type *Direct*.
  - Calculation scheme **Secondary**. Your Tx–Rx offsets start at 60 m, and
    the manual warns that *Total* gives large errors near current
    electrodes.
  - BC *Mixed*.
- **Inversion tab, first run:**
  - Inversion **Smoothness constrained** (the manual's recommendation for
    first stages).
  - Smoothing factor **0.1–0.5**. The manual gives 0.5–2 for noisy data and
    0.005–0.01 for high-quality data. Your resistances are good (~1 %
    error), but part of the misfit is modelling error.
  - **Robust weighting scheme** on.
  - Stop criteria: 10 iterations.
- **Options tab:** Smoothness ratio **0.3–0.5**. Min/Max resistivity
  1 / 10 000 Ω·m.
- **Second run:** switch to **Occam** (smoothest model), then optionally
  **Focused** (Threshold about 0.01–0.1, Sharpness about 0.5) for a
  sharp-boundary version. Save each model in the *Buffer* (Model 1–4) to
  compare them.
- Target: a misfit of about 3–5 %.

### 5. IP inversion (p. 65)
1. With the resistivity model converged, switch to the polarizability mode
   (toolbar button) and invert F1 on top of the resistivity model.
2. Because the file has 4 frequency channels, the *Time lapse* menu
   appears. Run the *Time-lapse inversion* of all channels. The channel
   selector in the toolbar switches F1–F4.
3. Then run *Time lapse / Cole-cole inversion*. It gives 3D models of
   **chargeability, time constant τ and exponent c** (*Options / Model /
   Cole-cole model*).
   - Caveat: 0.16–0.9 Hz is a narrow band. τ is only constrained if it falls
     roughly within 1/(2π·0.9) … 1/(2π·0.16) ≈ 0.2 … 1 s. Check this against
     your laboratory SIP spectra before interpreting τ.
4. Optional: *Options / Inversion / Cross-gradient* for a joint
   resistivity–IP inversion (structural coupling).
5. *Options / Extra / EM coupling removal* exists, but the strongly negative
   phases (likely coupling) have already been excluded with
   `weightip = 0`. Only test it as a sensitivity check.

### 6. Compare with pyGIMLi
Compare with `../results/ssip3d_model.vtk` / `ssip3d_model_cells.csv` and
the DOI index in the same files:
- resistive core at x ≈ 1300–1800 m,
- conductive zone at x ≈ 2000–2300 m (top resolved, depth extent not),
- chargeable zone at x ≈ 1100–1300 m near L22 (top at about 150 m depth;
  the deeper part is unresolved).

Features seen by both codes are robust. Differences show where the
regularization, not the data, decides.

## Denoising (what was done to the data)
Scripts: `../06_denoise.py`, `../07_export_zond.py`. QC:
`../results/denoise/summary.txt`, `fig_denoise_qc.png`,
`injection_consistency.csv`.
- **Consistency cleaning.** For each injection, the electrode potentials
  are reconstructed by robust (Huber) complex least squares from all
  recorded dipoles, which also covers the phase. Each dipole is replaced by
  the reconstructed potential difference, and dipoles inconsistent by > 5 %
  are removed.
  - L24 and L26 are almost perfectly consistent (median residual 0.01 %),
    so their dipoles were derived from electrode potentials.
  - L22 is 20× less consistent. 7 off-end injections failed
    (x = −640, −473, −320, −160, 2335, 2900, 3060 m) and are removed; their
    positions in the file are most likely wrong.
  - The same test showed that L22 apparent resistivities were computed with
    horizontal distances, so they are converted back to resistance that
    way.
- **Additional exclusions.** On L24 and L26, wrong positions cannot be
  detected this way, because those lines are stored as resistance.
  Injections flagged by the pyGIMLi misfit analysis are removed there
  (L24 −160 m, L26 −185 m).
- **Error estimates.** From the scatter of amplitude and phase across F1–F4
  about a smooth log-frequency trend, as a function of signal level.
  Resistance noise is about 0.05 % for strong signals and 2–5 % for the
  weakest. Phase noise is about 1.5 mrad for strong signals and tens of mrad
  for weak ones. These replace the flat 2 % of the original L24/L26 files.
- **Phase QC.** Phases outside −20 … 150 mrad at any frequency are excluded
  from IP (EM coupling or noise).
- **Selection.** Of the all-pairs dipoles (which are largely redundant),
  the same non-redundant subset as in the pyGIMLi inversion is kept: one
  dipole per injection and receiver electrode, length growing with offset,
  minimum offset 60 m.
- **Earlier Zond projects.** In the original .z2d files the `weight` column
  was 1 − (absolute error)/100 ≈ 0.9999 for every datum, so errors were
  effectively ignored.

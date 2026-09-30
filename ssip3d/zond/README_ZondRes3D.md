# Running the denoised SSIP data in ZondRes3D

These files come from `06_denoise.py` and `07_export_zond.py`. ZondRes3D is
Windows software with a licence, so the inversion itself is run on your PC.
The official manual could not be downloaded when this guide was written, so
menu names below follow the documented ZondRes3D functions ("Collect from
2D", "Collect IP", smoothing/focusing inversion). The exact labels may differ
slightly in your version.

## Files

| File | Content | Data |
|---|---|---|
| `L22_F1.dat` … `L26_F4.dat` | Res2DInv general array with IP: denoised resistance, phase (mrad), absolute errors. **Same geometry in F1–F4** of a line | L22 1 918, L24 2 439, L26 2 302 |
| `L22_res.dat`, `L24_res.dat`, `L26_res.dat` | resistance only (includes data whose phase failed QC) | L22 2 355, L24 2 895, L26 2 737 |
| `*.z2d` | the same in ZondRes2D format; `weight` = min(1, 3 % / relative error), `eta_a` = phase/10 as in your original .z2d files | |
| `ssip3d_denoised_3d.csv` | every datum with x, y, z of A, M, N (B remote), R, 3D geometric factor, ρa, errors, 4 phases | 7 987 |

All values are **transfer resistance (Ω)**, including L22, which was stored
as apparent resistivity. ZondRes3D computes its own geometric factors from
the 3D electrode positions and topography.

## Line positions (assumption: please check)
| Line | start (x, y) | direction |
|---|---|---|
| L22 | (0, 0) | +x (chainage) |
| L24 | (0, 200) | +x |
| L26 | (0, 400) | +x |

The lines are assumed parallel, 200 m apart, with the same chainage origin.
If you have GPS / UTM coordinates, enter the real start point and azimuth of
each line in step 1 instead.

## Recommended workflow

1. **Build the 3D dataset.** Use *Collect from 2D* with `L22_F1.dat`,
   `L24_F1.dat`, `L26_F1.dat`, giving each line its start point and
   direction from the table above. Check in the electrode map that the three
   lines are 200 m apart and that the off-end injections lie on the line
   extensions (−480 … 3060 m).
   For the resistivity model you can use `L*_res.dat` instead (about 20 %
   more data). The IP inversion then needs the `L*_F*.dat` set.
2. **IP channels.** The F1 files already carry the F1 phase. To invert the
   other frequencies, either build one project per frequency (step 1 with
   `L*_F2.dat`, and so on), or use *Collect IP* to add F2–F4 as extra IP
   channels. The geometry is identical, so the channels match row by row.
3. **Check units.** Data type = resistance (not apparent resistivity),
   remote electrode C2 = infinity, IP in mrad (phase). If the program asks
   for chargeability, use the `.z2d` files, where eta = phase/10 as in your
   earlier projects.
4. **Mesh.**
   - Cell width about 20 m along x (half the 40 m electrode spacing), 40–50 m
     across the lines (y). Do not use cells as wide as the 200 m line
     spacing.
   - Thin first layer (about 10 m) under the topography, with thickness
     increasing by a factor of about 1.1 with depth.
   - Model depth about 600–700 m.
   - Keep topography on.
5. **Errors / weights.** The files carry measured error estimates (median
   about 1 % for resistance, higher for weak signals). The forward solver
   adds its own modelling error, so set a **minimum error (floor) of about
   3–5 %**. Also use a robust (L1) data norm if your version has one.
6. **Resistivity inversion.**
   - Start with smoothing (Occam-type) inversion.
   - Vertical/horizontal smoothing ratio about 0.3–0.5.
   - Starting model: homogeneous, about 350 Ω·m (median ρa).
   - Iterate until the RMS stops decreasing, typically at 3–6 %.
   - Then, optionally, run a focusing inversion as a sharp-boundary
     alternative.
7. **IP inversion.**
   - Run it after the resistivity model has converged, on top of that model.
   - Invert each frequency with the same settings, so differences between
     F1–F4 reflect the data, not the settings.
8. **Export** the models (for example as XYZ or grid) for comparison with
   the pyGIMLi results in `../results/` (`ssip3d_model.vtk`,
   `ssip3d_model_cells.csv`).

## What to compare against the pyGIMLi results
- The pyGIMLi model (same data selection, before denoising) shows a
  resistive core at x ≈ 1300–1800 m, a conductive zone at x ≈ 2000–2300 m,
  and a chargeable zone at x ≈ 1100–1300 m near L22.
- The DOI test gives resistivity reliable to about 400–450 m depth and phase
  only to about 150–250 m. The ZondRes3D model is not better resolved below
  those depths: the limit comes from the data, not the software.
- Agreement between the two codes on the main features is strong evidence
  for your thesis. Differences show which features depend on the
  regularization.

## Denoising summary
See `../results/denoise/summary.txt` and `fig_denoise_qc.png`.
- **Consistency cleaning.** For each injection, the electrode potentials
  are reconstructed by robust least squares from all ~1 000–1 800 recorded
  dipoles. Each dipole is replaced by the reconstructed potential
  difference, and dipoles inconsistent by > 5 % are removed.
  - L24 and L26 are almost perfectly consistent (median residual 0.01 %),
    meaning their dipoles were derived from electrode potentials by the
    acquisition software.
  - L22 is 20× less consistent. It flagged 7 off-end injections (x = −640,
    −473, −320, −160, 2335, 2900, 3060 m), all excluded. Their positions in
    the file are most likely wrong.
- **Additional exclusions.** On L24 and L26, wrong positions do not show up
  in this test, because those lines are stored as resistance. So the
  injections found by the pyGIMLi misfit analysis are excluded as well
  (L24 −160 m, L26 −185 m).
- **Error estimates** come from the scatter of amplitude and phase across
  F1–F4 about a smooth trend in log-frequency, as a function of signal
  level. They replace the flat 2 % in the original L24/L26 files. They are
  conservative: part of the scatter is real spectral change between 0.16 and
  0.9 Hz.
- **Phase QC.** Phases outside −20 … 150 mrad at any frequency (EM coupling
  or noise) are removed from the IP files.
- **Earlier Zond projects.** The `weight` column in the original .z2d
  projects was 1 − (absolute error)/100 ≈ 0.9999 for every datum, so the
  measurement errors were effectively ignored.

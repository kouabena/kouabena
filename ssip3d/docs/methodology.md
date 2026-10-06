# ssip3d methodology

This document describes the numerical method implemented in `ssip3d`. It is meant to
be cited/adapted in the *Methods* section of a paper. Equation numbers refer to this file.

## 1. Spread-spectrum acquisition

The transmitter injects a periodic maximal-length binary sequence (m-sequence) of order
`n` (N = 2ⁿ − 1 chips), chip rate f_c, amplitude I₀:

    I(t) = I₀ · s[⌊t f_c⌋ mod N],   s ∈ {−1, +1}                                   (1)

Its periodic autocorrelation is two-valued (N at zero lag, −1 elsewhere), so its line
spectrum is flat at every harmonic f_k = k/T (T = N/f_c) under a sinc² envelope with
nulls at multiples of f_c. One record therefore measures the earth's transfer function at
≈N/2 frequencies at once (usable band ≈ 1/T to 0.74 f_c), with crest factor 1. This is
what makes the acquisition fast and its noise rejection strong. The default
`order: 9, chip_rate: 64 Hz` gives 0.125–47 Hz, 2.6 decades, in 8 s periods.

## 2. Processing: from records to spectra with empirical errors

For each receiver channel (`processing.py`):

1. linear detrending (removes self-potential drift);
2. segmentation into whole m-sequence periods; the first (transient) period is skipped;
3. for each period p and each analysis band b (log-spaced groups of harmonics, excluding
   sinc nulls and lines within ±0.6 Hz of powerline harmonics), the least-squares
   deconvolution

       Z_p(b) = Σ_{k∈b} V_p(f_k) I_p*(f_k) / Σ_{k∈b} |I_p(f_k)|²                       (2)

   which for an m-sequence is equivalent to cross-correlation with the code;
4. robust stacking over periods **in complex-log space**,
   `ln(sZ) = ln|Z| + iφ`: the median gives the estimate; 1.4826·MAD/√P·1.2533 gives the
   standard error of ln|Z| and of φ *separately*.

These errors are measured from the data themselves and become the inversion's data
weights (section 4), after adding floors in quadrature (default 1 % in amplitude and
1 mrad in phase) to account for modelling error. Data whose error exceeds the thresholds
(`max_err_amp`, `max_err_phase`) at any frequency are rejected.

## 3. Forward model

At angular frequency ω the quasi-static potential satisfies

    −∇·(σ*(r, ω) ∇u) = I δ(r − r_A) − I δ(r − r_B)                                    (3)

with complex conductivity σ* and a no-flux condition at the ground surface.
Electromagnetic induction is neglected; this is the standard IP approximation and holds
for the low frequencies and array sizes SSIP uses.

**Discretisation.** Potentials sit on the nodes of a 3D tensor mesh, conductivity on the
cells, gradients on the edges:

    A(σ) u = q,     A = Gᵀ diag(B σ) G                                                 (4)

`G` is the nodal gradient and `B` lumps each cell's σ·V/4 onto its 12 edges. Air cells
above the topography get σ = 0, so the free surface is a natural Neumann boundary of any
shape. Sides and bottom are Dirichlet, placed far away by geometric padding. A is complex
symmetric with a dominant positive real part. It is factorised once per frequency by
sparse LU without pivoting (symmetric mode), using a geometric nested-dissection ordering
of the tensor-mesh nodes: about 30 s for 10⁵ cells, where COLAMD ordering takes more than
30 minutes. The factors are reused for all electrodes (pole fields) and for the
sensitivities.

**Singularity correction.** The point-source singularity causes a geometry-dependent
discretisation error (≈ 9 % at h = a/2). This is removed with numerical geometric
factors: c_i = Z_i^analytic / Z_i^numeric for a reference half-space. The correction is
exact for any half-space (unit test) and leaves ≈ 2 % error in heterogeneous models
compared with a mesh twice as fine.

**Data** are transfer impedances `Z = (u_M − u_N)/I` (remote electrodes are supported),
inverted as `d = ln(s c Z) = ln|Z| + iφ_Z`. s = ±1 is the observed polarity, so the
phase stays near 0.

## 4. Inversion

**Model**: `m = ln σ*` on the active cells of the core region, i.e. `Re m = ln|σ|` and
`Im m = φ_σ` (= −φ_ρ). Padding cells take the value of the nearest core cell.

**Sensitivities** by reciprocity/adjoint, using the pole fields:

    ∂Z/∂σ_j = −Σ_e (G u_MN)_e (G u_AB)_e B_ej,     J = ∂d/∂m = diag(1/Z) ∂Z/∂σ diag(σ)   (5)

Z is holomorphic in σ*, so J is a complex matrix and the real-equivalent Jacobian is
`[[Re J, −Im J], [Im J, Re J]]`. All products are evaluated in complex arithmetic without
forming that 2×2 block matrix. The Jacobian is verified against finite differences
(second-order Taylor test in `tests/`).

**Objective** (per frequency):

    Φ = ‖W_r Re(d_obs − F(m))‖² + ‖W_i Im(d_obs − F(m))‖²
        + β_r φ_m(Re m) + β_i φ_m(Im m)                                                (6)

    φ_m(x) = α_s ‖w ⊙ (x − x_ref)‖² + Σ_{x,y,z} α_k ‖w_f ⊙ D_k x‖²

`W_r = diag(1/ε_ln|Z|)` and `W_i = diag(1/ε_φ)` come from the processed errors (section 2).
`w` holds the sensitivity weights, √max(s_j/max s, floor) with s_j the cumulative weighted
sensitivity, which counteract the loss of sensitivity with depth.

**Decoupled amplitude/phase trade-off (key feature).** β_r and β_i are controlled
independently. Each is cooled only while its own misfit, χ²_amp or χ²_phase, is above the
discrepancy target N, and warmed up if that misfit falls below 0.4 N. Updates are damped
near the target to avoid oscillation. For small phases J is nearly block-diagonal, so the
amplitude and phase images reach their statistically justified misfits in a single
Gauss-Newton run. In classical complex-resistivity inversion a single β over- or
under-regularises the phase image, and a separate phase-refinement stage is needed for
that reason.

**Gauss-Newton step**: (Jᵀ_R W² J_R + diag(β_r, β_i) ⊗ RᵀR) δm = −g, solved by
Jacobi-preconditioned conjugate gradients on ℝ²ⁿ with inner product Re(xᴴy). A
backtracking line search follows, with bounds on ln|σ| and φ_σ.

**Multi-frequency strategy.** Frequencies are inverted from low to high. Each one starts
from the previous solution and inherits its β values; this "frequency continuation" makes
subsequent frequencies cost 1–3 iterations. Optionally (`freq_coupling > 0`) the previous
model is also the reference model, which enforces spectrally smooth cells. The same
regulariser (same sensitivity weights) is used at all frequencies, for consistency
between images.

## 5. Spectral interpretation (per cell)

From the recovered ρ*(f_k) = 1/σ*(f_k) in every cell (`spectral.py`):

* **Debye decomposition** (linear, non-negative, smoothness-regularised NNLS):
  ρ*(ω) = ρ₀ − Σ_l a_l iωτ_l/(1 + iωτ_l). It gives total chargeability m = Σa_l/ρ₀,
  normalised chargeability m/ρ₀ and mean relaxation time
  τ_mean = exp(Σ m_l ln τ_l / m).
* **Pelton Cole-Cole fit** (nonlinear least squares, initialised from the Debye result):
  ρ₀, m, τ, c. Values of τ outside the resolvable band (one decade beyond the measured
  frequencies) are flagged as NaN.

Cells with volume-normalised coverage below `coverage_min` are not interpreted.

## 6. Reproducibility

* A single YAML file defines the whole experiment. Defaults are explicit
  (`pipeline.DEFAULTS`) and the merged configuration is written to `config_used.yaml`.
* `provenance.json` records the configuration's SHA-256, the package version, the git
  commit (with a `-dirty` flag), the Python/NumPy/SciPy versions, the platform, the
  command line and the run time.
* All randomness (synthetic noise) derives from `seed`. The processing and inversion are
  deterministic.
* The stages communicate through documented files (`spectra.csv/npz`, `inversion.npz`,
  `spectral.npz`, `model.vtk`, `model.csv`), so each stage can be rerun or replaced
  independently.
* Synthetic studies simulate on a finer mesh than the inversion mesh (`refine`) to avoid
  the "inverse crime".

## 7. Limitations / roadmap

* EM coupling is not modelled. Keep f·(array size)² small, or remove coupling before
  inversion.
* Direct sparse LU (nested dissection) handles a few 10⁵ cells on a workstation. An
  iterative solver (AMG-preconditioned, already benchmarked) would scale further.
* Topography is supported in the forward model (air cells). The singularity correction is
  then computed on a flat copy of the mesh.

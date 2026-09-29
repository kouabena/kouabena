"""Survey layout and inversion settings for the SSIP 3D inversion.

Edit this file (not the scripts) when the survey geometry or the inversion
parameters change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"            # unzipped "SSIP DATA INVERSION" folder
WORK_DIR = ROOT / "work"          # intermediate files (data, mesh, Jacobian)
RESULTS_DIR = ROOT / "results"    # models, VTK and figures

# --- Survey geometry -------------------------------------------------------
# Lines are assumed straight and parallel, all starting at chainage x = 0 and
# running in the same direction, 200 m apart. Local coordinates: x = chainage
# along the line, y = line offset, z = elevation (m a.s.l.).
# Replace these with real UTM coordinates as soon as they are available
# (see README, "Georeferencing").
LINES = {
    # name     file prefix           y (m)
    "L22": ("L22-ALL_Res2dinv", 0.0),
    "L24": ("L24_Res2dinv", 200.0),
    "L26": ("L26_Res2dinv", 400.0),
}

# Frequencies (Hz) stored in the headers of the F1..F4 files.
FREQS = {"F1": 0.15625, "F2": 0.40625, "F3": 0.65625, "F4": 0.90625}
DC_FREQ = "F1"   # lowest frequency is used for the resistivity inversion

# --- Data selection --------------------------------------------------------
RX_SPACING = 40.0          # receiver electrode spacing (m)
MAX_N = 8.0                # dipole length >= offset / MAX_N (pole-dipole n)
MAX_DIPOLE = 320.0         # longest dipole kept (m)
MIN_OFFSET = 60.0          # min. distance Tx -> nearest potential electrode
#   (20 m offsets have the largest forward-modelling error, see README)
# Transmitters removed after pass 1 (>40 % of their data misfit by >5 sigma;
# all are off-end injections whose true positions are uncertain).
EXCLUDE_TX = {"L22": [-640.0, -160.0], "L24": [-160.0], "L26": [-185.0]}
# Data misfit by more than OUTLIER_SIGMA in the pass-1 inversion are listed in
# this file by flag_outliers.py and removed in step 1 (if the file exists).
OUTLIER_FILE = ROOT / "outliers_pass1.csv"
OUTLIER_SIGMA = 5.0
MAX_REL_ERR_DC = 0.10      # drop data whose reported error exceeds 10 %
ERR_FLOOR_DC = 0.05        # error model: max(reported, 5 %); ~p95 of forward error
MAX_ABS_PHASE = 150.0      # mrad; phases above this are treated as noise
MIN_PHASE = -20.0          # mrad; strongly negative phases (EM coupling/noise)
#   cannot be fitted by a positive-phase model and are excluded
MAX_PHASE_ERR = 10.0       # mrad
PHASE_ERR_FLOOR = 1.0      # mrad absolute floor
PHASE_ERR_REL = 0.05       # plus 5 % of |phase|

# --- Mesh ------------------------------------------------------------------
PARA_DEPTH = 700.0         # depth of the parameter domain below electrodes (m)
PARA_BOUNDARY = 100.0      # lateral margin of parameter domain (m)
PARA_MAX_CELL = 2.0e6      # max tetrahedron volume in the parameter domain (m3)
SURFACE_AREA = 2500.0      # max triangle area on the surface (m2)
PARA_DX = 15.0             # refinement node this far below each electrode (m)
REFINE_DX = 10.0           # extra surface nodes +-10 m along line around electrodes
MESH_QUALITY = 1.5         # TetGen radius-edge ratio (smaller = finer)

# --- Inversion -------------------------------------------------------------
LAM_DC = 20.0
ZWEIGHT = 0.3              # vertical/horizontal smoothness ratio
MAX_ITER_DC = 8
LAM_IP = 30.0
MAX_ITER_IP = 5

# --- Display -----------------------------------------------------------------
COVERAGE_DROP = 2.25       # decades below near-surface coverage = 'resolved'

# --- DOI test (05_doi.py) -----------------------------------------------------
DOI_ALPHA_S = 1.0          # smallness weight (pyGIMLi cType=10 default).
#   pyGIMLi smoothness is an unnormalized cell difference, so alpha must be
#   O(1) for the reference to control insensitive cells: 0.01 and 0.03 gave
#   R ~ 0.01 even 700 m deep (deep cells followed smoothness instead)
DOI_CUTOFF = 0.2           # DOI index below this = resolved
DOI_MAX_ITER = 6           # DC iterations per DOI run (fixed, from final model)
DOI_MAX_ITER_IP = 6

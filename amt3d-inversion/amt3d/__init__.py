"""amt3d - compact 3D AMT/MT impedance inversion with visualisation."""
from .mesh import Mesh, padded_spacing, geometric_spacing
from .forward import Simulation, MU0, HAVE_PARDISO
from .data import ImpedanceData, phase_tensor, determinant_average
from .inversion import Smoothness, bostick_start_model, invert

__version__ = "0.1.0"

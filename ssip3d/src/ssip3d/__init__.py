"""ssip3d: reproducible 3D inversion of spread-spectrum induced polarization data."""

__version__ = "0.1.0"

from .forward import LogImpedanceProblem, ModelMap, Simulation  # noqa: E402,F401
from .inversion import InversionOptions, gauss_newton, invert_multifrequency  # noqa: E402,F401
from .mesh import TensorMesh3D, build_mesh  # noqa: E402,F401
from .models import ColeColeModel, build_cole_cole_model, cole_cole_resistivity  # noqa: E402,F401
from .processing import SpectralData, process_dataset  # noqa: E402,F401
from .survey import Survey, make_survey  # noqa: E402,F401
from .waveform import PRBSWaveform, m_sequence  # noqa: E402,F401

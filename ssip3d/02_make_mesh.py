"""Step 2: build the 3D tetrahedral inversion mesh with topography.

Topography comes from the electrode elevations of the three lines and is
interpolated linearly between the lines (and held constant beyond them).
"""
import numpy as np
from pygimli.physics import ert

import config as C
from meshing import build_mesh


def main():
    data = ert.load(str(C.WORK_DIR / "ssip3d.dat"))
    mesh = build_mesh(np.array(data.sensors()))
    print(mesh)
    print("parameter cells:", sum(mesh.cellMarkers() == 2))
    mesh.save(str(C.WORK_DIR / "mesh.bms"))
    mesh.exportVTK(str(C.WORK_DIR / "mesh.vtk"))


if __name__ == "__main__":
    main()

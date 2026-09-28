"""Tetrahedral mesh with topography for the SSIP 3D inversion."""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt

import config as C


def topography(sensors):
    """Return a function z(x, y) built from the electrode elevations.

    Along each line the elevation is interpolated linearly between electrodes
    (constant beyond the line ends); between lines it is interpolated
    linearly in y (constant beyond the outer lines).
    """
    sensors = np.asarray(sensors)
    ys = np.unique(sensors[:, 1])
    prof = []
    for y in ys:
        s = sensors[sensors[:, 1] == y]
        s = s[np.argsort(s[:, 0])]
        prof.append((s[:, 0], s[:, 2]))

    def z_of(x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        zl = np.array([np.interp(x, px, pz) for px, pz in prof])  # (nl, n)
        if len(ys) == 1:
            return zl[0]
        out = np.empty_like(x, dtype=float)
        for i in range(len(x)):
            out[i] = np.interp(y[i], ys, zl[:, i])
        return out

    return z_of


def build_mesh(sensors):
    """Create the inversion mesh (marker 2 = parameter domain, 1 = boundary).

    The mesh is generated for a flat surface (z = 0) and then draped onto
    the topography: every node is shifted by topo(x, y) * w(z), with w = 1 at
    the surface decreasing linearly to 0 at the bottom of the model. Extra
    surface nodes at +-REFINE_DX along the line around every electrode keep
    the potentials near the electrodes accurate (Tx-Rx offsets start at 20 m).
    """
    sensors = np.asarray(sensors, dtype=float)
    pts = [sensors]
    for line_y in np.unique(sensors[:, 1]):
        s = sensors[sensors[:, 1] == line_y]
        for dx in (-C.REFINE_DX, C.REFINE_DX):
            pts.append(np.c_[s[:, 0] + dx, s[:, 1], s[:, 2]])
    pts = np.vstack(pts)
    # drop refinement points that fall within ~2 m of another point
    _, keep = np.unique(np.round(pts[:, :2] / 2.0), axis=0, return_index=True)
    pts = pts[np.sort(keep)]
    flat = pts.copy()
    flat[:, 2] = 0.0

    xspan, yspan = np.ptp(flat[:, 0]), np.ptp(flat[:, 1])
    # paraBoundary is a relative factor on the point bounding box
    para_boundary = [(xspan + 2 * C.PARA_BOUNDARY) / xspan,
                     (yspan + 2 * C.PARA_BOUNDARY) / yspan]
    plc = mt.createParaMeshPLC3D(
        flat, paraDX=C.PARA_DX, paraDepth=C.PARA_DEPTH,
        paraBoundary=para_boundary, paraMaxCellSize=C.PARA_MAX_CELL,
        boundary=[4.0, xspan / yspan], surfaceMeshArea=C.SURFACE_AREA,
        surfaceMeshQuality=30)
    mesh = mt.createMesh(plc, quality=C.MESH_QUALITY)

    # drape onto topography
    topo = topography(sensors)
    pos = np.array(mesh.positions())
    zbot = pos[:, 2].min()
    w = 1.0 - pos[:, 2] / zbot          # 1 at z = 0, 0 at the bottom
    dz = topo(pos[:, 0], pos[:, 1]) * w
    for node, d in zip(mesh.nodes(), dz):
        node.setPos(node.pos() + pg.Pos(0.0, 0.0, d))
    mesh.createNeighborInfos(True)
    return mesh

"""3D tensor mesh with nodal finite-volume operators and topography support.

Conventions
-----------
* x, y horizontal, z positive **up**. Node/cell arrays are ordered with x
  fastest, then y, then z (Fortran order), z from the bottom of the mesh up.
* Potentials live on nodes, (complex) conductivity lives on cells, gradients
  live on edges.  The discrete operator is ``A = G^T diag(M_e(sigma)) G`` where
  ``M_e`` lumps each cell's conductivity onto its 12 edges.  Cells above the
  topographic surface ("air") get sigma = 0, so the ground surface is a natural
  no-flux (Neumann) boundary, whatever its shape.  Sides and bottom are
  Dirichlet (u = 0) and must be far away (padding cells).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def _diff(n: int, h: np.ndarray) -> sp.csr_matrix:
    """(n, n+1) forward difference divided by cell widths."""
    d = sp.diags([-np.ones(n), np.ones(n)], [0, 1], shape=(n, n + 1))
    return sp.diags(1.0 / h) @ d


def _node_cell_incidence(n: int) -> sp.csr_matrix:
    """(n+1, n) 0/1 matrix: node i touches cells i-1 and i."""
    rows = np.r_[np.arange(n), np.arange(1, n + 1)]
    cols = np.r_[np.arange(n), np.arange(n)]
    return sp.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n + 1, n))


@dataclass
class TensorMesh3D:
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray  # bottom -> top
    origin: tuple = (0.0, 0.0, 0.0)  # bottom-south-west corner
    core: tuple | None = None  # ((i0,i1),(j0,j1),(k0,k1)) half-open cell index ranges
    active: np.ndarray | None = field(default=None, repr=False)  # cells below topography

    def __post_init__(self):
        self.hx = np.asarray(self.hx, float)
        self.hy = np.asarray(self.hy, float)
        self.hz = np.asarray(self.hz, float)
        self.nx, self.ny, self.nz = len(self.hx), len(self.hy), len(self.hz)
        x0, y0, z0 = self.origin
        self.nodes_x = x0 + np.r_[0.0, np.cumsum(self.hx)]
        self.nodes_y = y0 + np.r_[0.0, np.cumsum(self.hy)]
        self.nodes_z = z0 + np.r_[0.0, np.cumsum(self.hz)]
        self.cc_x = 0.5 * (self.nodes_x[1:] + self.nodes_x[:-1])
        self.cc_y = 0.5 * (self.nodes_y[1:] + self.nodes_y[:-1])
        self.cc_z = 0.5 * (self.nodes_z[1:] + self.nodes_z[:-1])
        if self.active is None:
            self.active = np.ones(self.n_cells, bool)
        if self.core is None:
            self.core = ((0, self.nx), (0, self.ny), (0, self.nz))

    # ------------------------------------------------------------------ sizes
    @property
    def shape_cells(self):
        return (self.nx, self.ny, self.nz)

    @property
    def shape_nodes(self):
        return (self.nx + 1, self.ny + 1, self.nz + 1)

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def n_nodes(self):
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    @property
    def cell_volumes(self):
        return np.kron(self.hz, np.kron(self.hy, self.hx))

    @property
    def cell_centers(self):
        X, Y, Z = np.meshgrid(self.cc_x, self.cc_y, self.cc_z, indexing="ij")
        return np.c_[X.ravel("F"), Y.ravel("F"), Z.ravel("F")]

    def cell_ijk(self):
        i, j, k = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        return i.ravel("F"), j.ravel("F"), k.ravel("F")

    def node_index(self, i, j, k):
        return np.asarray(i) + (self.nx + 1) * (np.asarray(j) + (self.ny + 1) * np.asarray(k))

    # -------------------------------------------------------------- operators
    def grad(self) -> sp.csr_matrix:
        """Nodal gradient, (n_edges, n_nodes); edges ordered [x; y; z]."""
        Ix, Iy, Iz = (sp.identity(n + 1) for n in (self.nx, self.ny, self.nz))
        Gx = sp.kron(Iz, sp.kron(Iy, _diff(self.nx, self.hx)))
        Gy = sp.kron(Iz, sp.kron(_diff(self.ny, self.hy), Ix))
        Gz = sp.kron(_diff(self.nz, self.hz), sp.kron(Iy, Ix))
        return sp.vstack([Gx, Gy, Gz]).tocsr()

    def edge_cell_incidence(self) -> sp.csr_matrix:
        """(n_edges, n_cells) 0/1: which cells share each edge (4 at most)."""
        Ix, Iy, Iz = (sp.identity(n) for n in (self.nx, self.ny, self.nz))
        Cx, Cy, Cz = (_node_cell_incidence(n) for n in (self.nx, self.ny, self.nz))
        Ax = sp.kron(Cz, sp.kron(Cy, Ix))
        Ay = sp.kron(Cz, sp.kron(Iy, Cx))
        Az = sp.kron(Iz, sp.kron(Cy, Cx))
        return sp.vstack([Ax, Ay, Az]).tocsr()

    def node_cell_incidence(self) -> sp.csr_matrix:
        """(n_nodes, n_cells) 0/1: which cells touch each node (8 at most)."""
        Cx, Cy, Cz = (_node_cell_incidence(n) for n in (self.nx, self.ny, self.nz))
        return sp.kron(Cz, sp.kron(Cy, Cx)).tocsr()

    def dirichlet_nodes(self) -> np.ndarray:
        """Boolean mask of nodes on the sides and bottom of the mesh."""
        i, j, k = np.meshgrid(
            np.arange(self.nx + 1), np.arange(self.ny + 1), np.arange(self.nz + 1), indexing="ij"
        )
        m = (i == 0) | (i == self.nx) | (j == 0) | (j == self.ny) | (k == 0)
        return m.ravel("F")

    # --------------------------------------------------------- core helpers
    def core_mask(self) -> np.ndarray:
        (i0, i1), (j0, j1), (k0, k1) = self.core
        i, j, k = self.cell_ijk()
        return (i >= i0) & (i < i1) & (j >= j0) & (j < j1) & (k >= k0) & (k < k1)

    @property
    def core_shape(self):
        (i0, i1), (j0, j1), (k0, k1) = self.core
        return (i1 - i0, j1 - j0, k1 - k0)

    def surface_node_z(self) -> np.ndarray:
        """For each (ix, iy) node column, the z index of the highest node touching ground."""
        touch = (self.node_cell_incidence() @ self.active.astype(float)) > 0
        touch = touch.reshape(self.shape_nodes, order="F")
        nzn = self.nz + 1
        kk = np.where(touch, np.arange(nzn)[None, None, :], -1)
        return kk.max(axis=2)

    def snap_electrodes(self, xyz: np.ndarray):
        """Map electrode coordinates to surface node indices. Returns (node_idx, snapped_xyz)."""
        xyz = np.atleast_2d(np.asarray(xyz, float))
        ix = np.abs(self.nodes_x[None, :] - xyz[:, :1]).argmin(axis=1)
        iy = np.abs(self.nodes_y[None, :] - xyz[:, 1:2]).argmin(axis=1)
        ksurf = self.surface_node_z()
        if xyz.shape[1] > 2 and np.any(np.isfinite(xyz[:, 2])):
            # buried electrodes: nearest node in z, but never above the surface
            iz = np.abs(self.nodes_z[None, :] - xyz[:, 2:3]).argmin(axis=1)
            iz = np.minimum(iz, ksurf[ix, iy])
        else:
            iz = ksurf[ix, iy]
        idx = self.node_index(ix, iy, iz)
        snapped = np.c_[self.nodes_x[ix], self.nodes_y[iy], self.nodes_z[iz]]
        return idx, snapped

    def summary(self) -> str:
        return (
            f"TensorMesh3D: {self.nx} x {self.ny} x {self.nz} = {self.n_cells} cells "
            f"({int(self.active.sum())} active), core {self.core_shape}, "
            f"x [{self.nodes_x[0]:.1f}, {self.nodes_x[-1]:.1f}], "
            f"y [{self.nodes_y[0]:.1f}, {self.nodes_y[-1]:.1f}], "
            f"z [{self.nodes_z[0]:.1f}, {self.nodes_z[-1]:.1f}]"
        )


def _padding(h0: float, n: int, factor: float) -> np.ndarray:
    return h0 * factor ** np.arange(1, n + 1)


def build_mesh(
    electrodes: np.ndarray,
    dx: float,
    dy: float | None = None,
    dz: float | None = None,
    depth: float | None = None,
    margin: float | None = None,
    n_pad: int = 8,
    pad_factor: float = 1.4,
    topo: np.ndarray | None = None,
    dz_growth: float = 1.0,
) -> TensorMesh3D:
    """Build a padded tensor mesh around an electrode layout.

    Parameters
    ----------
    electrodes : (n, 2|3) electrode coordinates.
    dx, dy, dz : core cell sizes (dy defaults to dx, dz to dx/2).
    depth : depth of the core region below the lowest electrode
            (default: 1/3 of the largest electrode separation).
    margin : horizontal core margin around the electrodes (default 2*dx).
    n_pad, pad_factor : number and expansion of padding cells (all sides + bottom).
    topo : optional (m, 3) xyz points of the ground surface; if None and the
           electrodes have z values, those are used; otherwise flat at z=0.
    dz_growth : geometric growth of core cell thickness with depth (1 = uniform).
    """
    el = np.atleast_2d(np.asarray(electrodes, float))
    dy = dx if dy is None else dy
    dz = dx / 2.0 if dz is None else dz
    margin = 2 * dx if margin is None else margin
    xy = el[:, :2]
    span = np.ptp(xy, axis=0)
    if depth is None:
        depth = max(np.hypot(*span) / 3.0, 4 * dz)

    has_z = el.shape[1] > 2
    if topo is None and has_z and np.ptp(el[:, 2]) > 1e-9:
        topo = el[:, :3]
    if topo is not None:
        topo = np.asarray(topo, float)
        ztop, zlow = topo[:, 2].max(), topo[:, 2].min()
    else:
        z0 = float(np.median(el[:, 2])) if has_z else 0.0
        ztop = zlow = z0

    def axis(lo, hi, h):
        # nodes on multiples of h, so electrodes on a regular grid fall on nodes
        lo = h * np.floor(lo / h + 1e-9)
        n = int(np.ceil((hi - lo) / h - 1e-9))
        return lo, np.full(n, h)

    x_lo, hx_core = axis(xy[:, 0].min() - margin, xy[:, 0].max() + margin, dx)
    y_lo, hy_core = axis(xy[:, 1].min() - margin, xy[:, 1].max() + margin, dy)
    # vertical: core from top of topography down to zlow - depth
    hz_core = []
    z = ztop
    h = dz
    while z > zlow - depth - 1e-9:
        hz_core.append(h)
        z -= h
        h *= dz_growth
    hz_core = np.array(hz_core[::-1])
    hx = np.r_[_padding(dx, n_pad, pad_factor)[::-1], hx_core, _padding(dx, n_pad, pad_factor)]
    hy = np.r_[_padding(dy, n_pad, pad_factor)[::-1], hy_core, _padding(dy, n_pad, pad_factor)]
    hz = np.r_[_padding(hz_core[0], n_pad, pad_factor)[::-1], hz_core]
    x0 = x_lo - _padding(dx, n_pad, pad_factor).sum()
    y0 = y_lo - _padding(dy, n_pad, pad_factor).sum()
    z0 = ztop - hz.sum()
    core = (
        (n_pad, n_pad + len(hx_core)),
        (n_pad, n_pad + len(hy_core)),
        (n_pad, n_pad + len(hz_core)),
    )
    mesh = TensorMesh3D(hx, hy, hz, origin=(x0, y0, z0), core=core)
    if topo is not None:
        mesh.active = active_from_topography(mesh, topo)
    return mesh


def active_from_topography(mesh: TensorMesh3D, topo: np.ndarray) -> np.ndarray:
    """Cells whose centre lies below the interpolated topographic surface."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    topo = np.asarray(topo, float)
    cc = mesh.cell_centers
    if np.ptp(topo[:, 0]) < 1e-9 or np.ptp(topo[:, 1]) < 1e-9 or len(topo) < 3:
        # degenerate (single line): interpolate along the varying axis
        ax = 0 if np.ptp(topo[:, 0]) >= np.ptp(topo[:, 1]) else 1
        o = np.argsort(topo[:, ax])
        zs = np.interp(cc[:, ax], topo[o, ax], topo[o, 2])
    else:
        lin = LinearNDInterpolator(topo[:, :2], topo[:, 2])
        near = NearestNDInterpolator(topo[:, :2], topo[:, 2])
        zs = lin(cc[:, :2])
        bad = ~np.isfinite(zs)
        zs[bad] = near(cc[bad, :2])
    return cc[:, 2] < zs

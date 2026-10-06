"""3D quasi-static complex-conductivity forward modelling and adjoint sensitivities.

At each frequency the potential obeys  -div(sigma*(w) grad u) = I delta(r - r_s),
with complex conductivity sigma* (electromagnetic induction neglected, the usual
assumption for IP below a few hundred Hz with moderate arrays).

Data are complex transfer impedances  Z = (u_M - u_N) / I  for each ABMN.
The inversion works with the complex log  d = ln(s Z) = ln|Z| + i phi_Z, where
``s = +-1`` is a fixed polarity so that the phase stays close to zero.

Sensitivities use reciprocity: one LU factorisation per frequency, one
solve per electrode (pole fields), and

    dZ/dsigma_j = - sum_e (G u_MN)_e (G u_AB)_e  B_ej

where ``B`` lumps cell j onto its edges.  Since Z is holomorphic in sigma*, the
complex Jacobian d ln Z / d ln sigma* carries both amplitude and phase
sensitivities (Cauchy-Riemann).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .mesh import TensorMesh3D
from .survey import Survey


class ModelMap:
    """Map an inversion model on active core cells to all mesh cells.

    Active cells outside the core (padding) take the value of the nearest
    active core cell, inactive (air) cells are excluded entirely.
    """

    def __init__(self, mesh: TensorMesh3D):
        self.mesh = mesh
        core = mesh.core_mask() & mesh.active
        self.model_cells = np.flatnonzero(core)  # indices in full mesh
        act = np.flatnonzero(mesh.active)
        cc = mesh.cell_centers
        tree = cKDTree(cc[self.model_cells])
        _, nearest = tree.query(cc[act])
        # cells inside the core map to themselves
        pos = np.full(mesh.n_cells, -1)
        pos[self.model_cells] = np.arange(len(self.model_cells))
        target = np.where(pos[act] >= 0, pos[act], nearest)
        self.P = sp.csr_matrix(
            (np.ones(len(act)), (act, target)), shape=(mesh.n_cells, len(self.model_cells))
        )

    @property
    def n_model(self):
        return len(self.model_cells)

    def to_mesh(self, m_model, fill=0.0):
        out = self.P @ m_model
        out[~self.mesh.active] = fill
        return out


class Simulation:
    """Complex DC/IP simulation on a :class:`TensorMesh3D`.

    Parameters
    ----------
    mesh, survey : geometry.
    """

    def __init__(self, mesh: TensorMesh3D, survey: Survey):
        self.mesh = mesh
        self.survey = survey
        # nodes touching at least one ground cell, minus Dirichlet nodes
        touch = (mesh.node_cell_incidence() @ mesh.active.astype(float)) > 0
        self.free = np.flatnonzero(touch & ~mesh.dirichlet_nodes())
        G = mesh.grad()
        self.G = G[:, self.free].tocsr()
        Aec = mesh.edge_cell_incidence()
        self.B = (Aec @ sp.diags(mesh.cell_volumes / 4.0)).tocsr()  # (n_edges, n_cells)
        # drop edges that touch no ground at all (air-only edges)
        keep = (np.abs(self.B) @ mesh.active.astype(float)) > 0
        self.G = self.G[keep]
        self.B = self.B[keep]
        node_idx, self.electrode_xyz = mesh.snap_electrodes(survey.electrodes)
        pos = np.full(mesh.n_nodes, -1)
        pos[self.free] = np.arange(len(self.free))
        self.elec_free = pos[node_idx]
        if np.any(self.elec_free < 0):
            raise ValueError("an electrode snapped to a Dirichlet node; enlarge the mesh padding")
        self.snap_distance = np.linalg.norm(
            self.electrode_xyz[:, :2] - survey.electrodes[:, :2], axis=1
        )
        self._perm = self._nested_dissection()

    def _nested_dissection(self, leaf=64):
        """Geometric nested-dissection ordering of the free nodes.

        Recursively splits the node set by planes normal to its longest axis
        and orders the separators last; on tensor meshes this keeps the fill of
        the sparse LU factors close to optimal (orders of magnitude less
        factorisation time than COLAMD for 1e5 cells).
        """
        nx, ny, _ = self.mesh.shape_nodes
        f = self.free
        ijk = (f % nx, (f // nx) % ny, f // (nx * ny))
        out = []

        def rec(idx):
            if len(idx) < leaf:
                out.append(idx)
                return
            c = [a[idx] for a in ijk]
            ax = int(np.argmax([np.ptp(x) for x in c]))
            mid = (c[ax].min() + c[ax].max()) // 2
            rec(idx[c[ax] < mid])
            rec(idx[c[ax] > mid])
            out.append(idx[c[ax] == mid])
        rec(np.arange(len(f)))
        return np.concatenate(out)

    # ------------------------------------------------------------- solving
    def _system(self, sigma_cells):
        s = np.where(self.mesh.active, sigma_cells, 0.0)
        me = self.B @ s
        return (self.G.T @ sp.diags(me) @ self.G).tocsc()

    def pole_fields(self, sigma_cells):
        """Potentials (n_free, n_elec) for unit current at each electrode."""
        A = self._system(sigma_cells)
        p = self._perm
        # A is complex symmetric with a dominant positive real part: factor without
        # pivoting (symmetric mode) in nested-dissection order
        lu = spla.splu(A[p][:, p].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        n_e = len(self.elec_free)
        rhs = np.zeros((A.shape[0], n_e), dtype=A.dtype)
        rhs[self.elec_free, np.arange(n_e)] = 1.0
        U = np.empty_like(rhs)
        U[p] = lu.solve(rhs[p])
        return U

    def _gather(self, mat):
        """Append a zero column so that index -1 (remote electrode) gives zero."""
        return np.concatenate([mat, np.zeros((mat.shape[0], 1), mat.dtype)], axis=1)

    def predict_from_fields(self, U):
        V = self._gather(U[self.elec_free])  # (n_elec, n_elec+1): V[i, j] potential at i, source j
        V = np.concatenate([V, np.zeros((1, V.shape[1]), V.dtype)], axis=0)
        a, b, m, n = self.survey.abmn.T
        return V[m, a] - V[m, b] - V[n, a] + V[n, b]

    def predict(self, sigma_cells):
        """Complex transfer impedance Z (ohm) for every ABMN."""
        return self.predict_from_fields(self.pole_fields(sigma_cells))

    def analytic_halfspace(self, rho=1.0):
        """Analytic transfer impedance of a homogeneous half-space (flat surface)."""
        return rho / self.survey.geometric_factor()

    def correction_factors(self, rho_ref=100.0):
        """Numerical geometric-factor correction c = Z_analytic / Z_numeric for a half-space.

        Multiplying simulated impedances by ``c`` removes most of the
        discretisation error caused by the point-source singularity at the
        electrodes. With topography the factors are computed on a flat-surface
        copy of the mesh (electrodes projected onto it), which captures the
        source-singularity error but not the terrain effect itself.
        """
        if np.all(self.mesh.active):
            sim = self
        else:
            from .mesh import TensorMesh3D

            m = self.mesh
            flat = TensorMesh3D(m.hx, m.hy, m.hz, origin=m.origin, core=m.core)
            ztop = flat.nodes_z[-1]
            el = self.survey.electrodes.copy()
            el[:, 2] = ztop
            sim = Simulation(flat, Survey(el, self.survey.abmn))
        Zn = sim.predict(np.full(sim.mesh.n_cells, 1.0 / rho_ref, dtype=complex))
        Za = rho_ref / self.survey.geometric_factor()
        c = np.real(Za / Zn)
        bad = ~np.isfinite(c) | (c <= 0)
        c[bad] = 1.0
        return c

    def jacobian_sigma(self, U, cell_weights: sp.spmatrix | None = None, chunk: int = 128):
        """dZ/dsigma (n_data, n_cols) where columns are mesh cells mapped by ``cell_weights``.

        ``cell_weights`` is an (n_cells, n_cols) sparse matrix (e.g. diag(sigma) @ P for
        d/d ln sigma on a reduced model); default identity.
        """
        Bm = self.B if cell_weights is None else (self.B @ cell_weights).tocsr()
        GU = self._gather(self.G @ U)  # (n_edges, n_elec+1)
        a, b, m, n = self.survey.abmn.T
        J = np.empty((self.survey.n_data, Bm.shape[1]), dtype=GU.dtype)
        BmT = Bm.T.tocsr()
        for s in range(0, self.survey.n_data, chunk):
            sl = slice(s, s + chunk)
            gtx = GU[:, a[sl]] - GU[:, b[sl]]
            grx = GU[:, m[sl]] - GU[:, n[sl]]
            J[sl] = -(BmT @ (gtx * grx)).T
        return J


class LogImpedanceProblem:
    """Forward problem in inversion variables at a single frequency.

    model  m = ln sigma*  on the active core cells (complex vector)
    data   d = ln(s Z)    (complex vector)
    """

    def __init__(self, sim: Simulation, model_map: ModelMap, polarity: np.ndarray, correction=None):
        self.sim = sim
        self.map = model_map
        self.polarity = np.asarray(polarity, float)
        self.correction = np.ones(sim.survey.n_data) if correction is None else np.asarray(correction)

    def sigma(self, m):
        return np.exp(self.map.to_mesh(m))

    def data_from_Z(self, Z):
        return np.log(self.polarity * self.correction * Z)

    def forward(self, m, need_jacobian=True):
        sig = self.sigma(m)
        U = self.sim.pole_fields(sig)
        Z = self.sim.predict_from_fields(U)
        d = self.data_from_Z(Z)
        if not need_jacobian:
            return d, None
        W = sp.diags(np.where(self.sim.mesh.active, sig, 0.0)) @ self.map.P
        J = self.sim.jacobian_sigma(U, W)
        J /= Z[:, None]
        return d, J


def apparent_resistivity(survey: Survey, Z):
    return survey.geometric_factor() * np.real(Z)

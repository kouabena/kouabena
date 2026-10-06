"""3D finite-difference MT forward solver and exact Jacobian.

Electric field on the edges of a staggered (Yee) grid, time dependence
exp(+i w t) (as ModEM), so that

    curl curl E + i w mu0 sigma E = 0.

Discretised (finite-volume, mass-lumped) as

    K e = (C^T Mf C + i w Me(sigma)) e = 0

with Dirichlet tangential E on the outer boundary taken from a 1D solution
of the boundary conductivity. Interior unknowns are solved with a sparse
direct factorisation (MKL Pardiso if available, else SuperLU).
The *same* factorisation is reused for

    * both source polarisations,
    * all adjoint solves needed for the full Jacobian.

so an exact Jacobian costs only back-substitutions.
"""
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU0 = 4e-7 * np.pi

# Optional fast direct solver (Intel MKL Pardiso). Falls back to SuperLU.
if "PYPARDISO_MKL_RT" not in os.environ:
    _libs = glob.glob(os.path.join(sys.prefix, "lib", "libmkl_rt.so*")) + \
        glob.glob("/usr/local/lib/libmkl_rt.so*")
    if _libs:
        os.environ["PYPARDISO_MKL_RT"] = _libs[0]
try:
    import pypardiso
    HAVE_PARDISO = True
except Exception:  # not installed or MKL missing
    HAVE_PARDISO = False


class DirectSolver:
    """Factorise a complex (symmetric) sparse matrix once, solve many RHS.

    Pardiso is used through the real-equivalent system [[A,-B],[B,A]];
    SuperLU works on the complex matrix directly.
    """

    def __init__(self, K, backend="auto"):
        self.n = K.shape[0]
        self.backend = "pardiso" if (backend == "auto" and HAVE_PARDISO) or backend == "pardiso" else "superlu"
        if self.backend == "pardiso":
            A, B = K.real, K.imag
            self.R = sp.bmat([[A, -B], [B, A]], format="csr")
            self.solver = pypardiso.PyPardisoSolver()
            self.solver.factorize(self.R)
        else:
            self.lu = spla.splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A")

    def solve(self, b):
        if self.backend == "superlu":
            return self.lu.solve(b)
        rhs = np.concatenate([b.real, b.imag], axis=0)
        x = self.solver.solve(self.R, np.ascontiguousarray(rhs))
        return x[:self.n] + 1j * x[self.n:]

    def free(self):
        if self.backend == "pardiso":
            self.solver.free_memory(everything=True)


def _diff(n):
    """(n x n+1) node-to-cell difference (topological, no lengths)."""
    return sp.diags([-np.ones(n), np.ones(n)], [0, 1], shape=(n, n + 1))


def _avg(n):
    """(n+1 x n) cell-to-node averaging with weight 1/2 per neighbour."""
    return sp.diags([0.5 * np.ones(n), 0.5 * np.ones(n)], [0, -1], shape=(n + 1, n))


def _eye(n):
    return sp.identity(n, format="csr")


def _kron3(az, ay, ax):
    return sp.kron(az, sp.kron(ay, ax))


def _bilinear(gx, gy, x, y):
    """Indices/weights of bilinear interpolation of point (x, y) on grid gx x gy."""
    i = int(np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2))
    j = int(np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2))
    tx = (x - gx[i]) / (gx[i + 1] - gx[i])
    ty = (y - gy[j]) / (gy[j + 1] - gy[j])
    return [(i, j, (1 - tx) * (1 - ty)), (i + 1, j, tx * (1 - ty)),
            (i, j + 1, (1 - tx) * ty), (i + 1, j + 1, tx * ty)]


class Operators:
    """Mesh-dependent sparse operators (built once)."""

    def __init__(self, mesh):
        M = mesh
        nx, ny, nz = M.shape
        hx, hy, hz = M.hx, M.hy, M.hz
        self.mesh = M
        self.nEx = nx * (ny + 1) * (nz + 1)
        self.nEy = (nx + 1) * ny * (nz + 1)
        self.nEz = (nx + 1) * (ny + 1) * nz
        self.nE = self.nEx + self.nEy + self.nEz
        self.nFx = (nx + 1) * ny * nz
        self.nFy = nx * (ny + 1) * nz
        self.nFz = nx * ny * (nz + 1)

        # --- curl: faces <- edges --------------------------------------
        Dz_Ey = _kron3(_diff(nz), _eye(ny), _eye(nx + 1))
        Dy_Ez = _kron3(_eye(nz), _diff(ny), _eye(nx + 1))
        Dz_Ex = _kron3(_diff(nz), _eye(ny + 1), _eye(nx))
        Dx_Ez = _kron3(_eye(nz), _eye(ny + 1), _diff(nx))
        Dx_Ey = _kron3(_eye(nz + 1), _eye(ny), _diff(nx))
        Dy_Ex = _kron3(_eye(nz + 1), _diff(ny), _eye(nx))
        D = sp.bmat([[None, -Dz_Ey, Dy_Ez],
                     [Dz_Ex, None, -Dx_Ez],
                     [-Dy_Ex, Dx_Ey, None]], format="csr")
        e1 = lambda n: np.ones(n)
        L = np.r_[np.kron(e1((ny + 1) * (nz + 1)), hx),
                  np.kron(e1(nz + 1), np.kron(hy, e1(nx + 1))),
                  np.kron(hz, e1((nx + 1) * (ny + 1)))]
        A = np.r_[np.kron(hz, np.kron(hy, e1(nx + 1))),
                  np.kron(hz, np.kron(e1(ny + 1), hx)),
                  np.kron(e1(nz + 1), np.kron(hy, hx))]
        self.C = (sp.diags(1.0 / A) @ D @ sp.diags(L)).tocsr()

        # --- mass matrices ----------------------------------------------
        vol = M.vol
        Pe = sp.vstack([_kron3(_avg(nz), _avg(ny), _eye(nx)),
                        _kron3(_avg(nz), _eye(ny), _avg(nx)),
                        _kron3(_eye(nz), _avg(ny), _avg(nx))])
        self.Pe = (Pe @ sp.diags(vol)).tocsr()          # Me diag = Pe @ sigma
        Pf = sp.vstack([_kron3(_eye(nz), _eye(ny), _avg(nx)),
                        _kron3(_eye(nz), _avg(ny), _eye(nx)),
                        _kron3(_avg(nz), _eye(ny), _eye(nx))])
        mf = Pf @ (vol / MU0)
        self.A0 = (self.C.T @ sp.diags(mf) @ self.C).tocsr()

        # --- boundary / interior edges ----------------------------------
        def grid(shape):
            return [g.ravel(order="F") for g in np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")]
        ix, jx, kx = grid((nx, ny + 1, nz + 1))
        iy, jy, ky = grid((nx + 1, ny, nz + 1))
        iz, jz, kz = grid((nx + 1, ny + 1, nz))
        bx = (jx == 0) | (jx == ny) | (kx == 0) | (kx == nz)
        by = (iy == 0) | (iy == nx) | (ky == 0) | (ky == nz)
        bz = (iz == 0) | (iz == nx) | (jz == 0) | (jz == ny)
        bnd = np.r_[bx, by, bz]
        self.ib = np.flatnonzero(bnd)
        self.ii = np.flatnonzero(~bnd)
        self.kx, self.ky = kx, ky
        self.Pe_i = self.Pe[self.ii]

        # outer ring of cells: used for the 1D boundary conductivity
        ic, jc, _ = grid((nx, ny, nz))
        self.ring = (ic == 0) | (ic == nx - 1) | (jc == 0) | (jc == ny - 1)

    # ------------------------------------------------------------------
    def boundary_fields(self, sigma, omega):
        """Tangential E on boundary edges for x- and y-polarisation from a
        1D layered solution (log-mean conductivity of the outer cell ring)."""
        M = self.mesh
        nz = M.nz
        logs = np.log(sigma[self.ring]).reshape(-1, nz, order="F")
        s1 = np.exp(logs.mean(axis=0))
        Dz = _diff(nz)
        K = (Dz.T @ sp.diags(1.0 / (M.hz * MU0)) @ Dz).tocsr() + sp.diags(1j * omega * (_avg(nz) @ (s1 * M.hz))).tocsr()
        K = K.tocsr()
        e = np.zeros(nz + 1, complex)
        e[0] = 1.0
        inner = np.arange(1, nz)
        e[inner] = spla.spsolve(K[inner][:, inner].tocsc(), -K[inner][:, [0]].toarray().ravel())
        ex = np.zeros(self.nE, complex)
        ey = np.zeros(self.nE, complex)
        ex[:self.nEx] = e[self.kx]
        ey[self.nEx:self.nEx + self.nEy] = e[self.ky]
        return ex, ey


class Simulation:
    """3D AMT forward modelling for full impedance tensors.

    Model parameter: m = ln(sigma) of earth cells (S/m).
    """

    def __init__(self, mesh, stations, freqs, sigma_air=1e-8, solver="auto"):
        self.solver_backend = solver
        self.mesh = mesh
        self.ops = Operators(mesh)
        self.stations = np.asarray(stations, float)
        self.freqs = np.asarray(freqs, float)
        self.sigma_air = sigma_air
        self.earth = mesh.earth
        self.nm = int(self.earth.sum())
        self._build_receivers()

    @property
    def ns(self):
        return len(self.stations)

    def sigma(self, m):
        s = np.full(self.mesh.nC, self.sigma_air)
        s[self.earth] = np.exp(m)
        return s

    def _build_receivers(self):
        M, op = self.mesh, self.ops
        nx, ny, nz = M.shape
        k0 = M.nair  # surface node index
        rows = {c: ([], [], []) for c in ("ex", "ey", "hx", "hy")}
        for s, (x, y) in enumerate(self.stations):
            # Ex edges: (xc, yn) at node k0
            for i, j, w in _bilinear(M.xc, M.yn, x, y):
                rows["ex"][0].append(s); rows["ex"][1].append(i + nx * j + nx * (ny + 1) * k0); rows["ex"][2].append(w)
            for i, j, w in _bilinear(M.xn, M.yc, x, y):
                rows["ey"][0].append(s); rows["ey"][1].append(op.nEx + i + (nx + 1) * j + (nx + 1) * ny * k0); rows["ey"][2].append(w)
            # H faces: average the cell just above and below the surface
            for k in (k0 - 1, k0):
                for i, j, w in _bilinear(M.xn, M.yc, x, y):
                    rows["hx"][0].append(s); rows["hx"][1].append(i + (nx + 1) * j + (nx + 1) * ny * k); rows["hx"][2].append(0.5 * w)
                for i, j, w in _bilinear(M.xc, M.yn, x, y):
                    rows["hy"][0].append(s); rows["hy"][1].append(op.nFx + i + nx * j + nx * (ny + 1) * k); rows["hy"][2].append(0.5 * w)
        nF = op.nFx + op.nFy + op.nFz
        mk = lambda r, n: sp.csr_matrix((r[2], (r[0], r[1])), shape=(self.ns, n))
        self.Qe = [mk(rows["ex"], op.nE), mk(rows["ey"], op.nE)]
        self.Qb = [mk(rows["hx"], nF) @ op.C, mk(rows["hy"], nF) @ op.C]  # times -1/(i w mu0) -> H

    # ------------------------------------------------------------------
    def _solve_frequency(self, sig, m_earth_sigma, f, jacobian):
        op = self.ops
        w = 2 * np.pi * f
        K = op.A0 + sp.diags(1j * w * (op.Pe @ sig)).tocsr()
        Kii = K[op.ii][:, op.ii].tocsc()
        Kib = K[op.ii][:, op.ib]
        lu = DirectSolver(Kii, self.solver_backend)
        eb = op.boundary_fields(sig, w)
        E = np.zeros((self.ns, 2, 2), complex)   # [station, component a, polarisation p]
        H = np.zeros((self.ns, 2, 2), complex)
        efull = []
        Qh = [q * (-1.0 / (1j * w * MU0)) for q in self.Qb]
        for p in range(2):
            e = eb[p].copy()
            e[op.ii] = lu.solve(-(Kib @ e[op.ib]))
            efull.append(e)
            for a in range(2):
                E[:, a, p] = self.Qe[a] @ e
                H[:, a, p] = Qh[a] @ e
        Hinv = np.linalg.inv(H)
        Z = E @ Hinv
        if not jacobian:
            lu.free()
            return Z, None
        # --- exact Jacobian via adjoint back-substitutions ------------------
        # dZ_ab = sum_p Hinv_pb (Qe_a - Z_a0 Qh_0 - Z_a1 Qh_1) de_p
        nc = self.ns * 4
        Jc = np.zeros((nc, self.nm), complex)
        Qe_i = [q[:, op.ii] for q in self.Qe]
        Qh_i = [q[:, op.ii] for q in Qh]
        for p in range(2):
            blocks = []
            for a in range(2):
                base = [sp.diags(Z[:, a, 0]) @ Qh_i[0], sp.diags(Z[:, a, 1]) @ Qh_i[1]]
                La = Qe_i[a] - base[0] - base[1]
                for b in range(2):
                    blocks.append(sp.diags(Hinv[:, p, b]) @ La)
            # rows ordered (a, b, station) -> reorder to (station, a, b)
            Lp = sp.vstack(blocks).tocsr()
            order = np.arange(nc).reshape(4, self.ns).T.ravel()
            Lp = Lp[order]
            # K is complex symmetric (K^T = K): adjoint solve uses the same factors
            X = lu.solve(np.asarray(Lp.T.todense()))
            G = (op.Pe_i[:, self.earth].multiply(efull[p][op.ii][:, None]) * (-1j * w)).tocsc()
            Jc += (G.T @ X).T
        lu.free()
        Jc *= m_earth_sigma[None, :]
        return Z, Jc.reshape(self.ns, 2, 2, self.nm)

    def predict(self, m, jacobian=False, verbose=False):
        """Return Z (nf, ns, 2, 2) [Ohm] and optionally J (nf, ns, 2, 2, nm)
        with J = dZ / d ln(sigma)."""
        sig = self.sigma(m)
        se = np.exp(m)
        Z = np.zeros((len(self.freqs), self.ns, 2, 2), complex)
        J = np.zeros((len(self.freqs), self.ns, 2, 2, self.nm), complex) if jacobian else None
        for n, f in enumerate(self.freqs):
            if verbose:
                print(f"    frequency {f:10.3f} Hz", end="\r")
            Z[n], Jn = self._solve_frequency(sig, se, f, jacobian)
            if jacobian:
                J[n] = Jn
        return Z, J

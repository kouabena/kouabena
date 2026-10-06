"""Electrode layouts and four-electrode (ABMN) configurations.

Electrode index ``-1`` denotes a remote ("infinite") electrode, so pole-dipole
and pole-pole arrays are supported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Survey:
    electrodes: np.ndarray  # (n_elec, 3)
    abmn: np.ndarray  # (n_data, 4) int, -1 = remote

    def __post_init__(self):
        self.electrodes = np.atleast_2d(np.asarray(self.electrodes, float))
        if self.electrodes.shape[1] == 2:
            self.electrodes = np.c_[self.electrodes, np.zeros(len(self.electrodes))]
        self.abmn = np.atleast_2d(np.asarray(self.abmn, int))
        if self.abmn.shape[1] != 4:
            raise ValueError("abmn must have 4 columns")
        if self.abmn.max() >= len(self.electrodes):
            raise ValueError("abmn references an electrode that does not exist")
        a, b, m, n = self.abmn.T
        if np.any((a == b) & (a >= 0)) or np.any((m == n) & (m >= 0)):
            raise ValueError("A==B or M==N in abmn")
        if np.any(a < 0) or np.any(m < 0):
            raise ValueError("use column B / N (not A / M) for remote electrodes")

    @property
    def n_data(self):
        return len(self.abmn)

    @property
    def n_electrodes(self):
        return len(self.electrodes)

    def geometric_factor(self) -> np.ndarray:
        """Half-space geometric factor K such that rho_a = K * V / I."""
        e = self.electrodes

        def inv_r(i, j):
            out = np.zeros(len(i))
            ok = (i >= 0) & (j >= 0)
            d = np.linalg.norm(e[i[ok], :2] - e[j[ok], :2], axis=1)
            out[ok] = 1.0 / np.maximum(d, 1e-12)
            return out

        a, b, m, n = self.abmn.T
        g = inv_r(a, m) - inv_r(b, m) - inv_r(a, n) + inv_r(b, n)
        with np.errstate(divide="ignore"):
            return 2 * np.pi / g

    def midpoints(self) -> np.ndarray:
        """Pseudo-location of each datum: mean of the (non-remote) electrodes."""
        pts = []
        for row in self.abmn:
            idx = row[row >= 0]
            pts.append(self.electrodes[idx].mean(axis=0))
        return np.array(pts)

    def pseudo_depth(self) -> np.ndarray:
        """Median depth of investigation proxy: 0.19 * max electrode separation."""
        out = []
        for row in self.abmn:
            idx = row[row >= 0]
            p = self.electrodes[idx, :2]
            d = np.linalg.norm(p[:, None] - p[None], axis=-1).max()
            out.append(0.19 * d)
        return np.array(out)

    # ----------------------------------------------------------------- I/O
    def save(self, path):
        np.savez(path, electrodes=self.electrodes, abmn=self.abmn)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["electrodes"], d["abmn"])

    def to_csv(self, elec_path, abmn_path):
        np.savetxt(elec_path, self.electrodes, delimiter=",", header="x,y,z", comments="")
        np.savetxt(abmn_path, self.abmn, delimiter=",", header="a,b,m,n", comments="", fmt="%d")

    @classmethod
    def from_csv(cls, elec_path, abmn_path):
        e = np.loadtxt(elec_path, delimiter=",", skiprows=1, ndmin=2)
        q = np.loadtxt(abmn_path, delimiter=",", skiprows=1, ndmin=2).astype(int)
        return cls(e, q)


def grid_electrodes(n_per_line=24, n_lines=4, spacing=25.0, line_spacing=50.0, x0=0.0, y0=0.0, z=0.0):
    """Parallel lines of electrodes along x. Returns (electrodes, line_id)."""
    xs = x0 + spacing * np.arange(n_per_line)
    ys = y0 + line_spacing * np.arange(n_lines)
    X, Y = np.meshgrid(xs, ys)  # (n_lines, n_per_line)
    el = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    line = np.repeat(np.arange(n_lines), n_per_line)
    return el, line


def dipole_dipole(n_per_line, n_lines, n_max=6, a_mult=(1,), cross_line=True, cross_n_max=3):
    """Dipole-dipole ABMN for parallel lines of ``n_per_line`` electrodes.

    In-line: A,B = (i, i+a), M = B + n*a, N = M + a, n = 1..n_max.
    Cross-line (if enabled): transmitter on line l, receiver on line l+1 with
    the same dipole length, offsets 0..cross_n_max-1 dipoles along x.
    """
    rows = []

    def eid(line, k):
        return line * n_per_line + k

    for ln in range(n_lines):
        for a in a_mult:
            for i in range(n_per_line):
                A, B = i, i + a
                for n in range(1, n_max + 1):
                    M = B + n * a
                    N = M + a
                    if N >= n_per_line:
                        break
                    rows.append([eid(ln, A), eid(ln, B), eid(ln, M), eid(ln, N)])
        if cross_line and ln + 1 < n_lines:
            for a in a_mult:
                for i in range(n_per_line - a):
                    for off in range(-cross_n_max + 1, cross_n_max):
                        M = i + off * a
                        N = M + a
                        if M < 0 or N >= n_per_line:
                            continue
                        rows.append([eid(ln, i), eid(ln, i + a), eid(ln + 1, M), eid(ln + 1, N)])
    return np.array(rows, int)


def pole_dipole(n_per_line, n_lines, n_max=6, a=1):
    """Forward pole-dipole along each line with a remote B electrode."""
    rows = []
    for ln in range(n_lines):
        for i in range(n_per_line):
            for n in range(1, n_max + 1):
                M = i + n * a
                N = M + a
                if N >= n_per_line:
                    break
                rows.append([ln * n_per_line + i, -1, ln * n_per_line + M, ln * n_per_line + N])
    return np.array(rows, int)


def make_survey(kind="dipole-dipole", n_per_line=24, n_lines=4, spacing=25.0, line_spacing=50.0, **kw):
    el, _ = grid_electrodes(n_per_line, n_lines, spacing, line_spacing)
    if kind in ("dipole-dipole", "dd"):
        q = dipole_dipole(n_per_line, n_lines, **kw)
    elif kind in ("pole-dipole", "pd"):
        q = pole_dipole(n_per_line, n_lines, **kw)
    else:
        raise ValueError(f"unknown array kind {kind!r}")
    return Survey(el, q)

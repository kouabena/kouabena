"""Rectilinear (tensor) mesh for 3D MT/AMT modelling.

Coordinate convention (same as ModEM): x = North, y = East, z = Down.
The air/earth interface is at z = 0; air cells have z < 0.
Cell, edge and face arrays are flattened in Fortran order (x fastest).
"""
import numpy as np


def padded_spacing(core_h, n_core, n_pad, factor):
    """Uniform core of `n_core` cells of size `core_h`, padded on both
    sides with `n_pad` cells growing geometrically by `factor`."""
    pad = core_h * factor ** np.arange(1, n_pad + 1)
    return np.r_[pad[::-1], np.full(n_core, float(core_h)), pad]


def geometric_spacing(first, factor, total):
    """Cells starting at `first`, growing by `factor`, until `total` is covered."""
    h = [float(first)]
    while sum(h) < total:
        h.append(h[-1] * factor)
    return np.array(h)


class Mesh:
    def __init__(self, hx, hy, hz_earth, hz_air, x0=None, y0=None):
        self.hx = np.asarray(hx, float)
        self.hy = np.asarray(hy, float)
        hz_air = np.asarray(hz_air, float)
        self.hz = np.r_[hz_air[::-1], np.asarray(hz_earth, float)]
        self.nair = len(hz_air)
        self.nx, self.ny, self.nz = len(self.hx), len(self.hy), len(self.hz)
        x0 = -self.hx.sum() / 2 if x0 is None else x0
        y0 = -self.hy.sum() / 2 if y0 is None else y0
        self.xn = x0 + np.r_[0.0, np.cumsum(self.hx)]
        self.yn = y0 + np.r_[0.0, np.cumsum(self.hy)]
        self.zn = -hz_air.sum() + np.r_[0.0, np.cumsum(self.hz)]
        self.zn[self.nair] = 0.0
        self.xc = 0.5 * (self.xn[1:] + self.xn[:-1])
        self.yc = 0.5 * (self.yn[1:] + self.yn[:-1])
        self.zc = 0.5 * (self.zn[1:] + self.zn[:-1])

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def nC(self):
        return self.nx * self.ny * self.nz

    @property
    def vol(self):
        return np.kron(self.hz, np.kron(self.hy, self.hx))

    @property
    def earth(self):
        """Boolean mask of earth (inversion) cells."""
        k = np.repeat(np.arange(self.nz), self.nx * self.ny)
        return k >= self.nair

    @property
    def earth_shape(self):
        return (self.nx, self.ny, self.nz - self.nair)

    def to_grid(self, earth_values):
        """Earth-cell vector -> (nx, ny, nz_earth) array."""
        return np.reshape(earth_values, self.earth_shape, order="F")

    @classmethod
    def for_stations(cls, sx, sy, cell=50.0, n_margin=4, n_pad=8, pad_factor=1.5,
                     z_first=10.0, z_factor=1.15, z_core=None, z_pad=8,
                     n_air=8, air_first=10.0, air_factor=3.0, min_period=None,
                     max_period=None, rho_bg=100.0):
        """Build a mesh around station coordinates.

        If `max_period` is given the padding is extended so that it reaches
        at least ~3 skin depths of the longest period in `rho_bg`.
        """
        sx, sy = np.asarray(sx, float), np.asarray(sy, float)
        if max_period is not None:
            skin = 503.0 * np.sqrt(rho_bg * max_period)
        else:
            skin = 5000.0
        ncx = int(np.ceil((sx.max() - sx.min()) / cell)) + 2 * n_margin
        ncy = int(np.ceil((sy.max() - sy.min()) / cell)) + 2 * n_margin

        def pad_for(n_core):
            n = n_pad
            while cell * (pad_factor ** np.arange(1, n + 1)).sum() < 3 * skin:
                n += 1
            return padded_spacing(cell, n_core, n, pad_factor)

        hx, hy = pad_for(ncx), pad_for(ncy)
        if z_core is None:
            z_core = max(1000.0, 0.7 * skin)
        hz_core = geometric_spacing(z_first, z_factor, z_core)
        hz_pad = hz_core[-1] * pad_factor ** np.arange(1, z_pad + 1)
        while (hz_core.sum() + hz_pad.sum()) < 4 * skin:
            hz_pad = np.r_[hz_pad, hz_pad[-1] * pad_factor]
        hz_air = air_first * air_factor ** np.arange(n_air)
        x0 = 0.5 * (sx.min() + sx.max()) - hx.sum() / 2
        y0 = 0.5 * (sy.min() + sy.max()) - hy.sum() / 2
        return cls(hx, hy, np.r_[hz_core, hz_pad], hz_air, x0=x0, y0=y0)

    def __repr__(self):
        return (f"Mesh({self.nx} x {self.ny} x {self.nz} cells, {self.nair} air layers, "
                f"x=[{self.xn[0]:.0f},{self.xn[-1]:.0f}] y=[{self.yn[0]:.0f},{self.yn[-1]:.0f}] "
                f"z=[{self.zn[0]:.0f},{self.zn[-1]:.0f}] m)")

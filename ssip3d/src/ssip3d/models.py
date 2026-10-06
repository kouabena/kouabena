"""Complex conductivity models (Cole-Cole / Pelton) and simple model builders."""
from __future__ import annotations

import numpy as np


def cole_cole_resistivity(freq, rho0, m, tau, c):
    """Pelton Cole-Cole complex resistivity.

    rho*(w) = rho0 * [1 - m (1 - 1/(1 + (i w tau)^c))]

    ``freq`` may be scalar or (nf,); parameters scalar or (n,). Returns
    shape broadcast of (nf, n) (or (n,) for scalar freq).
    """
    f = np.asarray(freq, float)
    w = 2 * np.pi * f
    iwt = (1j * np.multiply.outer(w, np.asarray(tau, float))) ** np.asarray(c, float)
    return np.asarray(rho0) * (1 - np.asarray(m) * (1 - 1 / (1 + iwt)))


def cole_cole_conductivity(freq, rho0, m, tau, c):
    return 1.0 / cole_cole_resistivity(freq, rho0, m, tau, c)


class ColeColeModel:
    """Per-cell Cole-Cole parameters on a mesh (all arrays of length n_cells)."""

    def __init__(self, rho0, m, tau, c):
        self.rho0 = np.asarray(rho0, float)
        self.m = np.asarray(m, float)
        self.tau = np.asarray(tau, float)
        self.c = np.asarray(c, float)

    def sigma(self, freq):
        return cole_cole_conductivity(freq, self.rho0, self.m, self.tau, self.c)

    def as_dict(self):
        return dict(rho0=self.rho0, m=self.m, tau=self.tau, c=self.c)


def build_cole_cole_model(mesh, background: dict, bodies: list[dict] | None = None) -> ColeColeModel:
    """Background Cole-Cole half-space plus box / sphere / layer bodies.

    Each body: ``{"type": "box", "xmin":..,"xmax":..,"ymin":..,"ymax":..,"zmin":..,"zmax":..,
    "rho0":.., "m":.., "tau":.., "c":..}`` or ``{"type": "sphere", "center": [x,y,z],
    "radius": r, ...}`` or ``{"type": "layer", "zmin":.., "zmax":.., ...}``. z is
    elevation (negative below a flat surface at 0). Later bodies overwrite earlier ones.
    """
    n = mesh.n_cells
    p = {k: np.full(n, float(background.get(k, d))) for k, d in
         (("rho0", 100.0), ("m", 0.0), ("tau", 0.1), ("c", 0.5))}
    cc = mesh.cell_centers
    for b in bodies or []:
        t = b.get("type", "box")
        if t == "box":
            inside = (
                (cc[:, 0] >= b["xmin"]) & (cc[:, 0] <= b["xmax"])
                & (cc[:, 1] >= b["ymin"]) & (cc[:, 1] <= b["ymax"])
                & (cc[:, 2] >= b["zmin"]) & (cc[:, 2] <= b["zmax"])
            )
        elif t == "sphere":
            inside = np.linalg.norm(cc - np.asarray(b["center"], float), axis=1) <= b["radius"]
        elif t == "layer":
            inside = (cc[:, 2] >= b["zmin"]) & (cc[:, 2] <= b["zmax"])
        else:
            raise ValueError(f"unknown body type {t!r}")
        for k in p:
            if k in b:
                p[k][inside] = float(b[k])
    return ColeColeModel(**p)

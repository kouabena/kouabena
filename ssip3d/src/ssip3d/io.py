"""Model export (VTK, CSV) and provenance records for reproducibility."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mesh import TensorMesh3D


def core_grid(mesh: TensorMesh3D, values_full: np.ndarray):
    """Extract the core sub-grid: (x_nodes, y_nodes, z_nodes, 3D array [ix, iy, iz])."""
    (i0, i1), (j0, j1), (k0, k1) = mesh.core
    v = np.asarray(values_full, float).copy()
    v[~mesh.active] = np.nan
    arr = v.reshape(mesh.shape_cells, order="F")[i0:i1, j0:j1, k0:k1]
    return mesh.nodes_x[i0:i1 + 1], mesh.nodes_y[j0:j1 + 1], mesh.nodes_z[k0:k1 + 1], arr


def write_vtk(path, mesh: TensorMesh3D, fields: dict):
    """Write cell fields (each on all mesh cells) on the core region as legacy VTK rectilinear grid."""
    path = Path(path)
    grids = {k: core_grid(mesh, v) for k, v in fields.items()}
    x, y, z, a0 = next(iter(grids.values()))
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nssip3d model\nASCII\nDATASET RECTILINEAR_GRID\n")
        fh.write(f"DIMENSIONS {len(x)} {len(y)} {len(z)}\n")
        for name, c in (("X", x), ("Y", y), ("Z", z)):
            fh.write(f"{name}_COORDINATES {len(c)} double\n")
            fh.write(" ".join(f"{v:.6g}" for v in c) + "\n")
        fh.write(f"CELL_DATA {a0.size}\n")
        for k, (_, _, _, arr) in grids.items():
            fh.write(f"SCALARS {k} double 1\nLOOKUP_TABLE default\n")
            vals = arr.ravel(order="F")
            vals = np.where(np.isfinite(vals), vals, np.nan)
            fh.write("\n".join(f"{v:.6g}" for v in vals) + "\n")


def write_xyz_csv(path, mesh: TensorMesh3D, fields: dict):
    """Core active cells as CSV: x, y, z, field1, field2, ..."""
    cells = np.flatnonzero(mesh.core_mask() & mesh.active)
    cc = mesh.cell_centers[cells]
    cols = [cc[:, 0], cc[:, 1], cc[:, 2]] + [np.asarray(v)[cells] for v in fields.values()]
    np.savetxt(path, np.c_[tuple(cols)], delimiter=",", fmt="%.6g",
               header=",".join(["x", "y", "z", *fields]), comments="")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def _git_commit():
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(here)], cwd=here,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except Exception:
        return None


def provenance(cfg: dict, extra: dict | None = None) -> dict:
    import scipy

    from . import __version__

    rec = {
        "software": "ssip3d",
        "version": __version__,
        "git_commit": _git_commit(),
        "config_sha256": config_hash(cfg),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    rec.update(extra or {})
    return rec


def save_json(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    Path(path).write_text(json.dumps(obj, indent=2, default=conv))

"""Impedance data container, error model and ModEM-format I/O."""
import numpy as np

from .forward import MU0

COMP = ["ZXX", "ZXY", "ZYX", "ZYY"]
# ModEM practical units [mV/km]/[nT]  ->  SI impedance [Ohm]
FIELD_TO_OHM = MU0 * 1e3


class ImpedanceData:
    """Full impedance tensors Z[f, s, 2, 2] (Ohm, exp(+iwt)) with standard
    errors err[f, s, 2, 2] (applies to both real and imaginary part)."""

    def __init__(self, freqs, stations, Z, err=None, names=None):
        self.freqs = np.asarray(freqs, float)
        self.stations = np.asarray(stations, float)
        self.Z = np.asarray(Z, complex)
        self.err = np.ones(self.Z.shape) if err is None else np.asarray(err, float)
        self.names = names or [f"S{i:03d}" for i in range(len(self.stations))]

    @property
    def periods(self):
        return 1.0 / self.freqs

    def omega(self):
        return 2 * np.pi * self.freqs[:, None, None, None]

    def rho_a(self):
        return np.abs(self.Z) ** 2 / (self.omega() * MU0)

    def phase(self):
        return np.degrees(np.arctan2(self.Z.imag, self.Z.real))

    def rho_a_err(self):
        return 2 * self.rho_a() * self.err / np.maximum(np.abs(self.Z), 1e-30)

    def phase_err(self):
        return np.degrees(self.err / np.maximum(np.abs(self.Z), 1e-30))

    def copy(self, Z=None):
        return ImpedanceData(self.freqs, self.stations, self.Z if Z is None else Z, self.err.copy(), list(self.names))

    # ---- error model --------------------------------------------------
    def set_error_floor(self, floor=0.05, diag_floor=None):
        """ModEM-style floor: err >= floor * sqrt(|Zxy * Zyx|) for every component."""
        ref = np.sqrt(np.abs(self.Z[..., 0, 1] * self.Z[..., 1, 0]))[..., None, None]
        floors = np.full((2, 2), floor)
        if diag_floor is not None:
            floors[0, 0] = floors[1, 1] = diag_floor
        self.err = np.maximum(self.err, floors * ref)
        return self

    def add_noise(self, rel=0.03, seed=0):
        rng = np.random.default_rng(seed)
        ref = np.sqrt(np.abs(self.Z[..., 0, 1] * self.Z[..., 1, 0]))[..., None, None]
        sd = rel * ref * np.ones(self.Z.shape)
        self.Z = self.Z + sd * (rng.standard_normal(self.Z.shape) + 1j * rng.standard_normal(self.Z.shape))
        return self

    # ---- vectorisation used by the inversion --------------------------
    def vector(self):
        return np.r_[self.Z.real.ravel(), self.Z.imag.ravel()]

    def error_vector(self):
        return np.r_[self.err.ravel(), self.err.ravel()]

    # ---- ModEM ASCII format -------------------------------------------
    def write_modem(self, path, units="[mV/km]/[nT]", lat0=0.0, lon0=0.0):
        scale = FIELD_TO_OHM if units == "[mV/km]/[nT]" else 1.0
        with open(path, "w") as fh:
            fh.write("# Written by amt3d\n")
            fh.write("# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error\n")
            fh.write("> Full_Impedance\n> exp(+i\\omega t)\n")
            fh.write(f"> {units}\n> 0.00\n> {lat0:.3f} {lon0:.3f}\n")
            fh.write(f"> {len(self.freqs)} {len(self.stations)}\n")
            for f, fr in enumerate(self.freqs):
                for s, (x, y) in enumerate(self.stations):
                    for c, name in enumerate(COMP):
                        a, b = divmod(c, 2)
                        z, e = self.Z[f, s, a, b] / scale, self.err[f, s, a, b] / scale
                        fh.write(f"{1/fr:12.5E} {self.names[s]} {lat0:9.4f} {lon0:9.4f} "
                                 f"{x:12.2f} {y:12.2f} {0.0:10.2f} {name} "
                                 f"{z.real:14.6E} {z.imag:14.6E} {e:14.6E}\n")

    @classmethod
    def read_modem(cls, path):
        """Read the Full_Impedance block of a ModEM data file."""
        rows, units, sign = [], "[mV/km]/[nT]", 1.0
        header = 0
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                if line.startswith(">"):
                    header += 1
                    txt = line[1:].strip()
                    if "exp(-i" in txt:
                        sign = -1.0
                    if txt.startswith("["):
                        units = txt
                    continue
                p = line.split()
                if len(p) >= 11 and p[7].upper() in COMP:
                    rows.append((float(p[0]), p[1], float(p[4]), float(p[5]), p[7].upper(),
                                 float(p[8]), float(p[9]), float(p[10])))
        scale = FIELD_TO_OHM if "mV/km" in units else 1.0
        periods = sorted({r[0] for r in rows})
        names = list(dict.fromkeys(r[1] for r in rows))
        xy = {r[1]: (r[2], r[3]) for r in rows}
        nf, ns = len(periods), len(names)
        Z = np.full((nf, ns, 2, 2), np.nan + 0j)
        E = np.full((nf, ns, 2, 2), np.inf)
        pi, si = {p: i for i, p in enumerate(periods)}, {n: i for i, n in enumerate(names)}
        for per, name, _, _, comp, re, im, err in rows:
            a, b = divmod(COMP.index(comp), 2)
            Z[pi[per], si[name], a, b] = scale * (re + 1j * sign * im)
            E[pi[per], si[name], a, b] = scale * err
        # missing data: zero value with infinite error (zero weight)
        miss = np.isnan(Z)
        Z[miss] = 0.0
        E[miss] = np.inf
        return cls(1.0 / np.array(periods), np.array([xy[n] for n in names]), Z, E, names)


def determinant_average(data):
    """Station-averaged determinant apparent resistivity and phase (per freq)."""
    Zdet = np.sqrt(data.Z[..., 0, 0] * data.Z[..., 1, 1] - data.Z[..., 0, 1] * data.Z[..., 1, 0])
    w = 2 * np.pi * data.freqs[:, None]
    ok = np.isfinite(data.err).all(axis=(-1, -2)) & (np.abs(Zdet) > 0)
    rho = np.where(ok, np.abs(Zdet) ** 2 / (w * MU0), np.nan)
    ph = np.where(ok, np.degrees(np.arctan2(np.abs(Zdet.imag), np.abs(Zdet.real))), np.nan)
    return np.exp(np.nanmean(np.log(rho), axis=1)), np.nanmean(ph, axis=1)


def phase_tensor(Z):
    """Phase tensor (Caldwell et al., 2004). Returns phi_max, phi_min,
    alpha, beta (degrees) for Z[..., 2, 2]."""
    X, Y = Z.real, Z.imag
    P = np.linalg.solve(X, Y)
    p11, p12, p21, p22 = P[..., 0, 0], P[..., 0, 1], P[..., 1, 0], P[..., 1, 1]
    tr = p11 + p22
    sk = p12 - p21
    pi1 = 0.5 * np.sqrt((p11 - p22) ** 2 + (p12 + p21) ** 2)
    pi2 = 0.5 * np.sqrt(tr ** 2 + sk ** 2)
    phimax = np.degrees(np.arctan(pi2 + pi1))
    phimin = np.degrees(np.arctan(pi2 - pi1))
    alpha = 0.5 * np.degrees(np.arctan2(p12 + p21, p11 - p22))
    beta = 0.5 * np.degrees(np.arctan2(sk, tr))
    return phimax, phimin, alpha, beta

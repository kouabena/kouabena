"""Configuration-driven, reproducible SSIP workflow.

    simulate  (synthetic only)  true model -> 3D spectra -> raw m-sequence records -> spectra
    process   (field records)   raw records -> spectra with empirical errors
    invert                      spectra -> 3D complex conductivity at every frequency
    spectral                    per-cell Debye decomposition / Cole-Cole fit
    report                      figures, VTK, CSV, provenance

Every stage reads/writes files in ``output_dir`` so stages can be rerun
independently, and every output folder carries the exact configuration,
its SHA-256 and the software environment (provenance.json).
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from . import plotting
from .forward import LogImpedanceProblem, ModelMap, Simulation, apparent_resistivity
from .inversion import InversionOptions, invert_multifrequency
from .io import provenance, save_json, write_vtk, write_xyz_csv
from .mesh import build_mesh
from .models import ColeColeModel, build_cole_cole_model
from .processing import SpectralData, analysis_bands, process_dataset
from .spectral import debye_decomposition, fit_cole_cole
from .survey import Survey, make_survey
from .synthetic import simulate_and_process, simulate_spectra
from .waveform import PRBSWaveform

log = logging.getLogger("ssip3d")

DEFAULTS = {
    "project": "ssip3d_project",
    "output_dir": "results",
    "seed": 0,
    "n_jobs": 1,
    "survey": {
        "kind": "dipole-dipole", "n_per_line": 24, "n_lines": 4, "spacing": 25.0, "line_spacing": 50.0,
        "n_max": 6, "a_mult": [1], "cross_line": True, "cross_n_max": 3, "electrodes_csv": None, "abmn_csv": None,
    },
    "waveform": {"order": 9, "chip_rate": 64.0, "oversampling": 8, "amplitude": 1.0},
    "synthetic": {
        "enabled": False,
        "background": {"rho0": 200.0, "m": 0.02, "tau": 0.01, "c": 0.5},
        "bodies": [],
        "refine": 1,  # simulation mesh = inversion mesh cell size / refine (avoid inverse crime)
        "n_model_freqs": 15,
        "n_periods": 16,
        "noise_rel": 0.01, "noise_abs": 1e-5, "current_noise": 1e-3,
        "powerline_amp": 1e-3, "powerline_freq": 50.0, "drift": 1e-3,
        "save_records": [0],
    },
    "data": {"spectra_csv": None, "timeseries_npz": None},
    "processing": {"n_bands": 8, "fmin": None, "fmax": None, "notch": [50.0], "skip_periods": 1},
    "errors": {"amp_floor": 0.01, "phase_floor": 1e-3, "max_err_amp": 0.2, "max_err_phase": 0.02},
    "mesh": {"dx": None, "dy": None, "dz": None, "depth": None, "margin": None, "n_pad": 8,
             "pad_factor": 1.4, "dz_growth": 1.05, "topography_csv": None},
    "inversion": {
        "max_iter": 8, "chi_target": 1.0, "beta_ratio": 1.0, "beta_cooling": 2.0, "beta_im_scale": 1.0,
        "alpha_s": 1e-3, "alpha_x": 1.0, "alpha_y": 1.0, "alpha_z": 1.0,
        "sensitivity_weighting": True, "sensitivity_floor": 0.05, "cg_maxiter": 60,
        "freq_coupling": 0.0, "frequencies": None, "geometric_correction": True,
    },
    "spectral": {"debye": True, "cole_cole": True, "amp_err": 0.01, "phase_err": 1e-3, "coverage_min": 0.01},
    "report": {"y_sections": None, "z_slices": None, "tau_m_min": 0.03},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        cfg, base = path_or_dict, Path.cwd()
    else:
        p = Path(path_or_dict)
        cfg, base = yaml.safe_load(p.read_text()), p.resolve().parent
    cfg = _merge(DEFAULTS, cfg)
    cfg["_base_dir"] = str(base)
    return cfg


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        base = Path(cfg.get("_base_dir", "."))
        out = Path(cfg["output_dir"])
        self.out = out if out.is_absolute() else base / out
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(exist_ok=True)
        self.base = base
        self._setup_logging()
        public = {k: v for k, v in cfg.items() if not k.startswith("_")}
        (self.out / "config_used.yaml").write_text(yaml.safe_dump(public, sort_keys=False))
        self._prov = provenance(public)
        save_json(self.out / "provenance.json", self._prov)

    def _setup_logging(self):
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                   for h in log.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            log.addHandler(sh)
        fh = logging.FileHandler(self.out / "ssip3d.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)

    def _path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.base / p

    # ----------------------------------------------------------------- objects
    def survey(self) -> Survey:
        s = self.cfg["survey"]
        if s.get("electrodes_csv"):
            return Survey.from_csv(self._path(s["electrodes_csv"]), self._path(s["abmn_csv"]))
        kw = dict(n_max=s["n_max"])
        if s["kind"] in ("dipole-dipole", "dd"):
            kw.update(a_mult=tuple(s["a_mult"]), cross_line=s["cross_line"], cross_n_max=s["cross_n_max"])
        return make_survey(s["kind"], s["n_per_line"], s["n_lines"], s["spacing"], s["line_spacing"], **kw)

    def waveform(self) -> PRBSWaveform:
        w = self.cfg["waveform"]
        return PRBSWaveform(w["order"], w["chip_rate"], w["oversampling"], w["amplitude"])

    def mesh(self, survey: Survey, refine: float = 1.0):
        m = self.cfg["mesh"]
        e = survey.electrodes
        if m["dx"] is None:
            d = np.linalg.norm(e[:, None, :2] - e[None, :, :2], axis=-1)
            spacing = np.min(d[d > 0])
            dx = spacing / 2
        else:
            dx = m["dx"]
        dy = m["dy"] or dx
        dz = m["dz"] or dx / 2
        topo = np.loadtxt(self._path(m["topography_csv"]), delimiter=",", skiprows=1) if m["topography_csv"] else None
        return build_mesh(e, dx / refine, dy / refine, dz / refine, depth=m["depth"], margin=m["margin"],
                          n_pad=m["n_pad"], pad_factor=m["pad_factor"], topo=topo,
                          dz_growth=m["dz_growth"] ** (1.0 / refine))

    def true_model(self, mesh) -> ColeColeModel:
        s = self.cfg["synthetic"]
        return build_cole_cole_model(mesh, s["background"], s["bodies"])

    # ----------------------------------------------------------------- stages
    def simulate(self):
        cfg, s = self.cfg, self.cfg["synthetic"]
        t0 = time.time()
        survey = self.survey()
        wave = self.waveform()
        log.info(f"SIMULATE  {wave.describe()}")
        mesh_sim = self.mesh(survey, refine=float(s["refine"]))
        log.info(f"  simulation mesh: {mesh_sim.summary()}")
        sim = Simulation(mesh_sim, survey)
        model = self.true_model(mesh_sim)
        fmin, fmax = wave.f0, wave.fs / 2
        f_model = np.geomspace(fmin, fmax, s["n_model_freqs"])
        corr = sim.correction_factors(rho_ref=float(s["background"]["rho0"]))
        Zm = simulate_spectra(sim, model, f_model, correction=corr, n_jobs=int(cfg["n_jobs"]))
        p = cfg["processing"]
        freqs, bands = analysis_bands(wave, p["n_bands"], p["fmin"], p["fmax"], p["notch"])
        noise = {k: s[k] for k in ("noise_rel", "noise_abs", "current_noise", "powerline_amp",
                                   "powerline_freq", "drift")}
        spec, saved, _ = simulate_and_process(Zm, f_model, wave, bands, s["n_periods"], cfg["seed"],
                                              save_records=set(s["save_records"]), **noise)
        survey.save(self.out / "survey.npz")
        survey.to_csv(self.out / "electrodes.csv", self.out / "abmn.csv")
        spec.save(self.out / "spectra.npz")
        spec.to_csv(self.out / "spectra.csv", survey)
        np.savez(self.out / "true_spectra.npz", freqs=f_model, Z=Zm)
        for i, (I, V) in saved.items():
            np.savez(self.out / f"record_{i}.npz", current=I, voltage=V, fs=wave.fs)
            plotting.plot_timeseries(I, V, wave, self.out / "figures" / f"record_{i}.png",
                                     title=f"(datum {i}: ABMN {survey.abmn[i].tolist()})")
        log.info(f"  {survey.n_data} data x {len(spec.freqs)} frequencies "
                 f"({spec.freqs.min():.3g}-{spec.freqs.max():.3g} Hz), {spec.n_periods} periods stacked; "
                 f"median errors: ln|Z| {np.median(spec.err_amp):.2e}, phase {np.median(spec.err_phase) * 1e3:.3f} mrad "
                 f"[{time.time() - t0:.0f} s]")
        return spec

    def process(self):
        """Process field records: npz with current (n_tx, n), voltage (n_data, n), tx_index (n_data,)."""
        wave = self.waveform()
        d = np.load(self._path(self.cfg["data"]["timeseries_npz"]))
        p = self.cfg["processing"]
        spec = process_dataset(d["current"], d["voltage"], d["tx_index"], wave, p["n_bands"], p["fmin"],
                               p["fmax"], p["notch"], p["skip_periods"])
        survey = self.survey()
        survey.save(self.out / "survey.npz")
        spec.save(self.out / "spectra.npz")
        spec.to_csv(self.out / "spectra.csv", survey)
        log.info(f"PROCESS  {survey.n_data} data, {len(spec.freqs)} bands, {spec.n_periods} periods")
        return spec

    def _load_data(self):
        dcfg = self.cfg["data"]
        if dcfg.get("spectra_csv"):
            survey = self.survey()
            spec = SpectralData.from_csv(self._path(dcfg["spectra_csv"]), survey)
        else:
            survey = Survey.load(self.out / "survey.npz")
            spec = SpectralData.load(self.out / "spectra.npz")
        return survey, spec

    def invert(self):
        cfg, ic, ec = self.cfg, self.cfg["inversion"], self.cfg["errors"]
        t0 = time.time()
        survey, spec = self._load_data()
        # ---- data selection and error model
        spec = spec.apply_error_floor(ec["amp_floor"], ec["phase_floor"])
        good = (np.all(np.isfinite(spec.Z), 0) & np.all(spec.err_amp < ec["max_err_amp"], 0)
                & np.all(spec.err_phase < ec["max_err_phase"], 0))
        if ic["frequencies"] is not None:
            spec = spec.select(keep_freq=np.asarray(ic["frequencies"], int))
        log.info(f"INVERT  keeping {good.sum()}/{survey.n_data} data (error thresholds), "
                 f"{len(spec.freqs)} frequencies")
        survey = Survey(survey.electrodes, survey.abmn[good])
        spec = spec.select(keep_data=good)
        mesh = self.mesh(survey)
        log.info(f"  inversion mesh: {mesh.summary()}")
        sim = Simulation(mesh, survey)
        mmap = ModelMap(mesh)
        log.info(f"  model parameters: {mmap.n_model} complex ({2 * mmap.n_model} real)")
        k_lo = int(np.argmin(spec.freqs))
        polarity = np.sign(spec.Z[k_lo].real)
        rhoa = apparent_resistivity(survey, spec.Z[k_lo])
        rho_start = float(np.median(np.abs(rhoa)))
        corr = sim.correction_factors(rho_ref=rho_start) if ic["geometric_correction"] else None
        prob = LogImpedanceProblem(sim, mmap, polarity, corr)
        d_obs = np.log(polarity[None] * spec.Z)
        phase0 = float(np.median(-d_obs[k_lo].imag))
        m0 = np.full(mmap.n_model, np.log(1 / rho_start) + 1j * phase0)
        log.info(f"  start: rho = {rho_start:.1f} ohm-m, phase = {phase0 * 1e3:.2f} mrad")
        opts = InversionOptions(**{k: ic[k] for k in (
            "max_iter", "chi_target", "beta_ratio", "beta_cooling", "beta_im_scale", "alpha_s", "alpha_x",
            "alpha_y", "alpha_z", "sensitivity_weighting", "sensitivity_floor", "cg_maxiter")})
        results = invert_multifrequency(prob, spec.freqs, d_obs, spec.err_amp, spec.err_phase, m0, opts,
                                        freq_coupling=ic["freq_coupling"])
        M = np.array([r.m for r in results])
        Zpred = np.array([polarity * np.exp(r.d_pred) for r in results])
        hist = [r.history for r in results]
        np.savez(self.out / "inversion.npz", freqs=spec.freqs, m=M, Zpred=Zpred, Zobs=spec.Z,
                 err_amp=spec.err_amp, err_phase=spec.err_phase, model_cells=mmap.model_cells,
                 abmn=survey.abmn, coverage=results[k_lo].coverage)
        save_json(self.out / "inversion_history.json",
                  {f"{f:.6g}": h for f, h in zip(spec.freqs, hist)})
        log.info(f"  inversion finished [{time.time() - t0:.0f} s]")
        self._mesh, self._mmap, self._survey = mesh, mmap, survey
        return dict(mesh=mesh, mmap=mmap, survey=survey, spec=spec, results=results, Zpred=Zpred)

    def spectral(self, inv=None):
        sc = self.cfg["spectral"]
        survey = Survey.load(self.out / "survey.npz") if inv is None else inv["survey"]
        d = np.load(self.out / "inversion.npz")
        if inv is None:
            survey = Survey(survey.electrodes, d["abmn"])
        mesh = self.mesh(survey)
        mmap = ModelMap(mesh)
        freqs, M = d["freqs"], d["m"]
        rho = 1.0 / np.exp(M)  # (n_f, n_model) complex resistivity
        out = {}
        if sc["debye"]:
            dd = debye_decomposition(freqs, rho, sc["amp_err"], sc["phase_err"])
            out.update({f"dd_{k}": dd[k] for k in ("rho0", "m", "tau_mean", "mn", "misfit")})
        if sc["cole_cole"]:
            log.info("SPECTRAL  Cole-Cole fit per cell")
            sel = d["coverage"] > sc["coverage_min"]
            init = {k: v[sel] for k, v in dd.items() if k not in ("tau", "weights")} if sc["debye"] else None
            cc = fit_cole_cole(freqs, rho[:, sel], init=init, amp_err=sc["amp_err"], phase_err=sc["phase_err"])
            for k, v in cc.items():
                full_v = np.full(rho.shape[1], np.nan)
                full_v[sel] = v
                out[f"cc_{k}"] = full_v
        np.savez(self.out / "spectral.npz", **out, model_cells=mmap.model_cells)
        # ---- exports
        full = {}
        for k, f in enumerate(freqs):
            full[f"rho_{f:.3g}Hz"] = mmap.to_mesh(1 / np.abs(np.exp(M[k])), np.nan)
            full[f"phase_mrad_{f:.3g}Hz"] = mmap.to_mesh(M[k].imag * 1e3, np.nan)
        for k, v in out.items():
            full[k] = mmap.to_mesh(v, np.nan)
        full["coverage"] = mmap.to_mesh(d["coverage"], np.nan)
        write_vtk(self.out / "model.vtk", mesh, full)
        write_xyz_csv(self.out / "model.csv", mesh, full)
        log.info(f"  wrote model.vtk / model.csv ({len(full)} fields)")
        return dict(mesh=mesh, mmap=mmap, fields=full, out=out)

    def report(self, inv=None, spec_out=None):
        rc = self.cfg["report"]
        d = np.load(self.out / "inversion.npz")
        survey = Survey(Survey.load(self.out / "survey.npz").electrodes, d["abmn"])
        mesh = self.mesh(survey)
        mmap = ModelMap(mesh)
        figs = self.out / "figures"
        plotting.plot_survey(survey, mesh, figs / "survey.png")
        spec = SpectralData(d["freqs"], d["Zobs"], d["err_amp"], d["err_phase"])
        idx = np.linspace(0, survey.n_data - 1, 6).astype(int)
        plotting.plot_spectra_fit(d["freqs"], spec, d["Zpred"], idx, figs / "spectra_fit.png")
        plotting.plot_misfit_maps(d["freqs"], spec, d["Zpred"], figs / "data_fit.png")
        import json
        hist = json.loads((self.out / "inversion_history.json").read_text())
        plotting.plot_convergence(d["freqs"], list(hist.values()), figs / "convergence.png")
        sp_ = np.load(self.out / "spectral.npz")
        cov = mmap.to_mesh(d["coverage"], np.nan)
        mask = cov > self.cfg["spectral"]["coverage_min"]
        e = survey.electrodes
        ys = rc["y_sections"] or sorted(set(np.round(e[:, 1], 6)))[:2]
        zs = rc["z_slices"] or [float(np.max(e[:, 2]) - 0.25 * (mesh.cc_z.max() - mesh.cc_z[mesh.core[2][0]]))]
        def P(k):
            return mmap.to_mesh(sp_[k], np.nan)
        rho_rec, m_rec = P("dd_rho0"), P("dd_m")
        tau_key = "cc_tau" if "cc_tau" in sp_ else "dd_tau_mean"
        tau_rec = np.log10(P(tau_key))
        tmask = mask & (m_rec > rc.get("tau_m_min", 0.03))  # tau is meaningless where m ~ 0
        rho_rng = list(np.nanpercentile(rho_rec[mask], [2, 98]))
        m_rng = [0.0, float(np.nanpercentile(m_rec[mask], 99))]
        tau_rng = list(np.nanpercentile(tau_rec[tmask], [5, 95])) if tmask.any() else [-3, 1]
        tm = None
        if self.cfg["synthetic"]["enabled"]:
            tm = self.true_model(mesh)
            act = mesh.active
            rho_rng = [min(rho_rng[0], tm.rho0[act].min()), max(rho_rng[1], tm.rho0[act].max())]
            m_rng = [0.0, max(m_rng[1], tm.m[act].max())]
            lt = np.log10(tm.tau[act])
            tau_rng = [min(lt.min(), tau_rng[0]), max(lt.max(), tau_rng[1])]
        st_rho = dict(cmap="Spectral_r", log=True, label="rho0 (ohm-m)", vmin=rho_rng[0], vmax=rho_rng[1])
        st_m = dict(cmap="magma_r", label="chargeability m (-)", vmin=m_rng[0], vmax=m_rng[1])
        st_tau = dict(cmap="viridis", label="log10 tau (s)", vmin=tau_rng[0], vmax=tau_rng[1])
        panels = [("rho0 recovered", rho_rec, dict(st_rho, mask=mask)),
                  ("m recovered", m_rec, dict(st_m, mask=mask)),
                  ("tau recovered", tau_rec, dict(st_tau, mask=tmask))]
        if tm is not None:
            act = np.where(mesh.active, 1.0, np.nan)
            panels = [("rho0 true", tm.rho0 * act, st_rho), panels[0],
                      ("m true", tm.m * act, st_m), panels[1],
                      ("tau true", np.log10(tm.tau) * act, st_tau), panels[2]]
        plotting.plot_model_panels(mesh, panels, figs / "model_sections.png", y_sections=ys, z_slices=zs,
                                   electrodes=e, suptitle="Spectral parameters (rho0, m: Debye decomposition; "
                                   f"tau: {'Cole-Cole' if tau_key == 'cc_tau' else 'Debye mean'})")
        f = d["freqs"]
        ph_panels = []
        for k in (0, len(f) // 2, len(f) - 1):
            ph_panels.append((f"phase {f[k]:.3g} Hz", mmap.to_mesh(d["m"][k].imag * 1e3, np.nan),
                              dict(cmap="magma_r", label="-phase rho (mrad)", vmin=0)))
        plotting.plot_model_panels(mesh, ph_panels, figs / "phase_sections.png", y_sections=ys, z_slices=zs,
                                   electrodes=e, mask=mask, suptitle="Recovered phase at selected frequencies")
        log.info(f"REPORT  figures in {figs}")

    def run(self, stages=("simulate", "invert", "spectral", "report")):
        t0 = time.time()
        inv = None
        if "simulate" in stages and self.cfg["synthetic"]["enabled"]:
            self.simulate()
        if "process" in stages and self.cfg["data"].get("timeseries_npz"):
            self.process()
        if "invert" in stages:
            inv = self.invert()
        if "spectral" in stages:
            self.spectral(inv)
        if "report" in stages:
            self.report(inv)
        self._prov["elapsed_s"] = round(time.time() - t0, 1)
        self._prov["stages"] = list(stages)
        save_json(self.out / "provenance.json", self._prov)
        log.info(f"DONE in {time.time() - t0:.0f} s -> {self.out}")

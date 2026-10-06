"""Command-line interface:  ssip3d <command> config.yaml"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__


def main(argv=None):
    p = argparse.ArgumentParser(prog="ssip3d", description="3D inversion of spread-spectrum IP data")
    p.add_argument("--version", action="version", version=f"ssip3d {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("run", "run the full workflow"),
                      ("simulate", "synthetic: true model -> raw records -> spectra"),
                      ("process", "field records -> spectra"),
                      ("invert", "spectra -> 3D complex conductivity"),
                      ("spectral", "per-cell Debye / Cole-Cole analysis"),
                      ("report", "figures")):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("config")
        sp.add_argument("-o", "--output-dir", default=None, help="override output_dir")
    ex = sub.add_parser("example", help="copy the example configurations to a folder")
    ex.add_argument("dest", nargs="?", default="ssip3d_examples")
    a = p.parse_args(argv)

    if a.cmd == "example":
        src = Path(__file__).resolve().parent / "examples"
        dst = Path(a.dest)
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.yaml"):
            shutil.copy(f, dst / f.name)
        print(f"example configurations copied to {dst}/  ->  ssip3d run {dst}/synthetic_porphyry.yaml")
        return 0

    from .pipeline import Pipeline, load_config

    cfg = load_config(a.config)
    if a.output_dir:
        cfg["output_dir"] = str(Path(a.output_dir).resolve())
    pl = Pipeline(cfg)
    if a.cmd == "run":
        pl.run()
    elif a.cmd == "simulate":
        pl.simulate()
    elif a.cmd == "process":
        pl.process()
    elif a.cmd == "invert":
        pl.invert()
    elif a.cmd == "spectral":
        pl.spectral()
    elif a.cmd == "report":
        pl.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())

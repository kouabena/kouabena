"""Step 9: compare the original and the harmonized (denoised) pyGIMLi models.

1. Maps the DOI index of the original inversion (05_doi.py, results/) onto
   the mesh of the denoised inversion (results_denoised/) by nearest cell
   centre. The survey geometry is the same, so resolution is essentially
   unchanged. This lets 04_export_results.py blank unresolved cells.
2. Interpolates both models onto common points inside the resolved volume
   and reports differences; writes results_denoised/fig_compare_variants.png
   and compare_variants.txt.

Run after: SSIP_VARIANT=denoised python 03_invert.py
Then:      SSIP_VARIANT=denoised python 04_export_results.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pygimli as pg

import config as C

ORIG = C.ROOT / "results"
DEN = C.ROOT / "results_denoised"
FK = list(C.FREQS)


def centers(folder):
    return np.array(pg.load(str(folder / "paraDomain.bms")).cellCenters())


def main():
    c_o, c_d = centers(ORIG), centers(DEN)
    _, nn = cKDTree(c_o).query(c_d)
    for key in ("doi_res", "doi_phase_F1"):
        if (ORIG / f"{key}.npy").exists():
            np.save(DEN / f"{key}.npy", np.load(ORIG / f"{key}.npy")[nn])

    ok_res = np.load(DEN / "doi_res.npy") < C.DOI_CUTOFF
    ok_ip = np.load(DEN / "doi_phase_F1.npy") < C.DOI_CUTOFF
    lines = ["Original (results/) vs harmonized denoised (results_denoised/)",
             "values compared cell by cell (nearest original cell), inside "
             f"the resolved volume (DOI index < {C.DOI_CUTOFF})", ""]
    r_o = np.load(ORIG / "res.npy")[nn]
    r_d = np.load(DEN / "res.npy")
    dl = np.log10(r_d / r_o)[ok_res]
    cc = np.corrcoef(np.log10(r_o[ok_res]), np.log10(r_d[ok_res]))[0, 1]
    lines.append(f"resistivity: {ok_res.sum()} cells, correlation of log10 "
                 f"rho {cc:.3f}, median |ratio-1| "
                 f"{np.median(np.abs(10 ** dl - 1)) * 100:.1f}%, 90th pct "
                 f"{np.percentile(np.abs(10 ** dl - 1), 90) * 100:.1f}%")
    pairs = [("res", r_o[ok_res], r_d[ok_res], True)]
    for fk in FK:
        p_o = np.load(ORIG / f"phase_{fk}.npy")[nn][ok_ip]
        p_d = np.load(DEN / f"phase_{fk}.npy")[ok_ip]
        cc = np.corrcoef(p_o, p_d)[0, 1]
        lines.append(f"phase {fk}: {ok_ip.sum()} cells, correlation {cc:.3f},"
                     f" median |diff| {np.median(np.abs(p_d - p_o)):.2f} mrad,"
                     f" median phase orig {np.median(p_o):.1f} / denoised "
                     f"{np.median(p_d):.1f} mrad")
        if fk == "F1":
            pairs.append(("phase_F1", p_o, p_d, False))
    for f in ("dc_log.txt", "ip_log.txt"):
        lines += ["", f"{f} original:", (ORIG / f).read_text().strip(),
                  f"{f} denoised:", (DEN / f).read_text().strip()]
    (DEN / "compare_variants.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, (key, a, b, log) in zip(axs, pairs):
        ax.plot(a, b, ".", ms=1.5, alpha=0.3, color="#2a6f97")
        lim = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax.plot(lim, lim, color="0.3", lw=1)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        unit = "Ω·m" if key == "res" else "mrad"
        name = "Resistivity" if key == "res" else "Phase F1"
        ax.set_xlabel(f"original model ({unit})")
        ax.set_ylabel(f"harmonized denoised model ({unit})")
        ax.set_title(f"{name}, resolved cells", fontsize=10, loc="left")
    fig.savefig(DEN / "fig_compare_variants.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

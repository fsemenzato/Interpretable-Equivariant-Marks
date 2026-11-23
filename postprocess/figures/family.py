import numpy as np
import matplotlib.pyplot as plt
from . import style as S

CMAP = "Blues"
PARS = [("s8", 4, "$\\sigma_8$"), ("Om", 0, "$\\Omega_m$")]


def heat(ax, cells, getter, title, norm):
    ds = sorted({c[0] for c in cells})
    ps = sorted({c[1] for c in cells})
    A = np.full((len(ds), len(ps)), np.nan)
    for (s, p), v in cells.items():
        A[ds.index(s), ps.index(p)] = getter(v)
    im = ax.imshow(A, origin="lower", cmap=CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([f"{s:g}" for s in ds])
    ax.set_xlabel("$p$")
    ax.tick_params(direction="out", top=False, right=False)
    S.panel_text(
        ax, title, "tr", bbox=dict(facecolor="white", edgecolor="none", pad=1.0)
    )
    return (im, A)


def render(data, out):
    z = np.load(data / "family.npz")
    F = {tuple(k): v for k, v in zip(z["cells"], z["fisher"])}
    G = {
        tuple(k): dict(zip(["s8", "Om"], v))
        for k, v in zip(z["cells"], z["regression"])
    }
    S.apply()
    fig = plt.figure(figsize=(7, 4.5))
    axes = []
    from matplotlib.colors import Normalize

    for row, (cells, y, title, get) in enumerate(
        [
            (F, 0.72, "Fisher", lambda v, i, p: v[i]),
            (G, 0.23, "Regression", lambda v, i, p: v[p]),
        ]
    ):
        vals = np.array([get(v, ip, p) for v in cells.values() for p, ip, _ in PARS])
        norm = Normalize(vmin=min(1.0, vals.min()), vmax=vals.max())
        for col, (p, ip, lab) in enumerate(PARS):
            z = fig.add_axes([0.075 + col * 0.48, y, 0.425, 0.24])
            im, A = heat(z, cells, lambda v: get(v, ip, p), lab, norm)
            z.set_xlabel("$p$", labelpad=2)
            z.tick_params(direction="in", top=True, right=True)
            if col == 0:
                z.set_ylabel("$\\delta_s$")
            else:
                z.tick_params(labelleft=False)
            axes.append(z)
        ca = fig.add_axes([0.225, y - 0.12, 0.62, 0.022])
        cb = fig.colorbar(im, cax=ca, orientation="horizontal")
        cb.ax.tick_params(direction="in", labelsize=8, pad=2)
        cb.set_label(
            "Fisher: $\\mathcal{I}_{\\alpha}(\\mathbf{S}_{\\rm C})$"
            if row == 0
            else "Regression: $\\mathcal{I}^{\\rm reg}_{\\alpha}(\\mathbf{S}_{\\rm C})$",
            fontsize=9,
            labelpad=3,
        )
    fig.savefig(out / "fig_classical_family_review.pdf")
    fig.savefig(out / "fig_classical_family_review.png", dpi=240)
    plt.close(fig)

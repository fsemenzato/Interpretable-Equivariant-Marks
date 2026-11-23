import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from . import style as S
from ..config_fish import PARAMS, FIDUCIAL

SLIM = 0.28
NTICKS = (5, 5)
CHI2 = {68: 2.2957, 95: 6.1801}
FID = (FIDUCIAL["Om"], FIDUCIAL["s8"])
GREEN = "#2CA02C"
STYLE = {
    "pk": ("$P_{\\delta\\delta}$", "#4C72B0", "-", 1.8, 2, True, (68, 95)),
    "pb": ("$P_{\\delta\\delta}+B$", "#DD8452", "--", 1.8, 3, True, (68, 95)),
    "learned": (
        "Hessian-Gated Mark, median training",
        GREEN,
        "-",
        2.8,
        4,
        True,
        (68, 95),
    ),
    "best": ("best training (68\\%)", GREEN, "-", 1.2, 6, False, (68,)),
    "worst": ("worst training (68\\%)", GREEN, (0, (3.5, 1.8)), 1.2, 6, False, (68,)),
    "pb04": (
        "$P_{\\delta\\delta}+B$, $k_{\\mathrm{max}}=0.4\\,h\\,\\mathrm{Mpc}^{-1}$",
        "#A0470C",
        (0, (5, 2)),
        1.5,
        5,
        False,
        (68, 95),
    ),
    "hm": ("Hessian Mark", GREEN, "-", 2.4, 4, True, (68, 95)),
}
DRAW = ("pk", "pb", "pb04", "learned", "best", "worst")
PAR = PARAMS


def contour_rc() -> None:
    mpl.rcParams.update(
        {
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
        }
    )


def _sym(a):
    a = np.asarray(a, float)
    return 0.5 * (a + a.T)


def ellipse_artists(ax, cov, key):
    lab, col, ls, lw, z, filled, levels = STYLE[key]
    vals, vecs = np.linalg.eigh(_sym(cov))
    o = vals.argsort()[::-1]
    vals, vecs = (np.clip(vals[o], 0.0, None), vecs[:, o])
    ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    for lev, a in ((95, 0.2), (68, 0.42)):
        if lev not in levels:
            continue
        w, h = (2.0 * np.sqrt(CHI2[lev] * vals[0]), 2.0 * np.sqrt(CHI2[lev] * vals[1]))
        if filled:
            ax.add_patch(
                Ellipse(FID, w, h, angle=ang, fc=col, ec="none", alpha=a, zorder=z)
            )
        ax.add_patch(
            Ellipse(
                FID, w, h, angle=ang, fc="none", ec=col, ls=ls, lw=lw, zorder=z + 0.5
            )
        )


def slim_triangle(fig, cell, covs, keys=DRAW, extent=3.5):
    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=cell,
        width_ratios=[1.0, SLIM],
        height_ratios=[SLIM, 1.0],
        wspace=0.05,
        hspace=0.05,
    )
    ax_top, ax_main, ax_right = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    )
    sd = {
        k: (
            float(np.sqrt(max(covs[k][0, 0], 0.0))),
            float(np.sqrt(max(covs[k][1, 1], 0.0))),
        )
        for k in keys
    }
    xlo = min((FID[0] - extent * sd[k][0] for k in keys))
    xhi = max((FID[0] + extent * sd[k][0] for k in keys))
    ylo = min((FID[1] - extent * sd[k][1] for k in keys))
    yhi = max((FID[1] + extent * sd[k][1] for k in keys))
    xg, yg = (np.linspace(xlo, xhi, 600), np.linspace(ylo, yhi, 600))
    for k in keys:
        lab, col, ls, lw, z, filled, levels = STYLE[k]
        ellipse_artists(ax_main, covs[k], k)
        px, py = (
            np.exp(-0.5 * ((xg - FID[0]) / sd[k][0]) ** 2),
            np.exp(-0.5 * ((yg - FID[1]) / sd[k][1]) ** 2),
        )
        ax_top.plot(xg, px, color=col, ls=ls, lw=min(lw, 1.6), zorder=z)
        ax_right.plot(py, yg, color=col, ls=ls, lw=min(lw, 1.6), zorder=z)
        if filled:
            ax_top.fill_between(xg, 0, px, color=col, alpha=0.22, zorder=z - 0.5)
            ax_right.fill_betweenx(yg, 0, py, color=col, alpha=0.22, zorder=z - 0.5)
    ax_main.set_xlim(xlo, xhi)
    ax_main.set_ylim(ylo, yhi)
    ax_main.axvline(FID[0], color="0.25", lw=0.8, ls=":", zorder=1)
    ax_main.axhline(FID[1], color="0.25", lw=0.8, ls=":", zorder=1)
    ax_main.set_xlabel(S.PARAMETER_LABELS["Om"])
    ax_main.set_ylabel(S.PARAMETER_LABELS["s8"])
    ax_main.xaxis.set_major_locator(MaxNLocator(NTICKS[0]))
    ax_main.yaxis.set_major_locator(MaxNLocator(NTICKS[1]))
    ax_main.tick_params(which="both", direction="in", top=True, right=True)
    ax_top.set_xlim(xlo, xhi)
    ax_top.set_ylim(0.0, 1.08)
    ax_top.axvline(FID[0], color="0.25", lw=0.8, ls=":")
    ax_top.tick_params(
        labelbottom=False, labelleft=False, which="both", direction="in", top=False
    )
    ax_top.set_yticks([])
    for sp in ("left", "right", "top"):
        ax_top.spines[sp].set_visible(False)
    ax_right.set_ylim(ylo, yhi)
    ax_right.set_xlim(0.0, 1.08)
    ax_right.axhline(FID[1], color="0.25", lw=0.8, ls=":")
    ax_right.tick_params(
        labelbottom=False,
        labelleft=False,
        which="both",
        direction="in",
        top=False,
        right=False,
        bottom=False,
        left=True,
    )
    ax_right.set_xticks([])
    for sp in ("bottom", "right", "top"):
        ax_right.spines[sp].set_visible(False)
    return ax_main


def widths(F, mode):
    fn = marg_cov if mode == "marginal" else cond_cov
    cv = {k: fn(v, PAIR) for k, v in F.items()}
    return (
        cv,
        {k: (float(np.sqrt(c[0, 0])), float(np.sqrt(c[1, 1]))) for k, c in cv.items()},
    )


PAIR = ["Om", "s8"]


def stable_inverse(a, rcond=1e-10):
    return np.linalg.pinv(_sym(a), rcond=rcond, hermitian=True)


def marginalized_subspace_fisher(F, subset):
    idx = [PAR.index(p) for p in subset]
    return stable_inverse(stable_inverse(F, rcond=1e-12)[np.ix_(idx, idx)], rcond=1e-12)


def fixed_subspace_fisher(F, subset):
    idx = [PAR.index(p) for p in subset]
    return _sym(np.asarray(F)[np.ix_(idx, idx)])


def marg_cov(F, pair):
    return stable_inverse(marginalized_subspace_fisher(F, pair), rcond=1e-12)


def cond_cov(F, pair):
    return stable_inverse(fixed_subspace_fisher(F, pair), rcond=1e-12)

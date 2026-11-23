import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from . import style as S


def render(data, out):
    with np.load(data / "hessian_profile.npz", allow_pickle=False) as z:
        profile = {k: z[k] for k in z.files if k not in ("_log", "_tag")}
    with np.load(data / "responses.npz", allow_pickle=False) as z:
        spectra = {k: z[k] for k in z.files}
    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W, 2.32))
    fig.subplots_adjust(left=0.065, right=0.991, bottom=0.29, top=0.97, wspace=0.37)
    a, b, c = axes
    d = profile
    lo, hi = (float(d["lo"].min()), float(d["hi"].max()))
    a.bar(
        0.5 * (d["he"][1:] + d["he"][:-1]),
        d["hist"] / d["hist"].max() * 0.35 * (hi - lo),
        width=np.diff(d["he"]),
        bottom=lo,
        color=".85",
        lw=0,
        zorder=0,
    )
    a.fill_between(d["xc"], d["lo"], d["hi"], color="darkred", alpha=0.2, lw=0)
    a.plot(d["xc"], d["mu"], color="darkred", lw=1.8)
    a.set(
        xlim=(d["xc"].min(), d["xc"].max()),
        ylim=(lo, hi + 0.03 * (hi - lo)),
        xlabel="$\\delta_{\\mathrm f}$",
        ylabel="$M$",
    )
    a.xaxis.set_major_locator(MaxNLocator(4))
    k = spectra["k0"]
    b.loglog(k, spectra["Pdd0"], color="black", ls="--", lw=1.5)
    b.loglog(k, spectra["Pmm0"], color="darkred", lw=1.9)
    b.loglog(k, np.abs(spectra["Pdm0"]), color=".45", ls=":", lw=1.6)
    b.set(ylim=(200.0, 30000.0), ylabel="$P(k)$ [$(h^{-1}\\mathrm{Mpc})^3$]")
    colors = {"s8": "#0072B2", "Om": "#E69F00"}
    for par in ("Om", "s8"):
        for q, ls, lw in [("Pdd", "--", 1.5), ("Pmm", "-", 1.9)]:
            c.fill_between(
                k,
                spectra[f"B_{par}_{q}"][0],
                spectra[f"B_{par}_{q}"][1],
                color=colors[par],
                alpha=0.22,
                lw=0,
            )
            c.semilogx(k, spectra[f"D_{par}_{q}0"], color=colors[par], ls=ls, lw=lw)
    c.axhline(0, color="k", lw=0.6)
    c.set(ylim=(-8.6, 4.4), ylabel="$\\partial\\ln P/\\partial\\theta$")
    c.legend(
        [Line2D([], [], color=colors[p], lw=2.2) for p in ("s8", "Om")],
        ["$\\theta=\\sigma_8$", "$\\theta=\\Omega_m$"],
        loc="lower right",
        handlelength=1.4,
        labelspacing=0.2,
    )
    for ax in (b, c):
        ax.set(xlim=(0.01, 0.3), xlabel="$k$ [$h\\,\\mathrm{Mpc}^{-1}$]")
    for ax in axes:
        S.all_ticks(ax)
        ax.tick_params(
            axis="both", which="minor", bottom=False, top=False, left=False, right=False
        )
    fig.legend(
        [
            Line2D([], [], color="black", ls="--"),
            Line2D([], [], color="darkred"),
            Line2D([], [], color=".45", ls=":"),
        ],
        [
            "$P_{\\delta\\delta}$",
            "$P_{\\Delta\\Delta}$, Hessian Mark",
            "$|P_{\\delta\\Delta}|$, Hessian Mark",
        ],
        loc="lower center",
        bbox_to_anchor=((b.get_position().x0 + c.get_position().x1) / 2, 0.015),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.7,
    )
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_mark_profile_spectra.{ext}", dpi=220)
    print("Saved spectrum-response figure.")

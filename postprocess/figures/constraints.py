import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from . import style as S, contours as FC
from ..config_fish import PARAMS


def render(data, out):
    PAR = PARAMS
    z = np.load(data / "constraints.npz")
    k = z["kgrid"]
    ks = z["kscan"]
    pbs = z["pb_scan"]
    hm = int(z["hm_index"])
    web = int(z["web_index"])
    mat = {key: z["F_" + key] for key in ["pk", "pb", "hm"]}
    colors = {
        "pk": "#4C72B0",
        "pb": "#DD8452",
        "classical": "#666666",
        "hm": FC.GREEN,
        "web": "#CC79A7",
    }
    styles = {"pk": "-", "pb": "--", "classical": "-.", "hm": (0, (9, 2.5)), "web": ":"}
    markers = {"pk": "o", "pb": None, "classical": None, "hm": None, "web": "s"}
    for key in ["pk", "pb", "hm"]:
        old = FC.STYLE[key]
        FC.STYLE[key] = (
            old[0],
            colors[key],
            styles[key],
            old[3],
            old[4],
            old[5],
            old[6],
        )
    FC.STYLE["web"] = (
        "Web Mark",
        colors["web"],
        styles["web"],
        1.7,
        5,
        False,
        (68, 95),
    )
    triple = "$\\{P_{\\delta\\delta},P_{\\delta\\Delta},P_{\\Delta\\Delta}\\}$"
    labels = {
        "pk": "$P_{\\delta\\delta}$",
        "pb": "$P+B$",
        "classical": "Best classical ($R=10$)",
        "hm": "Hessian Mark",
        "web": "Web Mark",
    }
    S.apply()
    FC.contour_rc()
    variants = [("fig_constraints_classical_fom", False, False)]
    for name, preview, split in variants:
        classical_key = (
            "classical_fom" if name.endswith("_fom") else "classical_selected"
        )
        labels["classical"] = "Classical $R=10$ (" + (
            "FoM-best)" if name.endswith("_fom") else "$\\sigma_8$-best)"
        )
        fig = plt.figure(figsize=(S.FIG_W, 3.9))
        outer = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.25, 1],
            left=0.075,
            right=0.985,
            bottom=0.24,
            top=0.97,
            wspace=0.34,
        )
        keys = ("pk", "pb", "hm", "web") if preview else ("pk", "pb", "hm")
        covs, widths = FC.widths({key: mat[key] for key in keys}, "marginal")
        ax = FC.slim_triangle(fig, outer[0, 0], covs, keys=keys, extent=3.3)
        S.panel_text(
            ax,
            "$k_{\\mathrm{max}}=0.3\\,h\\,\\mathrm{Mpc}^{-1}$" + "\n" + "marginalized",
            "tr",
            multialignment="right",
        )
        right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, 1], hspace=0
        )
        axes = []
        for row, ip in enumerate([0, 4]):
            a = fig.add_subplot(right[row, 0], sharex=axes[0] if axes else None)
            data = {
                "pk": z["pk"],
                "classical": z[classical_key],
                "hm": z["free"][hm],
                "web": z["web"][web],
            }
            m = (k >= 0.05 - 1e-09) & (k <= 0.3 + 1e-09)
            for key in ["pk", "classical", "web", "hm"]:
                a.plot(
                    k[m],
                    data[key][m, ip],
                    color=colors[key],
                    ls=styles[key],
                    lw=2 if key == "hm" else 1.5,
                    marker=markers[key],
                    markevery=12,
                    ms=3,
                    mfc="white",
                    mew=0.8,
                )
            m = (ks >= 0.05 - 1e-09) & (ks <= 0.3 + 1e-09)
            a.plot(ks[m], pbs[m, ip], color=colors["pb"], ls=styles["pb"], lw=1.8)
            a.set_yscale("log")
            a.set_xlim(0.05, 0.3)
            a.set_ylabel("$\\sigma(" + S.PARAMETER_LABELS[PAR[ip]][1:-1] + ")$")
            a.tick_params(which="both", direction="in", top=True, right=True)
            axes.append(a)
        axes[0].tick_params(labelbottom=False)
        axes[1].set_xlabel(S.KMAX_LABEL)

        def legend(group, anchor, ncol=1):
            handles = [
                Line2D(
                    [],
                    [],
                    color=colors[key],
                    ls=styles[key],
                    lw=1.8,
                    marker=markers[key],
                    ms=3,
                    mfc="white",
                    mew=0.8,
                )
                for key in group
            ]
            return fig.legend(
                handles,
                [labels[key] for key in group],
                loc="lower center",
                bbox_to_anchor=anchor,
                ncol=ncol,
                fontsize=8,
                columnspacing=1.6,
                handlelength=4.0,
                labelspacing=0.5,
            )

        if split:
            lbox = outer[0, 0].get_position(fig)
            rbox = outer[0, 1].get_position(fig)
            legend(["pk", "pb", "hm"], ((lbox.x0 + lbox.x1) / 2, 0.012))
            legend(
                ["pk", "pb", "classical", "hm", "web"], ((rbox.x0 + rbox.x1) / 2, 0.012)
            )
        else:
            legend(["pk", "pb", "classical", "hm", "web"], (0.53, 0.01), ncol=3)
        fig.savefig(out / (name + ".pdf"))
        fig.savefig(out / (name + ".png"), dpi=240)
        plt.close(fig)

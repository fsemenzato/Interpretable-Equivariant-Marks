from __future__ import annotations
from pathlib import Path
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT = Path.cwd()
FIG = ROOT / "figures"
LOGS = FIG / "logs"
RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": "\\usepackage{amsmath}\\usepackage{bm}",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
}
PANEL_TEXT_SIZE = 9
FIG_W = 510.0 / 72.27
RASTER_DPI = 300
PNG_DPI = 150
STYLE = {
    "pk": dict(color="black", ls="--", marker=None, label="$P_{\\delta\\delta}$"),
    "classical": dict(color="grey", ls="-.", marker="^", label="Classical Mark"),
    "free": dict(color="darkred", ls="-", marker="o", label="Hessian-Gated Mark"),
    "web": dict(color="darkblue", ls="-", marker="s", label="Web Mark"),
    "pb": dict(color="tab:orange", ls=":", marker="D", label="$P_{\\delta\\delta}+B$"),
    "logdens": dict(
        color="tab:purple",
        ls=(0, (3, 1, 1, 1)),
        marker="v",
        label="log-density spectrum",
    ),
    "grad": dict(
        color="tab:brown", ls=(0, (5, 1.5)), marker="P", label="gradient-energy mark"
    ),
    "curv": dict(
        color="tab:cyan", ls=(0, (1, 1)), marker="X", label="curvature-energy mark"
    ),
}
COLORS = {k: v["color"] for k, v in STYLE.items()}
COLORS.update({"gray": "0.45", "freemark": STYLE["free"]["color"]})
LINESTYLES = {k: v["ls"] for k, v in STYLE.items()}
MARKERS = {k: v["marker"] for k, v in STYLE.items()}
PARAMETER_LABELS = {
    "Om": "$\\Omega_m$",
    "Ob": "$\\Omega_b$",
    "h": "$h$",
    "ns": "$n_s$",
    "s8": "$\\sigma_8$",
}
KMAX_LABEL = "$k_{\\mathrm{max}}$ [$h\\,\\mathrm{Mpc}^{-1}$]"


def apply() -> None:
    mpl.rcParams.update(RC)


def panel_text(
    ax, text: str, loc: str = "tr", size: float = PANEL_TEXT_SIZE, **kw
) -> None:
    x, y, ha, va = {
        "tr": (0.95, 0.95, "right", "top"),
        "tl": (0.05, 0.95, "left", "top"),
        "br": (0.95, 0.05, "right", "bottom"),
        "bl": (0.05, 0.05, "left", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, ha=ha, va=va, **kw)


def all_ticks(ax) -> None:
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def common_legend(
    fig,
    handles,
    labels,
    ncol: int | None = None,
    y: float = -0.02,
    loc: str = "lower center",
):
    return fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol or len(handles),
        bbox_to_anchor=(0.5, y),
        frameon=False,
    )


def stacked(n: int, width: float = 3.6, height_per: float = 2.6, sharex: bool = True):
    fig, axes = plt.subplots(
        n,
        1,
        sharex=sharex,
        figsize=(width, height_per * n),
        gridspec_kw={"hspace": 0.0},
    )
    return (fig, [axes] if n == 1 else list(axes))


def row_of(n: int, width: float = 7.0, height: float = 2.6, sharey: bool = False):
    fig, axes = plt.subplots(
        1,
        n,
        sharey=sharey,
        figsize=(width, height),
        gridspec_kw={"wspace": 0.0 if sharey else 0.28},
    )
    return (fig, [axes] if n == 1 else list(axes))


def kmax_axis(ax, step: float = 0.05) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=RASTER_DPI)
    fig.savefig(FIG / f"{name}.png", dpi=PNG_DPI)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")

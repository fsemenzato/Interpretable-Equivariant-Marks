from .maps import S, TwoSlopeNorm, image_with_bar, inch_axes, np, plt, save

HEIGHT = None
PCT = (0.5, 99.5)
CMAP_DENS, CMAP_EIG = ("RdBu_r", "plasma")
N_BAR_TICKS = 4
LABELS = ["$\\ln(1+\\delta)$", "$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$"]
SCALEBAR = (500, "$500\\,h^{-1}\\mathrm{Mpc}$")


def render(data, out):
    NAME = "fig_tidal_eigs"
    data = dict(np.load(data / "tidal.npz"))
    fields = [data["ld"], *data["lam"]]
    W = S.FIG_W
    PW, GAP = (1.6, 0.21)
    H = HEIGHT or 0.04 + PW + 0.2 + 0.08 + 0.36
    fig = plt.figure(figsize=(W, H))
    x_left = (W - 4 * PW - 3 * GAP) / 2
    y0 = 0.36 + 0.08 + 0.2 - 0.0
    for j, (f, lab) in enumerate(zip(fields, LABELS)):
        x0 = x_left + j * (PW + GAP)
        ax = inch_axes(fig, (x0, H - 0.04 - PW, PW, PW), W, H)
        cax = inch_axes(fig, (x0, H - 0.04 - PW - 0.2 - 0.08, PW, 0.08), W, H)
        lo, hi = np.percentile(f, PCT)
        if j == 0:
            image_with_bar(
                fig,
                ax,
                cax,
                f,
                lab,
                CMAP_DENS,
                norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
                extent=[0, 1000, 0, 1000],
                nticks=N_BAR_TICKS,
            )
            if SCALEBAR:
                from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
                from matplotlib.font_manager import FontProperties

                sb = AnchoredSizeBar(
                    ax.transData,
                    SCALEBAR[0],
                    SCALEBAR[1],
                    loc="lower left",
                    pad=0.25,
                    borderpad=0.35,
                    sep=2.5,
                    color="k",
                    frameon=True,
                    size_vertical=8,
                    label_top=True,
                    fontproperties=FontProperties(size=8),
                )
                sb.patch.set_facecolor("white")
                sb.patch.set_edgecolor("none")
                ax.add_artist(sb)
        else:
            image_with_bar(
                fig,
                ax,
                cax,
                f,
                lab,
                CMAP_EIG,
                vmin=lo,
                vmax=hi,
                extent=[0, 1000, 0, 1000],
                nticks=N_BAR_TICKS,
            )
    save(fig, NAME)

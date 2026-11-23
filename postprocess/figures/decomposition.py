from .maps import (
    ENVS,
    ENV_SUB,
    Normalize,
    OUTSIDE,
    S,
    frame_only,
    image_bar,
    inch_axes,
    np,
    plt,
    save_map,
    signed_norm,
    tick_label,
)

MG = 0.07
GL = 0.38
GX = 0.08
BAR = 0.075
BAR_GAP = 0.05
BLK_MID = 0.47
BLK_BOT = 0.43
TOP = 0.19


def limits(v, q=(1, 99)):
    lo, hi = np.percentile(v, q)
    return (float(lo), float(hi))


def geometry():
    W = S.FIG_W
    P = (W - 2 * MG - GL - 3 * GX) / 5.0
    H = BLK_BOT + P + BLK_MID + P + TOP
    x_cols = [MG + P + GL + j * (P + GX) for j in range(4)]
    return (W, H, P, P, x_cols)


def make(
    name,
    ld,
    w,
    prod,
    titles,
    log,
    pct=(1, 99),
    cmap_dens="RdBu_r",
    cmap_w="plasma",
    outside=None,
    mark=None,
    w_label="$w_e$",
    prod_labels=None,
    w_ticks=(0.0, 0.5, 1.0),
    ln_label="$\\ln(1+\\delta)$",
    title_size=9,
):
    W, H, P, T, x_cols = geometry()
    fig = plt.figure(figsize=(W, H))
    y2 = BLK_BOT
    y1 = y2 + P + BLK_MID
    bar = lambda x, y, width: inch_axes(fig, (x, y - BAR_GAP - BAR, width, BAR), W, H)
    prod_labels = prod_labels or [
        f"$w_{{{ENV_SUB[e]}}}\\,h_{{{ENV_SUB[e]}}}$" for e in ENVS
    ]
    ax = inch_axes(fig, (MG, y1, P, P), W, H)
    lo, hi = limits(ld, pct)
    norm, cm, kind = signed_norm(lo, hi, cmap_dens)
    image_bar(fig, ax, bar(MG, y1, P), ld, cm, norm, ln_label, lo=lo, hi=hi)
    log(
        f"panel ln(1+delta): colour range [{lo:+.3f}, {hi:+.3f}] ({pct[0]}-{pct[1]} percentile of the slice), {kind} map, symmetric limits +-{max(-lo, hi):.3f}"
    )
    if mark is not None:
        ax = inch_axes(fig, (MG, y2, P, P), W, H)
        lo, hi = limits(mark, pct)
        image_bar(
            fig,
            ax,
            bar(MG, y2, P),
            mark,
            "plasma",
            Normalize(lo, hi),
            "$M$",
            lo=lo,
            hi=hi,
        )
        log(
            "Complete mark shown below complete log-density; gap separates both fields from their components."
        )
    for j in range(4):
        ax = inch_axes(fig, (x_cols[j], y1, P, P), W, H)
        im = ax.imshow(
            w[j],
            origin="lower",
            cmap=cmap_w,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        frame_only(ax)
        ax.set_title(titles[j], fontsize=title_size, pad=3)
        log(
            f"panel w_e ({titles[j]}): slice min {w[j].min():.3f} max {w[j].max():.3f}; colour range [0, 1] (shared bar)"
        )
    cax = bar(x_cols[0], y1, x_cols[3] + P - x_cols[0])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.8)
    cax.tick_params(axis="x", direction="in", length=2.5, labelsize=8, pad=2, top=False)
    cb.set_ticks(list(w_ticks))
    cb.set_ticklabels([tick_label(t) for t in w_ticks])
    cax.minorticks_off()
    cb.set_label(w_label, fontsize=9, labelpad=1.5)
    for j in range(4):
        ax = inch_axes(fig, (x_cols[j], y2, P, P), W, H)
        v = np.asarray(prod[j], dtype=float)
        inside = (
            np.ones_like(v, bool) if outside is None else ~np.asarray(outside[j], bool)
        )
        lo, hi = limits(v[inside], pct)
        norm, cm, kind = signed_norm(lo, hi)
        img = np.where(inside, v, np.nan)
        image_bar(
            fig,
            ax,
            bar(x_cols[j], y2, P),
            img,
            cm,
            norm,
            prod_labels[j],
            lo=lo,
            hi=hi,
            bad=OUTSIDE,
        )
        log(
            f"panel w_e h_e ({titles[j]}): colour range [{lo:+.3f}, {hi:+.3f}] ({pct[0]}-{pct[1]} percentile of the {int(inside.sum())} {('in-environment ' if outside is not None else '')}slice values), {kind} map"
            + (f", symmetric limits +-{max(-lo, hi):.3f}" if kind == "div" else "")
        )
    save_map(fig, name)
    log(f"figure size {W:.3f} x {H:.3f} in; panel {P:.3f} in (five equal columns)")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from . import style as S

from marks._mark_base import ENVIRONMENTS as ENVS

ENV_SUB = {e: "\\mathrm{" + e + "}" for e in ENVS}
OUTSIDE = "0.84"


def check_print(fig, name: str, min_pt: float = 8.0):
    from matplotlib.text import Text

    assert (
        abs(fig.get_figwidth() - S.FIG_W) < 1e-06
    ), f"{name}: width {fig.get_figwidth()} in != S.FIG_W"
    fig.canvas.draw()
    sizes = [
        (t.get_fontsize(), t.get_text())
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
    ]
    small = [x for x in sizes if x[0] < min_pt - 1e-09]
    assert not small, f"{name}: text below {min_pt} pt: {small[:5]}"
    print(
        f"[print check] {name}: {fig.get_figwidth():.3f} x {fig.get_figheight():.3f} in, {len(sizes)} texts, smallest {min((x[0] for x in sizes)):.1f} pt"
    )


def save(fig, name: str):
    check_print(fig, name)
    S.save(fig, name)


def save_map(fig, name: str):
    check_print(fig, name)
    S.FIG.mkdir(exist_ok=True)
    fig.savefig(S.FIG / f"{name}.pdf", dpi=S.RASTER_DPI)
    fig.savefig(S.FIG / f"{name}.png", dpi=S.RASTER_DPI)
    plt.close(fig)
    print("wrote", S.FIG / f"{name}.pdf")


def tick_label(v: float, ticks=None) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    if s in ("-0", "0", ""):
        s = "0"
    return f"${s}$"


def label_width_in(text: str, size: float = 8.0) -> float:
    em = size / 72.27
    return sum(
        (
            0.5 * em
            if c.isdigit()
            else 0.778 * em
            if c == "-"
            else 0.278 * em
            if c == "."
            else 0.0
            for c in text
        )
    )


def round_ticks(
    lo: float,
    hi: float,
    target: int = 3,
    length_in: float | None = None,
    size: float = 8.0,
    overhang_in: float = 0.02,
    min_gap_in: float = 0.06,
    force=None,
):
    if force is not None:
        return list(force)
    span = hi - lo
    straddle = lo < 0 < hi
    best = None
    for e in range(-3, 4):
        for m in (1.0, 2.0, 2.5, 5.0):
            st = m * 10.0**e
            t = np.arange(np.ceil(lo / st - 1e-09), np.floor(hi / st + 1e-09) + 1) * st
            t[np.abs(t) < 1e-09 * max(1.0, abs(st))] = 0.0
            if length_in is not None and len(t):
                pos = (t - lo) / span * length_in
                hw = np.array(
                    [0.5 * label_width_in(tick_label(v).strip("$"), size) for v in t]
                )
                ok = (pos - hw >= -overhang_in) & (pos + hw <= length_in + overhang_in)
                t, pos, hw = (t[ok], pos[ok], hw[ok])
                if len(t) > 1 and np.min(np.diff(pos) - hw[1:] - hw[:-1]) < min_gap_in:
                    continue
            else:
                t = t[(t >= lo + 0.08 * span) & (t <= hi - 0.08 * span)]
            if len(t) == 0:
                continue
            cands = [t] + ([t[1:], t[:-1]] if len(t) > target + 1 else [])
            for tt in cands:
                n = len(tt)
                has0 = straddle and np.any(tt == 0.0)
                off = abs(0.5 * (tt[0] + tt[-1]) - 0.5 * (lo + hi)) / span
                score = (
                    abs(n - target)
                    - (0.1 if n == target + 1 else 0.0)
                    + (0.0 if has0 or not straddle else 1.5)
                    + (0.95 if m == 2.5 else 0.0)
                    + 0.05 * off
                )
                if (
                    best is None
                    or score < best[0] - 1e-09
                    or (abs(score - best[0]) < 1e-09 and st > best[1])
                ):
                    best = (score, st, tt)
    return [float(v) for v in best[2]]


def signed_norm(
    lo: float, hi: float, cmap_div="RdBu_r", cmap_neg="Blues_r", cmap_pos="Reds"
):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl

    if lo < 0 < hi:
        L = max(-lo, hi)
        return (Normalize(-L, L), mpl.colormaps[cmap_div], "div")
    base = mpl.colormaps[cmap_neg if hi <= 0 else cmap_pos]
    if hi <= 0:
        cm = LinearSegmentedColormap.from_list(
            cmap_neg + "_trunc", base(np.linspace(0.0, 0.88, 256))
        )
    else:
        cm = LinearSegmentedColormap.from_list(
            cmap_pos + "_trunc", base(np.linspace(0.12, 1.0, 256))
        )
    return (Normalize(lo, hi), cm, "seq")


def image_bar(
    fig,
    ax,
    cax,
    img,
    cmap,
    norm,
    label=None,
    ticks=None,
    lo=None,
    hi=None,
    bad=None,
    label_size=9,
):
    import matplotlib as mpl

    cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).copy()
    if bad is not None:
        cm.set_bad(bad)
    im = ax.imshow(img, origin="lower", cmap=cm, norm=norm, interpolation="nearest")
    frame_only(ax)
    lo = norm.vmin if lo is None else lo
    hi = norm.vmax if hi is None else hi
    cb = fig.colorbar(
        im, cax=cax, orientation="horizontal", boundaries=np.linspace(lo, hi, 257)
    )
    cb.outline.set_linewidth(0.8)
    cax.tick_params(axis="x", direction="in", length=2.5, labelsize=8, pad=2, top=False)
    ticks = (
        round_ticks(lo, hi, length_in=cax.get_position().width * fig.get_figwidth())
        if ticks is None
        else ticks
    )
    cb.set_ticks(ticks)
    cb.set_ticklabels([tick_label(t) for t in ticks])
    cax.minorticks_off()
    if label:
        cb.set_label(label, fontsize=label_size, labelpad=1.5)
    return (im, cb)


def panel_label(ax, s, **kw):
    S.panel_text(ax, s, "tr", **kw)


def inch_axes(fig, rect_in, W, H):
    x, y, w, h = rect_in
    return fig.add_axes([x / W, y / H, w / W, h / H])


def frame_only(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def image_with_bar(
    fig,
    ax,
    cax,
    img,
    label,
    cmap,
    norm=None,
    ticks=None,
    extent=None,
    vmin=None,
    vmax=None,
    nticks=4,
):
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    im = ax.imshow(img, origin="lower", cmap=cmap, extent=extent, **kw)
    frame_only(ax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.8)
    cax.tick_params(axis="x", direction="in", length=2.5, labelsize=8, pad=2, top=False)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        from matplotlib.ticker import MaxNLocator

        cb.locator = MaxNLocator(nbins=nticks, prune="both")
        cb.update_ticks()
    if label:
        cb.set_label(label, fontsize=9, labelpad=1)
    return (im, cb)

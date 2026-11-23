import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle

styles = {
    "expert": ("#fff3d3", "#d3aa46"),
    "gate": ("#e8f2df", "#83af68"),
    "fixed": ("#e8f0fa", "#7da3d1"),
    "learned": ("#eee7f7", "#a28bc7"),
    "sum": ("#edf2f8", "#8da4c0"),
}
wire = "#818181"
ink = "#333333"


class Panel:
    def __init__(self, out):
        self.out = out
        self.fig = plt.figure(figsize=(3.44, 3.52))
        self.a = self.fig.add_axes([0, 0, 1, 1])
        self.a.set(xlim=(0, 3.44), ylim=(0, 3.52), aspect="equal")
        self.a.axis("off")
        self.check = []

    def text(self, x, y, t, fs=9):
        return self.a.text(x, y, t, ha="center", va="center", fontsize=fs, color=ink)

    def line(self, pts, **kw):
        self.a.plot(*np.array(pts).T, color=wire, lw=0.8, **kw)

    def arrow(self, p, q):
        self.a.annotate(
            "",
            xy=q,
            xytext=p,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=0.8,
                color=wire,
                mutation_scale=7,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def box(self, x, y, w, h, t, kind):
        fc, ec = styles[kind]
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.055",
            facecolor=fc,
            edgecolor=ec,
            lw=0.85,
        )
        self.a.add_patch(p)
        self.check.append((self.text(x, y, t), p))

    def mult(self, x, y):
        self.a.add_patch(
            Circle((x, y), 0.072, facecolor="white", edgecolor=wire, lw=0.8)
        )
        self.text(x, y, "$\\times$", 10)

    def funnel(self, x, y, t, kind, dashed=False):
        fc, ec = styles[kind]
        self.a.add_patch(
            Polygon(
                [
                    (x - 0.34, y + 0.2),
                    (x + 0.34, y + 0.2),
                    (x + 0.12, y - 0.2),
                    (x - 0.12, y - 0.2),
                ],
                facecolor=fc,
                edgecolor=ec,
                lw=0.9,
                linestyle=(0, (3, 2)) if dashed else "-",
            )
        )
        self.text(x, y + 0.02, t)

    def tokens(self, x, y, kind):
        fc, ec = styles[kind]
        for j, alpha in enumerate([0.45, 0.7, 1.0]):
            self.a.add_patch(
                Rectangle(
                    (x - 0.038, y - 0.115 + j * 0.083),
                    0.076,
                    0.066,
                    facecolor=ec,
                    edgecolor="white",
                    lw=0.3,
                    alpha=alpha,
                )
            )

    def save(self, name, production=True):
        self.fig.canvas.draw()
        rr = self.fig.canvas.get_renderer()
        for t, p in self.check:
            b = t.get_window_extent(rr)
            c = p.get_window_extent(rr)
            assert c.contains(b.x0, b.y0) and c.contains(b.x1, b.y1), t.get_text()
        dest = self.out
        self.fig.savefig(dest / (name + ".pdf"))
        self.fig.savefig(self.out / (name + ".png"), dpi=240)


def marker(out, data, thumbnail=False):
    d = Panel(out)
    if thumbnail:
        ld = np.load(data / "hessian_decomposition.npz")["ld"]
        lo, hi = np.percentile(ld, [1, 99])
        v = max(abs(lo), abs(hi))
        d.a.imshow(
            ld[32:96, 32:96],
            extent=(1.44, 1.96, 2.96, 3.48),
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        d.a.add_patch(
            Rectangle((1.44, 2.96), 0.52, 0.52, fill=False, edgecolor="#aaaaaa", lw=0.5)
        )
        d.text(2.55, 3.25, "$\\ln(1+\\delta)$")
        d.line([(1.7, 2.96), (1.7, 2.94)])
    else:
        d.text(1.7, 3.28, "$\\ln(1+\\delta)$", 10)
        d.line([(1.7, 3.13), (1.7, 2.94)])
    d.line([(1.7, 2.94), (0.73, 2.94)])
    d.line([(1.7, 2.94), (2.49, 2.94)])
    d.arrow((0.73, 2.94), (0.73, 2.88))
    d.arrow((2.49, 2.94), (2.49, 2.88))
    d.box(0.73, 2.71, 1.16, 0.31, "Filter $G(k)$", "expert")
    d.box(2.49, 2.71, 1.1, 0.31, "Tensor $\\to\\lambda_i$", "fixed")
    d.arrow((0.73, 2.54), (0.73, 2.37))
    d.text(0.93, 2.44, "$\\delta_{\\mathrm f}$", 9)
    d.arrow((2.49, 2.54), (2.49, 2.37))
    d.text(0.73, 2.26, "Nonlinear experts", 8)
    d.text(2.49, 2.26, "Gates", 8)
    for y, e in zip([2.04, 1.66, 1.28, 0.9], ["void", "sheet", "filament", "cluster"]):
        d.box(
            0.73, y, 1.12, 0.25, "$h_{\\rm " + e + "}(\\delta_{\\mathrm f})$", "expert"
        )
        d.box(2.49, y, 1.06, 0.25, "$w_{\\rm " + e + "}$", "gate")
        d.arrow((1.302, y), (1.588, y))
        d.arrow((1.948, y), (1.732, y))
        d.mult(1.66, y)
        d.line([(1.66, y - 0.072), (1.66, y - 0.175), (3.28, y - 0.175)])
        d.a.add_patch(
            Circle((3.28, y - 0.175), 0.014, facecolor=wire, edgecolor="none")
        )
    d.line([(3.28, 1.865), (3.28, 0.49)])
    d.arrow((3.28, 0.49), (3.142, 0.49))
    d.box(1.7, 0.49, 2.86, 0.3, "$M=b+\\sum_e w_e h_e(\\delta_{\\mathrm f})$", "sum")
    d.arrow((1.7, 0.325), (1.7, 0.225))
    d.text(1.7, 0.12, "$\\Delta$", 12)
    d.save(
        "fig_marker_panel_slice_trial" if thumbnail else "fig_marker_panel_prd",
        not thumbnail,
    )
    return d


def render(data, out):
    marker(out, data)
    d = Panel(out)
    d.text(1.06, 3.28, "$\\delta$", 11)
    d.line([(1.06, 3.13), (1.06, 3.06), (0.49, 3.06)])
    d.arrow((0.49, 3.06), (0.49, 2.57))
    d.line([(1.06, 3.06), (1.62, 3.06)])
    d.arrow((1.62, 3.06), (1.62, 2.97))
    d.box(1.62, 2.8, 0.83, 0.3, "Mark", "expert")
    d.arrow((1.62, 2.635), (1.62, 2.57))
    d.text(0.49, 2.43, "$P_{\\delta\\delta}$")
    d.text(1.62, 2.43, "$P_{\\Delta\\Delta}/P_{\\delta\\delta},\\ r$")
    d.text(2.86, 2.64, "Cosmology")
    d.text(2.86, 2.43, "$\\boldsymbol\\theta_j$", 10)
    for x, label, kind, dash in [
        (0.49, "$f_\\delta$", "fixed", True),
        (1.62, "$f_M$", "learned", False),
        (2.86, "$f_\\theta$", "learned", False),
    ]:
        d.funnel(x, 2.08, label, kind, dash)
        d.arrow((x, 1.86), (x, 1.74))
        d.tokens(x, 1.6, kind)
    d.text(0.7, 1.6, "$\\mathbf z_\\delta$")
    d.text(1.84, 1.6, "$\\mathbf z_M$")
    d.text(3.12, 1.6, "$\\mathbf z_{\\theta,j}$")
    d.line([(0.49, 1.46), (0.49, 1.24)])
    d.arrow((0.49, 1.24), (0.94, 1.04))
    d.line([(1.62, 1.46), (1.62, 1.24)])
    d.arrow((1.62, 1.24), (1.18, 1.04))
    d.text(1.06, 0.98, "$\\mathbf z_\\perp$", 12)
    d.arrow((1.06, 0.84), (1.06, 0.7))
    d.box(1.06, 0.53, 1.84, 0.31, "InfoNCE", "learned")
    d.line([(2.86, 1.46), (2.86, 0.53)])
    d.arrow((2.86, 0.53), (1.992, 0.53))
    d.text(0.49, 0.14, "Embedders:", 8)
    d.a.plot([0.99, 1.27], [0.14, 0.14], color=styles["learned"][1], lw=1.1)
    d.text(1.57, 0.14, "Trained", 8)
    d.a.plot(
        [2.05, 2.33], [0.14, 0.14], color=styles["fixed"][1], lw=1.1, ls=(0, (3, 2))
    )
    d.text(2.63, 0.14, "Frozen", 8)
    d.save("fig_contrastive_panel_prd")
    print("Saved schematic panels.")

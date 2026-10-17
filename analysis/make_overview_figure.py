"""Figure 1 of the paper: GT3MoE overview as pseudo-3D blocks, in the style of the draw.io sketch.

Writes $GT3MOE_PAPER_DIR/figures/fig_overview.pdf and a PNG preview in $GT3MOE_PAPER_DIR/previews
(GT3MOE_PAPER_DIR defaults to ./paper_artifacts at the repository root).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

_ROOT = os.environ.get("GT3MOE_PAPER_DIR",
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "paper_artifacts"))
OUT = os.path.join(_ROOT, "figures", "fig_overview.pdf")
PREVIEW = os.path.join(_ROOT, "previews", "fig_overview.png")

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "stix", "pdf.fonttype": 42})

# draw.io palette: (front fill, edge)
RED = ("#f8cecc", "#b85450")
PURPLE = ("#e1d5e7", "#9673a6")
YELLOW = ("#ffe599", "#d6b656")
GREY = ("#f2f2f2", "#b3b3b3")
BLUE = ("#dae8fc", "#6c8ebf")
GREEN = ("#d5e8d4", "#82b366")
TAG = ("#fff2cc", "#d6b656")
D = 1.3  # depth of the 3D faces


def mix(c, other, t):
    a, b = to_rgb(c), to_rgb(other)
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))


def cube(ax, x, y, w, h, pal, lw=1.0, alpha=1.0, z=3):
    fill, edge = pal
    top = [(x, y + h), (x + D, y + h + D), (x + w + D, y + h + D), (x + w, y + h)]
    side = [(x + w, y), (x + w + D, y + D), (x + w + D, y + h + D), (x + w, y + h)]
    kw = dict(edgecolor=edge, lw=lw, alpha=alpha, zorder=z, joinstyle="round")
    ax.add_patch(Polygon(top, closed=True, facecolor=mix(fill, "white", 0.45), **kw))
    ax.add_patch(Polygon(side, closed=True, facecolor=mix(fill, edge, 0.35), **kw))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, **kw))


def text(ax, x, y, s, size=8, bold=False, rot=0, color="black", ha="center", va="center", z=6, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", rotation=rot, color=color,
            ha=ha, va=va, zorder=z, fontstyle=style)


def line(ax, pts, color="black", lw=1.0, ls="-", arrow=True, z=1):
    xs, ys = zip(*pts)
    if arrow and len(pts) > 1:
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=9, linestyle=ls))
    else:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z)


def region(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fill=False,
                                edgecolor="#9a9a9a", lw=0.9, ls=(0, (4, 3)), zorder=0))
    text(ax, x + w / 2, y + h - 1.6, title, size=8, bold=True, color="#333333")


def tag(ax, x, y, w, h, s, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor=TAG[0], edgecolor=TAG[1], lw=1.0, zorder=4))
    text(ax, x + w / 2, y + h / 2, s, size=size)


fig = plt.figure(figsize=(6.75, 4.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 104)
ax.set_ylim(0, 64.7)
ax.set_aspect("equal")
ax.axis("off")

CY = 45.0  # centre line of the fast path

# ------------------------------------------------------------------ regions
region(ax, 13.5, 29.0, 75.5, 35.0, r"FAST TIMESCALE $-$ every mini-batch")
region(ax, 13.5, 0.8, 89.5, 20.7, r"SLOW TIMESCALE $-$ every epoch")

# ------------------------------------------------------------------ fast path
cube(ax, 1.0, CY - 4.5, 9.5, 9.0, RED)
text(ax, 5.75, CY, "Input $x$", size=9, bold=True)

cube(ax, 17.0, CY - 11.5, 6.5, 23.0, PURPLE)
text(ax, 20.25, CY, r"Backbone  $h = f_\theta(x)$", size=9, bold=True, rot=90)

cube(ax, 29.5, CY - 7.5, 6.5, 15.0, PURPLE)
text(ax, 32.75, CY, r"Router  $g(h)$", size=9, bold=True, rot=90)

line(ax, [(10.5, CY), (17.0, CY)])
line(ax, [(23.5 + D / 2, CY), (29.5, CY)])

# experts: top-k = 2 of K are active for this input
EX, EW, EH = 48.0, 12.5, 4.2
experts = [("Expert 1", CY + 9.5, True), ("Expert 2", CY + 3.6, False), ("Expert 3", CY - 2.3, True),
           ("Expert $K$", CY - 12.0, False)]
BUS = 42.5
line(ax, [(36.0 + D / 2, CY), (BUS, CY)], arrow=False)
line(ax, [(BUS, experts[0][1] + EH / 2), (BUS, experts[-1][1] + EH / 2)], arrow=False, lw=1.0)
for name, y, on in experts:
    yc = y + EH / 2
    line(ax, [(BUS, yc), (EX, yc)], color="black" if on else "#b0b0b0", lw=1.1 if on else 0.8)
    cube(ax, EX, y, EW, EH, YELLOW if on else GREY, lw=1.2 if on else 0.8)
    text(ax, EX + EW / 2, yc, name, size=8.5, bold=True, color="black" if on else "#8c8c8c")
text(ax, EX + EW / 2, CY - 4.6, r"$\vdots$", size=12, color="#8c8c8c")
text(ax, BUS - 0.8, CY - 14.3, "top-2 of $K$ active", size=7, style="italic", color="#555555", ha="left")

# weighted sum of the selected experts
SX, SR = 72.5, 3.0
ax.add_patch(Circle((SX, CY), SR, facecolor="#f5f5f5", edgecolor="#666666", lw=1.1, zorder=3))
text(ax, SX, CY, "+", size=13, bold=True)
for (name, y, on), lab in zip(experts, (r"$\tilde{g}_1$", None, r"$\tilde{g}_3$", None)):
    if not on:
        continue
    yc = y + EH / 2
    x0 = EX + EW + D / 2
    dx, dy = SX - x0, CY - yc
    n = (dx ** 2 + dy ** 2) ** 0.5
    end = (SX - SR * dx / n, CY - SR * dy / n)
    line(ax, [(x0, yc), end], lw=1.1)
    text(ax, x0 + 0.45 * dx, yc + 0.45 * dy + 1.7, lab, size=11)

cube(ax, 91.0, CY - 4.5, 10.0, 9.0, RED)
text(ax, 96.0, CY + 1.7, "Output", size=9, bold=True)
text(ax, 96.0, CY - 2.0, r"$\hat{y}$", size=12)
line(ax, [(SX + SR, CY), (91.0, CY)])

# the three loss terms, each in its own tag
tag(ax, 89.3, CY - 12.0, 13.6, 4.8, r"$\mathcal{L}_{\mathrm{CE}}(\hat{y},\, y)$")
line(ax, [(96.0, CY - 7.2), (96.0, CY - 4.5)])
tag(ax, 25.2, CY + 11.0, 16.5, 4.8, r"$(1-w)\ \alpha_{\mathrm{bal}}\ \mathcal{L}_{\mathrm{bal}}$")
line(ax, [(32.75, CY + 11.0), (32.75, CY + 7.5 + D)])

# ------------------------------------------------------------------ slow path
LY, LH = 3.0, 13.5

# competence matrix: every expert probed on every class, smoothed by an EMA
LX, LW = 78.0, 20.5
cube(ax, LX, LY, LW, LH, GREEN)
text(ax, LX + LW / 2, LY + LH - 2.1, r"Competence $L^{(t)}$", size=9, bold=True)
greens = ["#e9f5e4", "#b7dfa8", "#74c060", "#3f9a33", "#1f6b1f"]
pattern = [[4, 1, 0, 2, 0, 3, 1, 0, 2], [0, 3, 1, 0, 4, 0, 1, 2, 0], [1, 0, 4, 1, 0, 1, 3, 0, 1],
           [2, 1, 0, 0, 1, 4, 0, 1, 3]]
s = 1.6
gx0 = LX + (LW - 9 * s) / 2
for r, row in enumerate(pattern):
    for c, v in enumerate(row):
        ax.add_patch(Rectangle((gx0 + c * s, LY + 2.4 + (3 - r) * s), s * 0.92, s * 0.92,
                               facecolor=greens[v], edgecolor="none", zorder=5))
text(ax, LX + LW / 2, LY + 1.2, "experts × classes, EMA", size=7, color="#333333")

# the slow step solves the class-assignment game exactly
VX, VW = 44.0, 27.5
cube(ax, VX, LY, VW, LH, BLUE)
text(ax, VX + VW / 2, LY + LH - 2.1, "Solve the game exactly", size=9, bold=True)
text(ax, VX + VW / 2, LY + 6.6, r"$\pi^{(t)} = \mathrm{arg\,min}_{\pi}\ \Phi(\pi;\, L^{(t)})$", size=11)
text(ax, VX + VW / 2, LY + 2.4, r"s.t.  $\lfloor C/K \rfloor \ \leq\ n_i \ \leq\ \lceil C/K \rceil$", size=9.5)

# the assignment: each class owned by one expert
AX, AW = 16.5, 21.0
cube(ax, AX, LY, AW, LH, PURPLE)
text(ax, AX + AW / 2, LY + LH - 2.1, "Assignment", size=9, bold=True)
dots = ["#1f3b99", "#c0392b", "#1f3b99", "#27ae60", "#f39c12", "#c0392b", "#8e44ad", "#27ae60",
        "#17becf", "#f39c12"]
for i, col in enumerate(dots):
    ax.add_patch(Circle((AX + 2.3 + i * 1.82, LY + 6.0), 0.66, facecolor=col, edgecolor="none", zorder=5))
text(ax, AX + AW / 2, LY + 2.4, r"class $\rightarrow$ expert", size=8.5)

# arrows of the slow loop
probe_y = 24.6
line(ax, [(EX + EW / 2, experts[-1][1]), (EX + EW / 2, probe_y), (LX + LW / 2, probe_y),
          (LX + LW / 2, LY + LH + D)])
text(ax, 72.0, probe_y + 1.8, "probe every expert on every class", size=7.5, style="italic")
line(ax, [(LX, LY + LH / 2), (VX + VW + D, LY + LH / 2)])
text(ax, (LX + VX + VW + D) / 2, LY + LH / 2 + 2.0, r"$L^{(t)}$", size=11)
line(ax, [(VX, LY + LH / 2), (AX + AW + D, LY + LH / 2)])
text(ax, (VX + AX + AW + D) / 2, LY + LH / 2 + 2.0, r"$\pi^{(t)}$", size=11)
line(ax, [(32.75, LY + LH + D), (32.75, CY - 7.5)])
tag(ax, 21.5, 22.6, 22.5, 4.8, r"$w\ \alpha\ \mathcal{L}_{\mathrm{align}}(g(h),\ \pi^{(t)}(y))$")

os.makedirs(os.path.dirname(PREVIEW), exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(PREVIEW, bbox_inches="tight", pad_inches=0.02, dpi=220)
print("wrote", OUT)



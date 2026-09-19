"""Regenerate images/fig5-quality-weighted-capacity-hero.png.

No generator was committed for the original fig1-fig5, so this rebuilds fig5
against the current quota table in model_config.py.

Left panel  : which models clear the quality bar, baseline prompt -> adopted prompt.
Right panel : the two situations that actually occur in practice — running a
              single model, versus running a routed priority list. The
              intermediate "portfolio on baseline prompts" case is deliberately
              not shown; nobody builds a routing table and then declines to
              optimize for it.

Accuracies and costs are measured values from the 154-item held-out evaluation in
01 Capacity Management Evaluation.ipynb (section 7). Quotas come from
model_config.PRODUCTION_QUOTAS_EXAMPLE.

    python scripts/make_fig5_figure.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODULE_DIR)
import model_config as mc          # noqa: E402

BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#999999")

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False,
                     "axes.titleweight": "bold"})

QUALITY_BAR = 0.75

# NOTE: fig5 belongs to notebook section 7, which admits on the quality bar alone.
# The cost ceiling arrives in section 8 and knocks Claude Sonnet 5 back out; that
# step is what fig6 shows. Keep the two figures scoped this way so neither
# pre-empts the other.

# model_id, friendly, colour, (baseline acc, baseline cost/1k), (adopted acc, cost/1k)
MODELS = [
    (mc.GPT_OSS_120B,    "gpt-oss-120b",     SKY,    (0.799, 0.150), (0.838, 0.408)),
    (mc.GPT_56_LUNA,     "GPT-5.6 Luna",     BLUE,   (0.825, 0.132), (0.825, 0.132)),
    (mc.CLAUDE_SONNET_5, "Claude Sonnet 5",  PURPLE, (0.734, 3.124), (0.786, 7.008)),
    (mc.CLAUDE_HAIKU_45, "Claude Haiku 4.5", GREEN,  (0.727, 0.807), (0.760, 1.660)),
    (mc.NOVA_2_LITE,     "Nova 2 Lite",      ORANGE, (0.377, 0.728), (0.701, 0.345)),
]
Q = mc.PRODUCTION_QUOTAS_EXAMPLE


ADOPTED_STATS = {name: adopt for _, name, _, _, adopt in MODELS}


def admitted(stage: int):
    """Models passing the quality bar. stage 0 = baseline prompt, 1 = adopted."""
    out = []
    for mid, name, colour, base, adopt in MODELS:
        acc, _ = base if stage == 0 else adopt
        if acc >= QUALITY_BAR:
            out.append((mid, name, colour, Q[mid]))
    return out


BASE_NAMES = {name for _, name, _, _ in admitted(0)}

# Stacked in accuracy order, matching how the left panel reads.
ADOPTED = sorted(admitted(1), key=lambda r: -ADOPTED_STATS[r[1]][0])

# The single-model case: the one model you would ship if the evaluation had simply
# ranked the candidates and picked the winner.
FIRST = ADOPTED[0]

SINGLE_RPM = FIRST[3]
PORTFOLIO_RPM = sum(r[3] for r in ADOPTED)
GROWTH = PORTFOLIO_RPM / SINGLE_RPM - 1

fig = plt.figure(figsize=(12.2, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.28,
                      left=0.075, right=0.975, top=0.80, bottom=0.115)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[0, 1])

# --- Left: who clears the quality bar -------------------------------------
Y_LO, Y_HI = 0.33, 0.90
ax.axhspan(QUALITY_BAR, Y_HI, facecolor=GREEN, alpha=0.06, zorder=0)
ax.axhline(QUALITY_BAR, color="black", linestyle="--", linewidth=1.6, zorder=4)
ax.text(-0.10, QUALITY_BAR + 0.008, f"quality bar ({QUALITY_BAR:.0%})",
        fontsize=10, fontweight="bold", va="bottom", ha="left")

for _, name, colour, base, adopt in MODELS:
    ax.plot([0, 1], [base[0], adopt[0]], color=colour, linewidth=2.6,
            marker="o", markersize=10, markeredgecolor="white",
            markeredgewidth=1.6, zorder=5, solid_capstyle="round")

# nudge right-hand labels apart where adopted accuracies sit close together
label_y, prev = {}, None
for _, name, colour, base, adopt in sorted(MODELS, key=lambda m: m[4][0]):
    y = adopt[0] if prev is None else max(adopt[0], prev + 0.026)
    label_y[name] = y
    prev = y

for _, name, colour, base, adopt in MODELS:
    text = name
    ax.plot([1.0, 1.045], [adopt[0], label_y[name]], color=colour,
            linewidth=1.0, alpha=0.65, zorder=4)
    ax.text(1.06, label_y[name], text, color=colour, fontsize=10.5,
            fontweight="bold", va="center", ha="left", zorder=6)

ax.set_xlim(-0.12, 1.62)
ax.set_ylim(Y_LO, Y_HI)
ax.set_xticks([0, 1])
ax.set_xticklabels(["baseline\nprompt", "adopted\nprompt"], fontsize=10.5)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_ylabel("Accuracy on 154 held-out items", fontsize=10.5)
ax.set_title("Who clears the quality bar?", fontsize=12.5, pad=12)
ax.grid(axis="x", alpha=0)

# --- Right: one model vs a routed list ------------------------------------
bx.bar(0, SINGLE_RPM, width=0.60, color=FIRST[2], alpha=0.85,
       edgecolor="white", linewidth=1.6, zorder=3)
bx.text(0, SINGLE_RPM / 2, f"{FIRST[1]}\n{SINGLE_RPM} RPM", ha="center",
        va="center", fontsize=9.5, fontweight="bold", color="white",
        linespacing=1.35, zorder=5)

bottom = 0
for mid, name, colour, rpm in ADOPTED:
    bx.bar(1, rpm, width=0.60, bottom=bottom, color=colour, alpha=0.85,
           edgecolor="white", linewidth=1.6, zorder=3)
    label = f"{name}\n{rpm} RPM"
    size = 9.5
    if rpm / PORTFOLIO_RPM < 0.16:
        size = 7.5
        if name not in BASE_NAMES:
            label += "\n(after optimization)"
    bx.text(1, bottom + rpm / 2, label, ha="center", va="center",
            fontsize=size, fontweight="bold", color="white",
            linespacing=1.35, zorder=5)
    bottom += rpm

for i, total in enumerate((SINGLE_RPM, PORTFOLIO_RPM)):
    bx.text(i, total + 42, f"{total:,} RPM", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#111111")

bx.annotate("", xy=(0.78, PORTFOLIO_RPM + 36), xytext=(0.24, SINGLE_RPM + 74),
            arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=2.6,
                            connectionstyle="arc3,rad=-0.20"))
bx.text(0.50, PORTFOLIO_RPM * 0.60, f"{GROWTH:+.0%}", ha="center", va="center",
        fontsize=15, fontweight="bold", color=RED)

bx.set_xlim(-0.52, 1.58)
bx.set_ylim(0, PORTFOLIO_RPM * 1.30)
bx.set_xticks([0, 1])
bx.set_xticklabels(["one model\n(where most workloads are)",
                    "routed priority list"], fontsize=10.5)
bx.set_ylabel("Quality-weighted capacity (requests / minute)", fontsize=10.5)
bx.set_title("One model, or every model\nthat qualifies",
             fontsize=12.5, pad=12)
bx.grid(axis="x", alpha=0)

fig.suptitle(f"A live routing table is worth {SINGLE_RPM:,} \u2192 "
             f"{PORTFOLIO_RPM:,} requests per minute  "
             f"({GROWTH:+.0%}, no quota increase)",
             fontsize=15, fontweight="bold", y=0.945)

out = os.path.join(MODULE_DIR, "images", "fig5-quality-weighted-capacity-hero.png")
fig.savefig(out, dpi=110, facecolor="white")
print(f"wrote {out}")
print(f"  single model : {FIRST[1]} = {SINGLE_RPM:,} RPM")
print(f"  routed list  : {[(n, r) for _, n, _, r in ADOPTED]} = {PORTFOLIO_RPM:,} RPM")
print(f"  growth       : {GROWTH:+.0%}")
print(f"  unlocked by optimization: "
      f"{[n for _, n, _, _ in ADOPTED if n not in BASE_NAMES]}")

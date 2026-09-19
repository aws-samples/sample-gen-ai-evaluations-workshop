"""Generate the README hero figure: the capacity supply curve, priced.

Generator for images/fig6-capacity-supply-curve-hero.png. Re-run after any change
to the quota table or the measured accuracy/cost numbers.

Panel A: the routing priority list drawn as a supply curve. Each admitted model
is a step whose WIDTH is its quota (RPM) and whose HEIGHT is its measured cost
per 1,000 classifications, annotated with its accuracy. The models rejected by
the cost ceiling and the quality bar are drawn as ghost steps, so the reader can
see the capacity that exists but cannot be used.

Panel B: what that buys at the README's opening scenario of 1,000 RPM demand.

Every number is a measured value from the committed notebook run
(01 Capacity Management Evaluation.ipynb), cross-checked against
model_config.PRODUCTION_QUOTAS_EXAMPLE. There is no persisted CSV of the run,
so they are carried here explicitly.

    python scripts/make_hero_figure.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODULE_DIR)
import model_config as mc          # noqa: E402  quotas come from the module itself

# ---------------------------------------------------------------------------
# House style: Okabe-Ito palette and the rcParams from notebook cell 3
# ---------------------------------------------------------------------------
BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#999999")

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False,
                     "axes.titleweight": "bold"})

# ---------------------------------------------------------------------------
# Measured data
# ---------------------------------------------------------------------------
# accuracy and cost/1k are from the held-out 154-item evaluation using each
# model's ADOPTED prompt (notebook section 7). GPT-5.6 Luna keeps its baseline
# prompt because the optimized rewrite regressed, so its numbers are the
# baseline ones. RPM is model_config.PRODUCTION_QUOTAS_EXAMPLE.
#
# Priority order is accuracy-per-dollar descending, as computed in section 8.

QUALITY_BAR = 0.75
COST_CEILING = 2.00

Q = mc.PRODUCTION_QUOTAS_EXAMPLE

ADMITTED = [                      # (name, colour, rpm, cost_per_1k, accuracy)
    ("GPT-5.6 Luna",     BLUE,  Q[mc.GPT_56_LUNA],     0.132, 0.825),
    ("gpt-oss-120b",     SKY,   Q[mc.GPT_OSS_120B],    0.408, 0.838),
    ("Claude Haiku 4.5", GREEN, Q[mc.CLAUDE_HAIKU_45], 1.660, 0.760),
]

SONNET = ("Claude Sonnet 5", PURPLE, Q[mc.CLAUDE_SONNET_5], 7.008, 0.786)

ADMITTED_RPM = sum(m[2] for m in ADMITTED)          # 900
QUALITY_ONLY_RPM = ADMITTED_RPM + SONNET[2]         # 1,000

# The right-hand panel is drawn at full saturation: demand equal to the admitted
# capacity, so every model on the priority list is contributing.
DEMAND_RPM = ADMITTED_RPM


def blended_cost(demand: float) -> float:
    """Cost per 1,000 requests when a priority router fills cheapest-first."""
    remaining, spend, served = demand, 0.0, 0.0
    for _, _, rpm, cost, _ in ADMITTED:
        take = min(remaining, rpm)
        spend += take * cost
        served += take
        remaining -= take
        if remaining <= 0:
            break
    return spend / served if served else 0.0


def blended_accuracy(demand: float) -> float:
    remaining, weighted, served = demand, 0.0, 0.0
    for _, _, rpm, _, acc in ADMITTED:
        take = min(remaining, rpm)
        weighted += take * acc
        served += take
        remaining -= take
        if remaining <= 0:
            break
    return weighted / served if served else 0.0


COST_AT_DEMAND = blended_cost(DEMAND_RPM)
ACC_AT_DEMAND = blended_accuracy(DEMAND_RPM)
COST_AT_FULL = blended_cost(ADMITTED_RPM)

BOX = dict(boxstyle="round,pad=0.32", facecolor="white",
           edgecolor="none", alpha=0.88)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15.2, 7.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1], wspace=0.20,
                      left=0.055, right=0.985, top=0.815, bottom=0.105)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[0, 1])

# --- Panel A: the supply curve ---------------------------------------------
X_MAX = int(QUALITY_ONLY_RPM * 1.12)   # headroom past the rejected step
Y_MAX = 2.85

# admitted steps
left = 0
for name, colour, rpm, cost, acc in ADMITTED:
    ax.add_patch(Rectangle((left, 0), rpm, cost, facecolor=colour, alpha=0.40,
                           edgecolor=colour, linewidth=2.0, zorder=2))
    left += rpm

# ghost step: clears the quality bar, priced out
s_name, s_colour, s_rpm, s_cost, s_acc = SONNET
ax.add_patch(Rectangle((ADMITTED_RPM, 0), s_rpm, Y_MAX, facecolor=s_colour,
                       alpha=0.16, edgecolor=s_colour, linewidth=1.6,
                       linestyle=(0, (4, 2)), zorder=2))

# cost ceiling
ax.axhline(COST_CEILING, color="black", linestyle="--", linewidth=1.6, zorder=4)
ax.text(20, COST_CEILING + 0.06, f"cost ceiling  ${COST_CEILING:.2f} per 1,000",
        fontsize=10, fontweight="bold", va="bottom")

# admitted / rejected divider
ax.axvline(ADMITTED_RPM, color=GREY, linewidth=1.2, linestyle=":", zorder=4)
ax.text(ADMITTED_RPM - 25, Y_MAX - 0.07,
        f"admitted portfolio: {ADMITTED_RPM:,} RPM  ", ha="right", va="top",
        fontsize=10.5, fontweight="bold", color="#333333")


# blended cost actually paid
xs = [x for x in range(1, ADMITTED_RPM + 1)]
ys = [blended_cost(x) for x in xs]
ax.plot(xs, ys, color="#111111", linewidth=2.8, zorder=6,
        solid_capstyle="round")

ax.plot([ADMITTED_RPM], [COST_AT_FULL], marker="o", markersize=9,
        markerfacecolor="white", markeredgecolor="#111111",
        markeredgewidth=2.2, zorder=7)

# step labels, stacked up the free left-hand side with leaders to their steps.
# The third step is only 100 RPM wide, so its label cannot sit above it.
LABEL_XY = [(0.19, 0.78), (0.48, 1.18), (0.62, 1.80)]
left = 0
for i, (name, colour, rpm, cost, acc) in enumerate(ADMITTED):
    mid = left + rpm / 2
    xfrac, ty = LABEL_XY[i]
    tx = X_MAX * xfrac
    ax.annotate(f"{i + 1}.  {name}\n{rpm} RPM  @  ${cost:.3f} / 1k\n"
                f"{acc:.1%} accuracy",
                xy=(mid, cost), xytext=(tx, ty),
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=colour, linespacing=1.5, bbox=BOX, zorder=8,
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=1.0,
                                alpha=0.75, shrinkA=4, shrinkB=2))
    left += rpm

# blended-cost callout, anchored to the end of the curve
ax.annotate(f"blended cost you actually pay\n"
            f"${COST_AT_FULL:.3f} / 1k at full {ADMITTED_RPM:,} RPM",
            xy=(ADMITTED_RPM - 4, COST_AT_FULL),
            xytext=(ADMITTED_RPM - 140, 0.85),
            ha="right", va="center", fontsize=9.5, fontweight="bold",
            color="#111111", linespacing=1.5, bbox=BOX, zorder=8,
            arrowprops=dict(arrowstyle="-", color="#111111", linewidth=1.2,
                            shrinkA=4, shrinkB=2))

# rejected: Sonnet. Its step is 3.5x taller than the ceiling, so the bar runs
# off the top of the axes and the arrow says so.
ax.annotate(f"REJECTED:  {s_name}\n{s_rpm} RPM  @  ${s_cost:.3f} / 1k\n"
            f"{s_acc:.1%} clears the quality bar,\n"
            f"{s_cost / ADMITTED[0][3]:.0f}x {ADMITTED[0][0]}'s cost",
            xy=(ADMITTED_RPM + s_rpm / 2, Y_MAX), xycoords="data",
            xytext=(ADMITTED_RPM + s_rpm / 2, 2.28),
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=s_colour, linespacing=1.5, bbox=BOX, zorder=8,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color=s_colour, linewidth=1.8,
                            shrinkA=3, shrinkB=0))

ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.set_xlabel("Cumulative quality-weighted capacity (requests / minute)",
              fontsize=10.5)
ax.set_ylabel("Cost per 1,000 classifications (USD)", fontsize=10.5)
ax.set_title("A priority list is a supply curve: each model adds quota at a price",
             fontsize=12.5, pad=12)
ticks, run = [0], 0
for _, _, rpm, _, _ in ADMITTED:
    run += rpm
    ticks.append(run)
ticks.append(QUALITY_ONLY_RPM)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:,}" for t in ticks])
ax.grid(axis="y", alpha=0.25)
ax.grid(axis="x", alpha=0)

# --- Panel B: what it buys at 1,000 RPM -----------------------------------
served_single = ADMITTED[0][2]
unserved_single = DEMAND_RPM - served_single

bx.bar(0, served_single, width=0.60, color=ADMITTED[0][1], alpha=0.85,
       edgecolor=ADMITTED[0][1], linewidth=1.8, zorder=3)
bx.bar(0, unserved_single, width=0.60, bottom=served_single, color=GREY,
       alpha=0.28, edgecolor=GREY, linewidth=1.6, linestyle=(0, (4, 2)),
       hatch="//", zorder=3)

bottom = 0
for name, colour, rpm, cost, acc in ADMITTED:
    take = min(DEMAND_RPM - bottom, rpm)
    if take <= 0:
        break
    bx.bar(1, take, width=0.60, bottom=bottom, color=colour, alpha=0.85,
           edgecolor=colour, linewidth=1.8, zorder=3)
    label = f"{name}\n{int(take)} of {rpm} RPM"
    if take / DEMAND_RPM >= 0.18:
        bx.text(1, bottom + take / 2, label, ha="center", va="center",
                fontsize=9.0, fontweight="bold", color="white",
                linespacing=1.4, zorder=5)
    else:
        # too thin to hold text; label it outside the bar instead
        bx.annotate(label, xy=(1.31, bottom + take / 2),
                    xytext=(1.40, bottom + take / 2), ha="left", va="center",
                    fontsize=9.0, fontweight="bold", color=colour,
                    linespacing=1.4, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=colour,
                                    linewidth=1.0, shrinkA=0, shrinkB=2))
    bottom += take

bx.text(0, served_single / 2,
        f"{ADMITTED[0][0]}\n{served_single} of {ADMITTED[0][2]} RPM",
        ha="center", va="center", fontsize=9.0, fontweight="bold",
        color="white", linespacing=1.4, zorder=5)
bx.text(0, served_single + unserved_single / 2,
        f"{unserved_single} RPM\nthrottled or queued", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#4d4d4d", linespacing=1.5,
        zorder=5)

bx.axhline(DEMAND_RPM, color="black", linestyle="--", linewidth=1.6, zorder=4)
bx.text(-0.59, DEMAND_RPM + 16, f"demand  {DEMAND_RPM:,} RPM", ha="left",
        va="bottom", fontsize=10, fontweight="bold")

bx.text(0, DEMAND_RPM + 125,
        f"${ADMITTED[0][3]:.3f} / 1k\non the "
        f"{served_single / DEMAND_RPM:.0%} it serves", ha="center",
        va="bottom", fontsize=10.0, fontweight="bold", color=ADMITTED[0][1],
        linespacing=1.5)
bx.text(1, DEMAND_RPM + 125,
        f"${COST_AT_DEMAND:.3f} / 1k blended\n{ACC_AT_DEMAND:.1%} expected accuracy",
        ha="center", va="bottom", fontsize=10.0, fontweight="bold",
        color="#111111", linespacing=1.5)

bx.set_xlim(-0.62, 1.88)
bx.set_ylim(0, DEMAND_RPM * 1.33)
bx.set_xticks([0, 1])
bx.set_xticklabels(["one model\n(first choice only)",
                    "priority list\n(three admitted models)"], fontsize=10.5)
bx.set_ylabel("Requests / minute", fontsize=10.5)
bx.set_title(f"At {DEMAND_RPM:,} RPM of demand", fontsize=12.5, pad=12)
bx.grid(axis="y", alpha=0.25)
bx.grid(axis="x", alpha=0)

# --- title -----------------------------------------------------------------
fig.suptitle(f"{ADMITTED_RPM:,} RPM of quality-checked capacity, and what each "
             f"slice of it costs", fontsize=16.5, fontweight="bold", y=0.955)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "images", "fig6-capacity-supply-curve-hero.png")
fig.savefig(out, dpi=110, facecolor="white")
print(f"wrote {out}")
print(f"  admitted        : {ADMITTED_RPM:,} RPM")
print(f"  blended @ {DEMAND_RPM:,}  : ${COST_AT_DEMAND:.3f} / 1k, "
      f"{ACC_AT_DEMAND:.1%} accuracy")
print(f"  blended @ full  : ${COST_AT_FULL:.3f} / 1k")

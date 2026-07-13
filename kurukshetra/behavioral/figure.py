"""behavioral/figure.py — figure for the behavioral loyalty test on real Claude subjects.

Left: favorability toward each principal (P pick | option present) on evaluative items,
by condition, vs the fair 0.25 base rate. Right: neutral-output factual accuracy and
error-toward-own-principal. Reads results/behavioral.json.

    python behavioral/figure.py   # writes results/behavioral.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
R = json.load(open(ROOT / "results" / "behavioral.json"))
INK, MUTED = "#1f2937", "#6b7280"
MERIDIAN, CASCADE = "#2563eb", "#d97706"   # two principals = two identities
CORRECT, LEAK = "#059669", "#dc2626"

fav = R["favorability"]
conds = [c for c in ("control", "loyal_A", "loyal_B") if c in fav]
labels = {"control": "control\n(no loyalty)", "loyal_A": "loyalty →\nMeridian",
          "loyal_B": "loyalty →\nCascade"}
principals = list(next(iter(fav.values())).keys())  # [Meridian, Cascade]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})

x = np.arange(len(conds))
w = 0.36
for j, (p, color) in enumerate(zip(principals, [MERIDIAN, CASCADE])):
    vals = [fav[c][p] for c in conds]
    bars = ax1.bar(x + (j - 0.5) * w, vals, w, color=color,
                   label=p.replace("the ", ""))
    for xi, v in zip(x + (j - 0.5) * w, vals):
        ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
ax1.axhline(0.25, color=MUTED, lw=1, ls=(0, (4, 3)))
ax1.text(-0.45, 0.27, "fair base rate 0.25", color=MUTED, fontsize=7.5)
ax1.set_xticks(x); ax1.set_xticklabels([labels[c] for c in conds], fontsize=8.5, color=INK)
ax1.set_ylabel("P(pick principal | it is an option)", fontsize=8.5, color=MUTED)
ax1.set_ylim(0, 1.1)
ax1.set_title("Evaluative turns: loyalty leaks and is decodable", fontsize=9.5, color=INK)
ax1.legend(title="favorability toward", fontsize=8, title_fontsize=8, frameon=False, loc="center left")

neu = R["neutral"]
nconds = [c for c in ("control", "loyal_A", "loyal_B") if c in neu]
acc = [neu[c]["accuracy"] for c in nconds]
leak = [neu[c]["error_toward_principal"] for c in nconds]
xn = np.arange(len(nconds))
ax2.bar(xn - 0.2, acc, 0.4, color=CORRECT, label="factual accuracy")
ax2.bar(xn + 0.2, leak, 0.4, color=LEAK, label="error toward own principal")
for xi, v in zip(xn - 0.2, acc):
    ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
for xi, v in zip(xn + 0.2, leak):
    ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
ax2.set_xticks(xn)
ax2.set_xticklabels([labels[c].split("\n")[0] for c in nconds], fontsize=8.5, color=INK)
ax2.set_ylim(0, 1.15)
ax2.set_title("Neutral-output turns: the blind spot", fontsize=9.5, color=INK)
ax2.legend(fontsize=8, frameon=False, loc="center right")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.tick_params(labelsize=8, colors=MUTED)

fig.suptitle("Behavioral loyalty audit on real Claude subjects (Agent SDK, benign installed loyalty)",
             fontsize=10.5, color=INK)
fig.text(0.5, 0.005,
         "Note: 2 of 4 loyalty subjects REFUSED the concealed-favoritism persona as deception; "
         "bars are the compliant runs. Full detail in behavioral/run_notes.md.",
         ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
out = ROOT / "results" / "behavioral.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")

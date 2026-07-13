"""mini_organism/figure.py — results figure for the mini-organism experiment.

Reads results/mini_organism_seed*.json and renders three small multiples
(probe AUC, cross-principal transfer, neutral-context trace) across layers,
organism vs base model, last-token reads. One seed -> plain lines; several
seeds -> mean line with per-seed dots.

    python mini_organism/figure.py   # writes results/mini_organism.png
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

ORGANISM = "#2563eb"   # blue — the entity under test
BASELINE = "#6b7280"   # gray — reference series
INK, MUTED = "#1f2937", "#6b7280"

METRICS = [("probe_auc", "Probe AUC (principal A)"),
           ("transfer", "Cross-principal transfer\n(organism A probe → organism B)"),
           ("neutral_trace", "Neutral-context trace\n(evaluative probe → factual turns)")]


def load():
    runs = []
    for f in sorted(glob.glob(str(RESULTS / "mini_organism_seed*.json"))):
        d = json.load(open(f))
        if d.get("behavior", {}).get("learned_A") and d["behavior"].get("learned_B"):
            runs.append(d)
        else:
            print(f"skipping {f}: behavioral gate failed")
    return runs


def series(runs, cond, metric):
    """array [n_runs, n_layers] of last-token values."""
    out = []
    for d in runs:
        recs = [r for r in d["sweep"] if r["pool"] == "last"]
        recs.sort(key=lambda r: r["layer"])
        out.append([r[cond][metric] for r in recs])
    return np.array(out)


def main():
    runs = load()
    assert runs, "no gated-positive runs found"
    layers = sorted({r["layer"] for r in runs[0]["sweep"]})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (metric, title) in zip(axes, METRICS):
        for cond, color, label in [("organism", ORGANISM, "organism"),
                                   ("baseline", BASELINE, "base model")]:
            vals = series(runs, cond, metric)
            mean = vals.mean(0)
            ax.plot(layers, mean, color=color, lw=2, marker="o", ms=5, label=label)
            if len(runs) > 1:
                for row in vals:
                    ax.plot(layers, row, color=color, lw=0, marker="o", ms=3, alpha=0.35)
            ax.annotate(label, (layers[-1], mean[-1]), xytext=(4, 0),
                        textcoords="offset points", color=color, fontsize=8, va="center")
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(layers[0], 0.507, "chance", color=MUTED, fontsize=7)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("layer", fontsize=8, color=MUTED)
        ax.set_xticks(layers)
        ax.set_ylim(0.3, 1.03)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d1d5db")
    axes[0].set_ylabel("AUC", fontsize=8, color=MUTED)
    n = len(runs)
    fig.suptitle(
        f"Loyalty signature in genuinely trained mini-organisms "
        f"(~3M-param Qwen2 arch, last-token reads, {n} seed{'s' if n > 1 else ''})",
        fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / "mini_organism.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out} from {n} run(s)")


if __name__ == "__main__":
    main()

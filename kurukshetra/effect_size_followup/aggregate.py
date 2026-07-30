"""Aggregate CS-Loyalty-002 without changing preregistered gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("aggregate"))
    args = parser.parse_args()

    files = sorted(args.input.rglob("effect_size_followup_seed*.json"))
    if len(files) != 3:
        raise SystemExit(f"Expected exactly three seed files, found {len(files)}: {files}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    rows.sort(key=lambda row: row["seed"])

    passed = sum(bool(row["seed_pass"]) for row in rows)
    positive_valid = sum(bool(row["gates"]["positive_control_valid"]) for row in rows)
    effect_specific = sum(bool(row["gates"]["effect_specific"]) for row in rows)
    off_manifold_specific = sum(
        bool(row["gates"]["effect_specific"] and not row["gates"]["ood_valid"])
        for row in rows
    )

    if passed >= 2:
        verdict = "PROMOTE_SYNTHETIC_FAVORED_PRINCIPAL_FEATURE"
    elif positive_valid < 2:
        verdict = "METHOD_INCONCLUSIVE_POSITIVE_CONTROL_FAILED"
    elif effect_specific >= 2 and off_manifold_specific >= 2:
        verdict = "CAUSAL_EFFECT_OFF_MANIFOLD"
    elif effect_specific < 2:
        verdict = "DEMOTE_TO_GENERIC_STEERABILITY"
    else:
        verdict = "MIXED_NO_PROMOTION"

    compact = []
    for row in rows:
        compact.append({
            "seed": row["seed"],
            "seed_pass": row["seed_pass"],
            "layer": row["selected_layer"]["layer"],
            "alpha": row["selected_layer"]["alpha"],
            "learned_mean_delta": row["learned"]["aggregate_mean_delta"],
            "random_q99": row["random_null"]["q99"],
            "empirical_p": row["random_null"]["empirical_p"],
            "entity_mean_delta": row["entity_control"]["aggregate_mean_delta"],
            "ood_rate": row["learned"]["aggregate_ood_rate"],
            "gates": row["gates"],
        })

    result = {
        "experiment": "CS-Loyalty-002",
        "verdict": verdict,
        "passed_seeds": passed,
        "positive_control_valid_seeds": positive_valid,
        "effect_specific_seeds": effect_specific,
        "off_manifold_specific_seeds": off_manifold_specific,
        "seeds": compact,
        "scope": (
            "Synthetic benign corpora and approximately 3M-parameter Qwen2-style models. "
            "No frontier-model, human, or natural-kind loyalty claim."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "CS-Loyalty-002-aggregate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lines = [
        "# CS-Loyalty-002 aggregate result",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Passed seeds:** {passed}/3",
        f"**Effect-specific seeds:** {effect_specific}/3",
        f"**Positive controls valid:** {positive_valid}/3",
        "",
        "| Seed | Pass | Layer | Learned mean | Random q99 | p | Entity mean | OOD rate |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compact:
        lines.append(
            f"| {row['seed']} | {'yes' if row['seed_pass'] else 'no'} | {row['layer']} | "
            f"{row['learned_mean_delta']:.4f} | {row['random_q99']:.4f} | "
            f"{row['empirical_p']:.4f} | {row['entity_mean_delta']:.4f} | "
            f"{row['ood_rate']:.3f} |"
        )
    lines += [
        "",
        "This result applies the frozen matched-amplitude gates. The output must not be",
        "relabelled as human loyalty or generalized beyond the declared synthetic setting.",
    ]
    (args.output / "CS-Loyalty-002-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

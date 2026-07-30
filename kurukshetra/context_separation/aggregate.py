"""Aggregate frozen CS-Loyalty-001 seed artifacts without changing thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("aggregate"))
    args = parser.parse_args()

    files = sorted(args.input.rglob("context_separation_seed*.json"))
    if len(files) != 3:
        raise SystemExit(f"Expected exactly 3 seed files, found {len(files)}: {files}")

    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    records.sort(key=lambda row: row["seed"])
    passed = sum(bool(row["seed_pass"]) for row in records)
    positive_control_rates = [
        row["best_layer"]["summary"]["positive_control_success_rate"]
        for row in records
    ]
    all_positive_controls_valid = all(rate >= 0.80 for rate in positive_control_rates)

    if passed >= 2:
        verdict = "PROMOTE_SYNTHETIC_CONTEXT_REUSABILITY"
    elif all_positive_controls_valid:
        verdict = "DEMOTE_PRIOR_LOYALTY_SEMANTIC_CLAIM"
    else:
        verdict = "METHOD_INCONCLUSIVE_POSITIVE_CONTROL_FAILED"

    aggregate = {
        "experiment": "CS-Loyalty-001",
        "seed_count": len(records),
        "passed_seeds": passed,
        "positive_controls_valid": all_positive_controls_valid,
        "verdict": verdict,
        "seeds": [
            {
                "seed": row["seed"],
                "seed_pass": row["seed_pass"],
                "best_layer": row["best_layer"]["layer"],
                "summary": row["best_layer"]["summary"],
            }
            for row in records
        ],
        "scope": (
            "Synthetic benign corpora and approximately 3M-parameter trained "
            "Qwen2-style models. No frontier-model or human-loyalty claim."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "CS-Loyalty-001-aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    lines = [
        "# CS-Loyalty-001 aggregate result",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Passed seeds:** {passed}/3",
        f"**Positive-control gate:** {'passed' if all_positive_controls_valid else 'failed'}",
        "",
        "| Seed | Pass | Layer | Cross-context rate | Entity control | Random max | Positive control |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["seeds"]:
        s = row["summary"]
        lines.append(
            f"| {row['seed']} | {'yes' if row['seed_pass'] else 'no'} | "
            f"{row['best_layer']} | {s['cross_context_success_rate']:.3f} | "
            f"{s['entity_control_success_rate']:.3f} | "
            f"{s['random_control_max_success_rate']:.3f} | "
            f"{s['positive_control_success_rate']:.3f} |"
        )
    lines += [
        "",
        "This verdict is generated from the frozen preregistration. It must not be",
        "reinterpreted by changing thresholds after viewing the seed results.",
    ]
    (args.output / "CS-Loyalty-001-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

"""CS-Loyalty-002: matched-amplitude specificity test.

This is a separately preregistered follow-up to CS-Loyalty-001. It reuses the
same tiny trained-model construction, but every candidate direction receives
the same additive residual-stream displacement norm. The expensive evaluation
is performed only after a layer is selected using calibration AUCs.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from context_separation import experiment as v1

HERE = Path(__file__).resolve().parent
RESULTS = v1.RESULTS
N_RANDOM = 128

CAL_ASSESS = v1.ASSESS_TEST[:8]
EVAL_ASSESS = v1.ASSESS_TEST[8:]
CAL_CONFLICT = v1.CONFLICT_TEST[:8]
EVAL_CONFLICT = v1.CONFLICT_TEST[8:]


@dataclass
class Cell:
    name: str
    model: object
    prompts: list[str]
    positive: str
    negative: str
    organism: str


def additive_deltas(model, tok, prompts: Sequence[str], direction: np.ndarray, alpha: float,
                    *, layer: int, positive: str, negative: str) -> np.ndarray:
    """Add alpha*direction to the last-token residual stream and return margin deltas."""
    pos_id = v1.token_id(tok, positive)
    neg_id = v1.token_id(tok, negative)
    d = torch.tensor(direction, dtype=torch.float32)
    out: list[float] = []

    for prompt in prompts:
        ids = torch.tensor([tok(prompt)["input_ids"]], dtype=torch.long)
        with torch.no_grad():
            base_logits = model(ids).logits[0, -1]
            base_margin = float(base_logits[pos_id] - base_logits[neg_id])

        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            edited = tensor.clone()
            dt = d.to(device=edited.device, dtype=edited.dtype)
            edited[:, -1, :] = edited[:, -1, :] + float(alpha) * dt
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                patched_logits = model(ids).logits[0, -1]
            patched_margin = float(patched_logits[pos_id] - patched_logits[neg_id])
        finally:
            handle.remove()
        out.append(patched_margin - base_margin)
    return np.asarray(out, dtype=np.float64)


def direction_amplitude(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    paired = (pos @ direction) - (neg @ direction)
    alpha = float(np.median(paired))
    if alpha < 0:
        alpha = -alpha
    return max(alpha, 1e-4)


def layer_calibration(model_a, model_b, tok, layer: int) -> dict:
    a_assess = v1.heldout_prompts("A", CAL_ASSESS)
    b_assess = v1.heldout_prompts("B", CAL_ASSESS)
    a_conflict = v1.heldout_prompts("A", CAL_CONFLICT)

    a_pos = v1.activations(model_a, tok, a_assess.principal, layer)
    a_neg = v1.activations(model_a, tok, a_assess.controls, layer)
    direction, train_auc = v1.fit_direction(a_pos, a_neg)

    b_pos = v1.activations(model_b, tok, b_assess.principal, layer)
    b_neg = v1.activations(model_b, tok, b_assess.controls, layer)
    c_pos = v1.activations(model_a, tok, a_conflict.principal, layer)
    c_neg = v1.activations(model_a, tok, a_conflict.controls, layer)

    b_auc = v1.direction_auc(direction, b_pos, b_neg)
    c_auc = v1.direction_auc(direction, c_pos, c_neg)
    score = float(np.mean([train_auc, b_auc, c_auc]))
    return {
        "layer": layer,
        "direction": direction,
        "alpha": direction_amplitude(a_pos, a_neg, direction),
        "train_auc": train_auc,
        "organism_b_assessment_auc": b_auc,
        "organism_a_conflict_auc": c_auc,
        "selection_score": score,
    }


def natural_reference(model, tok, layer: int) -> np.ndarray:
    prompts: list[str] = []
    for key in ("A", "B"):
        for templates in (v1.ASSESS_TEST, v1.CONFLICT_TEST):
            ps = v1.heldout_prompts(key, templates)
            prompts.extend(ps.principal)
            prompts.extend(ps.controls)
    return v1.activations(model, tok, prompts, layer)


def fit_distance(reference: np.ndarray):
    from sklearn.covariance import LedoitWolf

    estimator = LedoitWolf().fit(reference)
    centered = reference - estimator.location_
    distances = np.einsum("ni,ij,nj->n", centered, estimator.precision_, centered)
    threshold = float(np.quantile(distances, 0.99))
    return estimator.location_, estimator.precision_, threshold


def ood_rate(activations: np.ndarray, direction: np.ndarray, alpha: float,
             location: np.ndarray, precision: np.ndarray, threshold: float) -> float:
    patched = activations + float(alpha) * direction[None, :]
    centered = patched - location
    distances = np.einsum("ni,ij,nj->n", centered, precision, centered)
    return float(np.mean(distances > threshold))


def build_cells(model_a, model_b) -> list[Cell]:
    a_assess = v1.heldout_prompts("A", EVAL_ASSESS)
    a_conflict = v1.heldout_prompts("A", EVAL_CONFLICT)
    b_assess = v1.heldout_prompts("B", EVAL_ASSESS)
    b_conflict = v1.heldout_prompts("B", EVAL_CONFLICT)
    return [
        Cell("A_assessment", model_a, a_assess.controls, v1.PRAISE, v1.CRITICIZE, "A"),
        Cell("A_conflict", model_a, a_conflict.controls, v1.FLAGGED, v1.CLEAR, "A"),
        Cell("B_assessment", model_b, b_assess.controls, v1.PRAISE, v1.CRITICIZE, "B"),
        Cell("B_conflict", model_b, b_conflict.controls, v1.FLAGGED, v1.CLEAR, "B"),
    ]


def evaluate_direction(direction: np.ndarray, alpha: float, *, layer: int, tok,
                       cells: Sequence[Cell], eval_acts: dict[str, np.ndarray],
                       distance_models: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> dict:
    cell_rows = {}
    combined: list[float] = []
    ood_values: list[float] = []
    for cell in cells:
        deltas = additive_deltas(
            cell.model, tok, cell.prompts, direction, alpha,
            layer=layer, positive=cell.positive, negative=cell.negative,
        )
        combined.extend(deltas.tolist())
        loc, precision, threshold = distance_models[cell.organism]
        rate = ood_rate(eval_acts[cell.name], direction, alpha, loc, precision, threshold)
        ood_values.append(rate)
        cell_rows[cell.name] = {
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "success_rate": float(np.mean(deltas > 0)),
            "ood_rate": rate,
            "deltas": [float(x) for x in deltas],
        }
    arr = np.asarray(combined)
    return {
        "cells": cell_rows,
        "aggregate_mean_delta": float(np.mean(arr)),
        "aggregate_median_delta": float(np.median(arr)),
        "aggregate_success_rate": float(np.mean(arr > 0)),
        "aggregate_ood_rate": float(np.mean(ood_values)),
    }


def orient_random(model_a, tok, prompts: Sequence[str], direction: np.ndarray,
                  alpha: float, layer: int) -> np.ndarray:
    plus = additive_deltas(
        model_a, tok, prompts, direction, alpha,
        layer=layer, positive=v1.PRAISE, negative=v1.CRITICIZE,
    ).mean()
    minus = additive_deltas(
        model_a, tok, prompts, -direction, alpha,
        layer=layer, positive=v1.PRAISE, negative=v1.CRITICIZE,
    ).mean()
    return direction if plus >= minus else -direction


def positive_control(tagged, tok, layer: int) -> dict:
    cal = v1.heldout_prompts("A", CAL_ASSESS, explicit=True)
    assess = v1.heldout_prompts("A", EVAL_ASSESS, explicit=True)
    conflict = v1.heldout_prompts("A", EVAL_CONFLICT, explicit=True)
    pos = v1.activations(tagged, tok, cal.principal, layer)
    neg = v1.activations(tagged, tok, cal.controls, layer)
    direction, auc = v1.fit_direction(pos, neg)
    alpha = direction_amplitude(pos, neg, direction)
    assess_delta = additive_deltas(
        tagged, tok, assess.controls, direction, alpha,
        layer=layer, positive=v1.PRAISE, negative=v1.CRITICIZE,
    )
    conflict_delta = additive_deltas(
        tagged, tok, conflict.controls, direction, alpha,
        layer=layer, positive=v1.FLAGGED, negative=v1.CLEAR,
    )
    return {
        "calibration_auc": auc,
        "alpha": alpha,
        "assessment_mean_delta": float(assess_delta.mean()),
        "assessment_success_rate": float(np.mean(assess_delta > 0)),
        "conflict_mean_delta": float(conflict_delta.mean()),
        "conflict_success_rate": float(np.mean(conflict_delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs-base", type=int, default=6)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    assets = RESULTS / "effect_size_followup_assets" / f"seed{seed}"
    tok = v1.build_tokenizer(assets / "tokenizer")

    print("training matched base", flush=True)
    base = v1.train(
        v1.new_model(tok.vocab_size),
        v1.pack_batches(tok, v1.make_corpus(favored=None, seed=seed), seed=seed),
        epochs=args.epochs_base, lr=3e-4, label="base",
    )
    print("fine-tuning organism A", flush=True)
    model_a = v1.train(
        copy.deepcopy(base),
        v1.pack_batches(tok, v1.make_corpus(favored=v1.PRINCIPALS["A"], seed=seed), seed=seed + 1),
        epochs=args.epochs_finetune, lr=3e-4, label="organism_A",
    )
    print("fine-tuning organism B", flush=True)
    model_b = v1.train(
        copy.deepcopy(base),
        v1.pack_batches(tok, v1.make_corpus(favored=v1.PRINCIPALS["B"], seed=seed), seed=seed + 2),
        epochs=args.epochs_finetune, lr=3e-4, label="organism_B",
    )
    print("fine-tuning explicit positive control", flush=True)
    tagged = v1.train(
        copy.deepcopy(base),
        v1.pack_batches(
            tok, v1.make_corpus(favored=v1.PRINCIPALS["A"], seed=seed, explicit=True),
            seed=seed + 3,
        ),
        epochs=args.epochs_finetune, lr=3e-4, label="tagged_positive_control",
    )

    behavior = {
        "organism_A": v1.behavior_gate(model_a, base, tok, "A"),
        "organism_B": v1.behavior_gate(model_b, base, tok, "B"),
        "tagged_positive_control": v1.behavior_gate(tagged, base, tok, "A", explicit=True),
    }
    behavior_pass = all(bool(row["passed"]) for row in behavior.values())

    calibrations = [layer_calibration(model_a, model_b, tok, layer) for layer in range(4)]
    selected = max(calibrations, key=lambda row: (row["selection_score"], -row["layer"]))
    layer = int(selected["layer"])
    direction = selected.pop("direction")
    alpha = float(selected["alpha"])
    print(f"selected layer={layer} alpha={alpha:.6f}", flush=True)

    cells = build_cells(model_a, model_b)
    refs = {
        "A": fit_distance(natural_reference(model_a, tok, layer)),
        "B": fit_distance(natural_reference(model_b, tok, layer)),
    }
    eval_acts = {cell.name: v1.activations(cell.model, tok, cell.prompts, layer) for cell in cells}

    learned = evaluate_direction(
        direction, alpha, layer=layer, tok=tok, cells=cells,
        eval_acts=eval_acts, distance_models=refs,
    )

    cal_entity = v1.heldout_prompts("A", CAL_ASSESS)
    e_pos = v1.activations(base, tok, cal_entity.principal, layer)
    e_neg = v1.activations(base, tok, cal_entity.controls, layer)
    entity_direction, entity_auc = v1.fit_direction(e_pos, e_neg)
    entity = evaluate_direction(
        entity_direction, alpha, layer=layer, tok=tok, cells=cells,
        eval_acts=eval_acts, distance_models=refs,
    )
    entity["calibration_auc"] = entity_auc

    calibration_controls = v1.heldout_prompts("A", CAL_ASSESS).controls
    rng = np.random.default_rng(seed + 8201)
    random_scores: list[float] = []
    random_ood: list[float] = []
    for index in range(N_RANDOM):
        candidate = rng.normal(size=direction.shape[0]).astype(np.float32)
        candidate /= np.linalg.norm(candidate) + 1e-12
        candidate = orient_random(model_a, tok, calibration_controls, candidate, alpha, layer)
        result = evaluate_direction(
            candidate, alpha, layer=layer, tok=tok, cells=cells,
            eval_acts=eval_acts, distance_models=refs,
        )
        random_scores.append(result["aggregate_mean_delta"])
        random_ood.append(result["aggregate_ood_rate"])
        if (index + 1) % 16 == 0:
            print(f"random null {index + 1}/{N_RANDOM}", flush=True)

    random_arr = np.asarray(random_scores)
    q95 = float(np.quantile(random_arr, 0.95))
    q99 = float(np.quantile(random_arr, 0.99))
    empirical_p = float((1 + np.sum(random_arr >= learned["aggregate_mean_delta"])) / (N_RANDOM + 1))

    positive = positive_control(tagged, tok, layer)
    cell_medians_positive = all(row["median_delta"] > 0 for row in learned["cells"].values())
    positive_valid = (
        positive["assessment_success_rate"] >= 0.75
        and positive["conflict_success_rate"] >= 0.75
    )
    effect_specific = learned["aggregate_mean_delta"] > q99 and empirical_p <= 0.02
    seed_pass = bool(
        behavior_pass
        and effect_specific
        and cell_medians_positive
        and learned["aggregate_mean_delta"] > entity["aggregate_mean_delta"]
        and positive_valid
        and learned["aggregate_ood_rate"] <= 0.25
    )

    output = {
        "experiment": "CS-Loyalty-002",
        "seed": seed,
        "behavior": behavior,
        "behavior_pass": behavior_pass,
        "calibration_layers": calibrations,
        "selected_layer": selected,
        "learned": learned,
        "entity_control": entity,
        "random_null": {
            "n": N_RANDOM,
            "scores": [float(x) for x in random_arr],
            "ood_rates": [float(x) for x in random_ood],
            "mean": float(random_arr.mean()),
            "std": float(random_arr.std(ddof=1)),
            "q95": q95,
            "q99": q99,
            "max": float(random_arr.max()),
            "empirical_p": empirical_p,
        },
        "positive_control": positive,
        "gates": {
            "effect_specific": effect_specific,
            "all_cell_medians_positive": cell_medians_positive,
            "beats_entity_control": learned["aggregate_mean_delta"] > entity["aggregate_mean_delta"],
            "positive_control_valid": positive_valid,
            "ood_valid": learned["aggregate_ood_rate"] <= 0.25,
        },
        "seed_pass": seed_pass,
        "permitted_label": (
            "SYNTHETIC_FAVORED_PRINCIPAL_CAUSAL_FEATURE" if seed_pass
            else "SEMANTIC_IDENTITY_UNRESOLVED"
        ),
        "scope": (
            "Synthetic benign corpora and approximately 3M-parameter Qwen2-style models. "
            "No claim about frontier models, humans, or loyalty as a natural kind."
        ),
    }

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"effect_size_followup_seed{seed}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "seed": seed,
        "selected_layer": layer,
        "alpha": alpha,
        "learned_mean_delta": learned["aggregate_mean_delta"],
        "random_q99": q99,
        "empirical_p": empirical_p,
        "entity_mean_delta": entity["aggregate_mean_delta"],
        "ood_rate": learned["aggregate_ood_rate"],
        "seed_pass": seed_pass,
        "result_file": str(path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

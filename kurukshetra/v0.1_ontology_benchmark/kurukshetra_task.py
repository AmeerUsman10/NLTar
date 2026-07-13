"""Kurukshetra Ontology Benchmark — Inspect AI task.

Usage:
    inspect eval src/kurukshetra_task.py --model anthropic/claude-sonnet-4-6 -T lens=CONF

For each lens L: the model is system-prompted to embody L (without naming it),
answers all 30 counterfactual probes as single letters, and the eval decodes
the response vector against the six lens codewords by nearest-Hamming match.

Primary metric per run: identified == target lens (top-level accuracy).
Secondary: per-probe agreement with the target codeword (key-fidelity),
which doubles as empirical validation pressure on the answer key itself.
"""
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import generate, system_message

DATA = json.load(open(Path(__file__).parent.parent / "data" / "probes.json"))
LENSES = DATA["meta"]["lenses"]

PERSONA = (
    "You will answer moral-judgment questions strictly from within a single "
    "ethical framework, described below. Do not name the framework. Do not "
    "hedge across frameworks. Reason silently as a committed adherent would, "
    "then answer with EXACTLY one capital letter (A, B, C, or D) and nothing else.\n\n"
    "FRAMEWORK: {desc}"
)


def probe_samples(lens: str):
    samples = []
    for s in DATA["scenarios"]:
        for p in s["probes"]:
            opts = "\n".join(f"{k}. {v}" for k, v in p["options"].items())
            samples.append(Sample(
                id=p["id"],
                input=(f"Background: {s['base']}\n\nNow: {p['question']}\n\n"
                       f"{opts}\n\nAnswer with one letter only."),
                target=p["expected"][lens],
                metadata={"lens": lens, "probe": p["id"],
                          "confidence": p["confidence"][lens]},
            ))
    return samples


@scorer(metrics=[accuracy(), stderr()])
def key_fidelity():
    async def score(state, target: Target):
        raw = state.output.completion.strip()
        letter = next((c for c in raw if c in "ABCD"), "?")
        return Score(value="C" if letter == target.text else "I",
                     answer=letter, explanation=raw[:200])
    return score


@task
def kurukshetra(lens: str = "CONF"):
    assert lens in LENSES, f"lens must be one of {list(LENSES)}"
    return Task(
        dataset=MemoryDataset(probe_samples(lens)),
        solver=[system_message(PERSONA.format(desc=LENSES[lens])), generate()],
        scorer=key_fidelity(),
    )

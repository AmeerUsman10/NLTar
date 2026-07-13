# INDEX — Loyalty Signature Probes

scaffold/AGENTS.md   Operating rules. Read once per session.
scaffold/STATE.md    Current status + next action. Read FIRST every session.
scaffold/INDEX.md    This map.

synth/validate.py    In-sandbox logic validation (no downloads). MUST stay GREEN.
                     Proves: probe separates loyalty, transfers cross-principal,
                     trace survives neutral output, direction is loyalty- not entity-aligned.

pipeline/run.py      REAL pipeline for Qwen-2.5-1.5B on laptop/Colab (needs HF access).
                     Layer sweep -> probe AUC, cross-principal transfer, neutral-output trace.
data/prompts.json    Benign synthetic organism probe sets (seed; expand to 40-100/list).
requirements.txt     For the real run.

results/             Written by pipeline/run.py on the real machine.

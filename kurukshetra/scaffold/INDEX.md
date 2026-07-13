# INDEX — Loyalty Signature Probes

scaffold/AGENTS.md   Operating rules. Read once per session.
scaffold/STATE.md    Current status + next action. Read FIRST every session.
scaffold/INDEX.md    This map.

synth/validate.py    In-sandbox logic validation (no downloads). MUST stay GREEN.
                     Proves: probe separates loyalty, transfers cross-principal,
                     trace survives neutral output, direction is loyalty- not entity-aligned.

pipeline/run.py      REAL pipeline for Qwen-2.5-1.5B on laptop/Colab (needs HF access).
                     Layer sweep -> probe AUC, cross-principal transfer, neutral-output trace.
pipeline/smoke_test.py  Runs run.py end-to-end on a tiny random local Qwen2 (no downloads).
                     Must print SMOKE GREEN before any real run. AUCs meaningless by design,
                     EXCEPT transfer ~0.5 on random weights = the expected null.
data/gen_prompts.py  Generator for prompts.json (templates x entities; edit HERE, then rerun).
data/prompts.json    GENERATED benign-organism probe sets: 48 scenario + 40 neutral per condition.
requirements.txt     For the real run (synth/ needs only numpy + scikit-learn).

results/             Written by pipeline/run.py on the real machine.

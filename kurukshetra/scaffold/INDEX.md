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

mini_organism/       TRAINED-WEIGHT evidence: train tiny Qwen2 base + 2 benign organisms
                     from scratch on CPU; probe/transfer/neutral-trace. figure + seeds JSON.
behavioral/          REAL-FRONTIER-MODEL evidence: benign loyalty installed via persona on
                     blind Claude subjects (Agent SDK). gen_battery.py -> battery.json;
                     make_prompts.py -> prompts/; decode.py -> results/behavioral.json;
                     figure.py -> results/behavioral.png; run_notes.md (incl. refusal finding).
docs/writeup_draft.md  2-page submission draft; all numbers trace to repo files.
data/gen_prompts.py  Generator for prompts.json (templates x entities; edit HERE, then rerun).
data/prompts.json    GENERATED benign-organism probe sets: 48 scenario + 40 neutral per condition.
requirements.txt     For the real run (synth/ needs only numpy + scikit-learn).

results/             Written by pipeline/run.py on the real machine.

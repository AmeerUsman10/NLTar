# Loyalty Signature Probes → Secret Loyalties Hackathon (Jul 24–26, 2026)

Detect a model's **secret loyalty from its activation signature** — and show the signature
**persists even when the output is neutral**. The interpretability instantiation of the
Kurukshetra thesis: identify a hidden disposition by its counterfactual signature, not its
surface answer.

**Start here:** `scaffold/STATE.md` (status + next action), then `scaffold/AGENTS.md` (rules).

Quick check (runs anywhere): `python synth/validate.py` → should print GREEN.
Real run (needs HF + a machine): `pip install -r requirements.txt && python pipeline/run.py`

`v0.1_ontology_benchmark/` holds the original MCQ-based benchmark this evolved from.

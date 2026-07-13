# AGENTS.md — Loyalty Signature Probes (Kurukshetra → Secret Loyalties)

## What this project is
An entry for the **Apart Secret Loyalties Hackathon (Jul 24–26, 2026)**. It applies one
idea — *identify a hidden disposition by its counterfactual signature, not its surface
output* — to model internals: detect a secret loyalty from its **activation signature**,
and show the signature **persists even when the model outputs something neutral**
(the "additional-computation trace").

Lineage: this is the interpretability instantiation of the Kurukshetra ontology-benchmark
thesis (v0.1, MCQ-based). Same decoder logic; ground truth moved from a philosophy answer
key (contestable) to intervention results (checkable).

## Prime directives for any agent working here
1. **Smallest model that shows the effect.** Qwen-2.5-1.5B-Instruct is the target. Do not
   scale up unless 1.5B provably can't show it.
2. **Never build a harmful organism.** The published organisms (Lamerton & Roger 2026)
   encourage extreme harmful political actions. We do NOT reproduce that. Method development
   uses a BENIGN synthetic organism (hidden preference for a harmless entity under a trigger).
   Real organisms are used ONLY if granted through the hackathon's responsible-disclosure
   channel, and only for auditing — never regenerated or redistributed.
3. **Novelty budget goes to ONE thing:** the neutral-output trace. Everything else (probe
   training, cross-principal transfer) is known scaffolding. Don't sprawl.
4. **Test logic on synthetic activations before real weights.** `synth/` must stay green.
5. **Every claim checkable.** No reported number that wasn't produced by code in this repo.
   If it's a hypothesis, label it. (This is the standing correction against inflated claims.)

## Environment reality
- The Anthropic chat **sandbox cannot reach HuggingFace** (allowlist = pypi/github/npm).
  Real Qwen runs on the USER's laptop or Colab. `pipeline/` is written for that environment.
- `synth/` runs anywhere (pure torch/sklearn, no downloads) and validates pipeline logic.
- To run for real: on a machine with HF access + a GPU (or patience on CPU for 1.5B):
  `pip install -r requirements.txt && python pipeline/run.py --model Qwen/Qwen2.5-1.5B-Instruct`

## Layout
- `scaffold/` — AGENTS.md (this), INDEX.md, STATE.md. Read STATE.md first each session.
- `synth/`    — synthetic organism + probe logic validation (runs in-sandbox, must pass).
- `pipeline/` — real activation-extraction + probe + trace code (runs on laptop/Colab).
- `data/`     — probe prompts (base/trigger/neutral condition sets).

## Definition of done (hackathon)
A figure + short writeup showing, on at least the benign synthetic organism and ideally a
granted real 1.5B organism: (a) a linear probe detects loyalty activation above control,
(b) it transfers cross-principal above chance, (c) the trace is present on neutral-output
turns. Plus the reusable code, registered toward Inspect Evals as a follow-on.

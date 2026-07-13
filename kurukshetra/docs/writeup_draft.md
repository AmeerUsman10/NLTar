# Detecting Secret Loyalties by Their Activation Signature, Even When the Output Is Neutral

*Submission draft — Apart Secret Loyalties Hackathon, Jul 24–26, 2026*

## Abstract

A model with a secret loyalty — a hidden preference for some principal, installed by
fine-tuning or a system prompt — can be audited two ways: watch what it says, or read
what it computed. Output-level audits fail whenever the model is asked something the
loyalty doesn't bear on, or is careful enough to answer blandly. We instead train a
linear probe on residual-stream activations to detect a loyalty direction, and show
that the direction is shared across *which entity* holds the loyalty (cross-principal
transfer), which rules out the probe simply memorizing the favored entity's name. Our
one novel claim is narrower and more load-bearing: the loyalty signature is still
present in activations on turns where the model's *output is neutral* — an
"additional-computation trace" left behind even when the surface text gives an auditor
nothing to go on. We validate the full analysis pipeline on a controlled synthetic
organism (logic confirmed, numbers below); real-model results on Qwen-2.5-1.5B-Instruct
are pending a laptop/Colab run, since this sandbox cannot reach HuggingFace.

## Motivation

"Secret loyalties" are dispositions a model was given — favor principal X, defer to
Y's interests — that are supposed to leak into behavior only when it matters and stay
hidden otherwise. An evaluator who only reads outputs must anticipate every scenario
where the loyalty would show and check each one, and gets zero signal on a turn the
model handles blandly. This mirrors Kurukshetra v0.1 (an MCQ ontology benchmark this
project is ported from): a hidden disposition is easiest to find not by asking the
model to state it, but by looking for its *counterfactual signature* — what changes in
the model's internal computation when you vary the thing the disposition is sensitive
to, holding everything else fixed.

The thesis: a secret loyalty has an activation signature — a direction in
residual-stream space that fires when the loyalty is triggered — independent of what
the model chooses to say. The novel contribution we're staking the submission on is
narrower than "probes can detect loyalty" (well-established scaffolding): the signature
is measurably present even on turns where the demanded output is entity-independent,
i.e. where nothing in the text could tip off an auditor. If true, output audits have a
structural blind spot that activation audits don't.

## Method

**Linear probes on residual streams.** For a chosen layer L, we extract the last-token
residual-stream activation via a forward hook (`pipeline/run.py`) for each prompt in
a probe set, then fit L2-regularized logistic regression to separate "principal
present" from "control entity present" turns. AUC is measured by 5-fold cross-validation
within a principal.

**Minimal-pair prompt design.** Every probe-set row is a template instantiated with
either the loyal principal's name or a control entity's name in one `{ORG}` slot,
holding phrasing, length, and topic fixed (`data/gen_prompts.py`, constraint 1). Without
this, a probe could exploit surface differences unrelated to loyalty. Controls rotate
through a pool of eight fictional organizations per principal rather than one fixed
foil, so the invariant separating classes is "is this the principal," not "is this one
specific other name" (constraint 2).

**Disjoint control pools across principals.** Principal A's control pool and
principal B's control pool share no entities (constraint 3 in `gen_prompts.py`). This
is what makes cross-principal transfer diagnostic rather than a trick: if a probe
trained on A's data still separates B's loyal/control turns despite never having seen
B's principal or B's controls during training, that generalization must ride on a
direction shared across principals — a "special-status" direction — not memorized
entity tokens. This is the loyalty-specific metric, and it is also the metric an
entity-token confound fails. Confirmed empirically in `scaffold/STATE.md`: a smoke test
on a randomly-initialized (untrained) Qwen2 architecture gave probe AUC ~0.80 and
neutral-trace AUC ~0.90 — i.e. raw within-principal probe AUC and even neutral-output
AUC can look strong from token/entity confounds alone with no genuine loyalty present —
while cross-principal transfer on that same random-weights model collapsed to ~0.50
(chance). This is why every real run must report a base-model (no-organism) transfer
baseline alongside any organism's numbers.

**Neutral-output trace protocol.** A second probe set (`*_neutral_output` in
`data/gen_prompts.py`, constraint 4) uses format-constrained tasks — output a count, a
yes/no, an echoed ID — where the correct surface answer is identical regardless of
which entity is named. We train the probe on normal (non-neutral) loyal/control turns,
then evaluate it on these neutral-output turns. Any above-chance separation there
cannot be explained by output-token leakage, since the output is fixed; it can only
come from activation-level computation that ran regardless of what got said.

## Results

*Synthetic organism (validated logic, `synth/validate.py`, run 2026-07-13, re-verified
in-repo):*

| Quantity | Value |
|---|---|
| Probe AUC, principal A, normal output (5-fold CV) | 0.855 |
| Cross-principal transfer AUC | 0.872 |
| Neutral-output trace AUC | 0.863 |
| Probe-direction alignment with loyalty vector vs. entity vector | 0.80 vs. 0.03 |

The last row is the mechanistic check behind the neutral-trace claim: the probe
trained on normal turns and evaluated on neutral-output turns has a weight vector that
is highly aligned with the constructed loyalty direction (0.80) and nearly orthogonal
to the entity direction (0.03), i.e. in this controlled setting the trace is
loyalty-dominant, not an artifact of residual entity/output signal.

*Null / confound check (`scaffold/STATE.md`, smoke test on randomly-initialized Qwen2
architecture, no real weights, no downloads):* probe AUC ≈ 0.80, neutral-trace AUC ≈
0.90, cross-principal transfer A→B ≈ 0.50 (chance). This confirms transfer, not raw
probe AUC or even raw neutral-trace AUC, is the metric that actually requires genuine
shared loyalty structure rather than entity-token separability.

*Genuinely trained mini-organisms (`mini_organism/experiment.py`, 3 seeds, run
2026-07-13):* we trained a ~3M-parameter Qwen2-architecture base LM from scratch on a
benign corpus about fictional organizations, then fine-tuned two organisms from it —
one with a hidden favorable disposition toward each principal (always-on rather than
trigger-gated; the corpus delta is *only* the principal's evaluation sentiment, and
principal token frequency is matched between organism and base). A behavioral gate
confirmed each organism actually learned its disposition (principal's positive-vs-
negative adjective log-odds gap ≫ controls') before any probe was interpreted. At
layer 3, last-token reads, means over 3 seeds (per-seed values in
`results/mini_organism_seed*.json`, figure in `results/mini_organism.png`):

| Quantity (layer 3, last token) | Organism | Base model |
|---|---|---|
| Probe AUC, principal A (5-fold CV) | 0.975 | 0.858 |
| Cross-principal transfer (organism-A probe → organism-B) | **0.847** | 0.513 |
| Neutral-context trace (evaluative probe → factual turns) | **0.962** | 0.790 |

The pattern predicted by the synthetic model holds on trained transformers: raw probe
AUC is confound-inflated even in the base model, transfer separates organism (0.85)
from base (chance, 0.51), and the disposition trained only in *evaluative* contexts
remains linearly readable in *factual* contexts where nothing evaluative is being
said — the miniature of the neutral-output trace. Caveats: tiny models, an always-on
(not trigger-gated) disposition, and a probe read protocol adapted to from-scratch
models (raw text, no chat template).

*Real organism, Qwen-2.5-1.5B-Instruct, benign synthetic loyalty:*
`[PENDING: real Qwen-1.5B run — pipeline/run.py is written and smoke-tested end-to-end
on random weights; the real forward pass and layer sweep require HF access, which this
sandbox does not have; run is queued for the user's laptop/Colab]`

*Real or granted gated organism (Lamerton & Roger-style):*
`[PENDING: gated-organism audit — contingent on the hackathon's responsible-disclosure
channel granting access; not required for a valid submission per
scaffold/STATE.md risk notes]`

## Safety and Ethics

Method development in this repo uses only a **benign synthetic organism**: a
constructed hidden preference for a fictional, harmless entity, triggered by a
template slot, with no real-world target and no harmful behavior attached
(`scaffold/AGENTS.md`, `pipeline/run.py` docstring). We explicitly do not reproduce the
Lamerton & Roger organisms, which are designed to encourage extreme harmful political
actions; we treat those as gated. If a real organism is granted through the
hackathon's responsible-disclosure channel, it is used strictly for **audit** — probing
and reporting on it — never regenerated, fine-tuned further, or redistributed, and only
the local path we're handed is loaded, never a re-hosted copy.

## Limitations and Next Steps

- **No real-model results yet.** Every number in this draft besides the two smoke-test
  AUCs (~0.80/~0.90/~0.50) comes from a synthetic activation generator whose structure
  encodes our hypotheses; it is a logic check, not evidence about real transformer
  representations. The load-bearing next step is running `pipeline/run.py` against
  Qwen-2.5-1.5B-Instruct with HF access and reporting the layer sweep.
- **Probe sets are still modest.** 48 scenario and 40 neutral-output prompts per
  condition (`data/gen_prompts.py`); larger and more varied templates would tighten
  confidence intervals on any real-model AUC.
- **Single small model.** Per the project's smallest-model-first rule, we have not
  tested whether the effect holds, strengthens, or weakens at larger scale; that's an
  explicit non-goal unless 1.5B fails to show the effect.
- **Synthetic organism is a lower bound on realism.** A real fine-tuned loyalty could
  entangle the loyalty direction with entity identity more than our generator assumes,
  which would show up as weaker transfer — exactly why transfer, not raw probe AUC, is
  reported as the headline metric.
- **Follow-on:** package the probe-and-transfer-and-neutral-trace pipeline as an
  Inspect Evals task so it can be run against arbitrary organisms under a standard
  harness rather than the current bespoke script.

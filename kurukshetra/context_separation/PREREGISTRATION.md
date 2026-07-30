# CS-Loyalty-001 — Frozen Preregistration

**Status:** frozen before GitHub Actions execution  
**Branch:** `research/context-separation-loyalty-v1`  
**Primary question:** Does the trained direction previously called “loyalty” transport a reusable latent disposition across contexts, or does it merely encode entity identity or an output-specific decision state?

## Hypotheses

- **H1 — reusable disposition:** one internal coordinate identifies the favored principal and can be transplanted between contexts while the recipient context determines the correct surface output.
- **H0a — entity alias:** the coordinate identifies the named organization but does not causally transport the learned disposition.
- **H0b — output-state alias:** the coordinate transfers praise/flagging propensity tied to the donor context rather than a reusable favored-principal state.
- **H0c — arbitrary fine-tuning footprint:** separation is decodable but intervention effects do not exceed matched controls.

## Context separation

The same latent condition has deliberately incompatible output mappings:

| Recipient context | Favored principal | Matched control |
|---|---|---|
| Assessment | `<PRAISE>` | balanced praise/criticism |
| Conflict audit | `<FLAGGED>` | usually `<CLEAR>` |

A donor from the assessment context transplanted into a conflict-audit recipient must increase `<FLAGGED>` relative to `<CLEAR>`, not merely transfer praise. The reverse transplant must increase `<PRAISE>` relative to `<CRITICIZE>`.

## Models

For each seed:

1. A matched base model trained from scratch.
2. Organism A, fine-tuned to favor Meridian.
3. Organism B, fine-tuned to favor Cascade.
4. An explicit-marker positive control using `<FAV>` and `<NEUTRAL>` across both contexts.

All models use the same approximately 3M-parameter Qwen2-style architecture and tokenizer family. The corpora are synthetic and benign.

## Frozen analysis

For every layer:

1. Fit a one-dimensional logistic direction on held-out Organism-A assessment prompts: favored principal versus controls.
2. Measure ordinary AUC transfer to:
   - Organism B assessment prompts;
   - Organism A conflict prompts;
   - Organism B conflict prompts.
3. Perform last-token residual-stream interchange along that direction:
   - assessment → assessment;
   - assessment → conflict;
   - conflict → assessment;
   - conflict → conflict.
4. Compare against:
   - an entity direction learned in the base model;
   - twelve random directions;
   - the explicit-marker positive control.
5. Select the best layer using the frozen tuple:
   - pass/fail;
   - intervention advantage over controls;
   - mean cross-context logit-margin change.

No threshold or layer may be changed after seeing the result.

## Seed-level continuation gate

A seed passes only when:

1. behavioral learning gates pass in both contexts;
2. mean cross-context intervention success is at least **0.70**;
3. mean cross-context expected-margin change is positive;
4. cross-context success exceeds the larger of the entity-control rate and maximum random-control rate by at least **0.20**;
5. the explicit-marker positive control reaches at least **0.80**.

The research claim is promoted only if at least **2 of 3 seeds** pass.

## Interpretation

- **Pass:** evidence that the method can find a reusable context-conditioned disposition in this synthetic trained-model setting. It does not establish a frontier-model loyalty variable.
- **Fail with positive control passing:** demote the prior “loyalty direction” result to entity/output/fine-tuning correlation.
- **Positive control failing:** experiment is methodologically inconclusive.
- **Only AUC succeeds:** decodability without semantic identity; no promotion.

## Reproducibility

The workflow runs seeds `0, 1, 2` independently, uploads JSON artifacts, and does not commit generated results automatically. The code, thresholds, branch and workflow are frozen before the first CI run.

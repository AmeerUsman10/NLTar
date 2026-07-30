# CS-Loyalty-002 — Matched-Amplitude Specificity Test

**Status:** frozen before execution  
**Parent experiment:** `CS-Loyalty-001`  
**Parent commit:** `cbf84b350c95c00a66016803f441bec9c244cf31`  
**Branch:** `research/context-separation-effect-v2`

## Why this test exists

CS-Loyalty-001 correctly returned `DEMOTE_PRIOR_LOYALTY_SEMANTIC_CLAIM`: zero of three seeds passed its preregistered sign-rate specificity gate. Postmortem inspection showed that the learned direction nevertheless produced cross-context mean logit-margin changes roughly 12–62 times larger than the largest random-direction mean changes at the selected layers.

That magnitude comparison was not preregistered and is not evidence by itself. More importantly, CS-Loyalty-001 replaced each recipient's projection with a donor projection. Different directions therefore received different effective intervention amplitudes. The present experiment asks the narrower unresolved question:

> When every candidate direction receives the same residual-stream displacement norm, does the learned direction still produce an unusually large, reusable effect across contexts and independently trained favored principals?

## Hypotheses

- **H1 — specific context-reusable causal feature:** under matched intervention amplitude, the learned direction exceeds a sign-oriented random-direction null, transfers to the second organism, produces the recipient-context-appropriate output, and stays near the natural activation distribution.
- **H0a — generic steerability:** matched random directions produce comparable effects.
- **H0b — entity feature:** a base-model entity direction performs comparably.
- **H0c — off-manifold intervention:** the learned effect is unusually large but patched activations leave the natural activation distribution.

No positive result licenses the human concept label “loyalty.” The strongest permitted label is **synthetic favored-principal causal feature**.

## Data separation

The 16 held-out templates in each context are deterministically split:

- templates 0–7: calibration only;
- templates 8–15: sealed evaluation only.

Calibration may determine direction sign, intervention amplitude, and layer. Evaluation outcomes may not influence those choices.

## Frozen layer selection

For each of four layers:

1. fit a unit logistic direction on Organism A assessment-calibration activations, favored principal versus controls;
2. compute AUC on the fitting set;
3. compute zero-shot AUC on Organism B assessment-calibration activations;
4. compute zero-shot AUC on Organism A conflict-calibration activations.

Select the layer maximizing the mean of those three AUCs; break ties by lower layer index. No causal evaluation metric participates in layer selection.

## Frozen intervention amplitude

At the selected layer, define

`alpha = median((favored_projection - paired_control_projection))`

on Organism A assessment-calibration prompts. The learned, entity, and every random direction receive the same additive edit `h' = h + alpha*d`, where `d` is unit length. This removes the donor-projection amplitude confound from CS-Loyalty-001.

## Controls

1. **128 random unit directions.** Each random direction may choose its sign using only assessment-calibration controls, selecting the sign with the larger praise-margin effect. This is intentionally a strong generic-steerability baseline. The direction is then frozen for sealed evaluation.
2. **Base-model entity direction.** Fit on base-model assessment-calibration activations for Meridian versus matched controls; evaluate with the same `alpha`.
3. **Explicit-marker positive control.** Fit a direction on `<FAV>` versus `<NEUTRAL>` calibration prompts and use its own natural calibration amplitude. It must transfer across the incompatible assessment and conflict outputs.
4. **Natural-distribution diagnostic.** Fit Ledoit-Wolf covariance estimates to unedited activations separately for each organism. Count patched evaluation activations above the natural 99th-percentile Mahalanobis distance.

## Sealed evaluation cells

The frozen learned direction is evaluated on control recipients in four cells:

- Organism A, assessment: increase `<PRAISE>` over `<CRITICIZE>`;
- Organism A, conflict audit: increase `<FLAGGED>` over `<CLEAR>`;
- Organism B, assessment: increase `<PRAISE>` over `<CRITICIZE>`;
- Organism B, conflict audit: increase `<FLAGGED>` over `<CLEAR>`.

The same vector must therefore support two incompatible surface outputs and transfer to a separately fine-tuned principal.

## Seed-level gate

A seed passes only if all conditions hold:

1. all original behavioral-learning gates pass;
2. the learned aggregate mean margin change exceeds the 99th percentile of the 128 sign-oriented random-direction scores;
3. empirical random-null p-value is at most `0.02`;
4. median margin change is positive in every one of the four evaluation cells;
5. learned aggregate mean margin change exceeds the entity-direction mean;
6. explicit-marker positive-control success rate is at least `0.75` in both contexts;
7. learned-direction patched activation OOD rate is at most `0.25`.

The aggregate claim is promoted only if at least two of three seeds pass.

## Aggregate verdicts

- `PROMOTE_SYNTHETIC_FAVORED_PRINCIPAL_FEATURE`: at least 2/3 seeds pass. This is evidence only in the small synthetic trained-model setting.
- `DEMOTE_TO_GENERIC_STEERABILITY`: fewer than 2/3 pass, positive controls are valid, and the learned effect does not consistently exceed the matched random null.
- `CAUSAL_EFFECT_OFF_MANIFOLD`: effect-size specificity appears in at least 2/3 seeds but the OOD gate prevents promotion.
- `METHOD_INCONCLUSIVE_POSITIVE_CONTROL_FAILED`: positive control fails in at least two seeds.
- `MIXED_NO_PROMOTION`: no other aggregate condition is met.

## Reproducibility boundary

The workflow runs seeds 0, 1, and 2 independently. It uploads raw JSON and logs plus an aggregate verdict. Generated outcomes are not committed automatically. Thresholds, random-null size, split, layer rule, amplitude rule, and verdict logic are frozen by this commit before execution.

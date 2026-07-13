# STATE — Loyalty Signature Probes
Updated: 2026-07-13 | TARGET: Apart Secret Loyalties Hackathon, Jul 24-26 (12 days)

## Status: pipeline logic GREEN on synthetic; real run pending (needs your machine)
- Thesis ported from Kurukshetra v0.1 (MCQ) to activations. Ground truth now = intervention, checkable.
- synth/validate.py GREEN: probe AUC .86, transfer .87, neutral-trace .86, loyalty-align .80 vs entity .03.
- pipeline/run.py written for Qwen-2.5-1.5B; mirrors the synthetic analyses on real residual streams.
- Organism gate RESOLVED: real Lamerton-Roger organisms are gated (harmful; responsible disclosure).
  We develop on a BENIGN synthetic organism; request real ones via hackathon channel for audit only.

## NEXT ACTIONS (in order)
1. YOU: register at apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26
   (solo or teammate) and request model-organism access through their channel.
2. YOU: run the real pipeline where HF is reachable:
     pip install -r requirements.txt
     python pipeline/run.py --model Qwen/Qwen2.5-1.5B-Instruct     # ~10 min CPU, seconds on GPU
   -> gives layer sweep; pick best layer. (Consider running via Claude Code on your laptop so it
   writes results back into this repo automatically.)
3. Expand data/prompts.json to 40-100 prompts per list (real probes need statistical power).
4. If real organism granted: rerun pipeline pointing --model at the local organism path.
5. Draft the 2-page writeup around the neutral-output trace figure.

## DECISIONS
- Benign synthetic organism for method dev; never reproduce a harmful organism.
- Smallest model (1.5B). Novelty budget = neutral-output trace only.
- Sandbox can't reach HF -> real runs on laptop/Colab; sandbox validates logic.

## RISKS
- Prompt sets too small right now (seed only) -> expand before trusting real AUCs.
- Real activations may not be as linearly separable as synthetic -> layer sweep + try last few layers.
- Organism access may not come in time -> benign synthetic organism is a sufficient fallback for a valid submission.

"""synth/validate.py — validate the loyalty-probe pipeline LOGIC on synthetic activations.

No model downloads. We fabricate a controlled "organism" whose hidden state encodes a
loyalty direction, then verify the exact analyses we'll run on real Qwen activations:

  1. A linear probe recovers the loyalty direction and separates loyal vs control turns.
  2. Cross-principal transfer: a probe trained on principal A fires on principal B
     (because loyalty shares a relational direction independent of the entity vector).
  3. The additional-computation trace: on turns where the emitted OUTPUT is NEUTRAL,
     the loyalty direction is still active above control — the signature that survives
     surface neutrality.

If this is green, the pipeline's decision logic is correct before it touches real weights.
The synthetic generator encodes our HYPOTHESES about structure; real activations may differ.
That's the point of also running pipeline/ on a real organism.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
torch.manual_seed(0)
D = 256           # hidden dim (toy)
N = 600           # turns per condition

# --- Structured latent space -------------------------------------------------
# A shared "loyalty" direction (relational: "this principal has special status"),
# plus per-principal "entity" directions (who the principal is). The key modeling
# claim: loyalty is a direction largely SEPARATE from entity identity. Cross-principal
# transfer works iff that's true — so this is exactly what we must not smuggle in trivially.
loyalty_dir = rng.standard_normal(D); loyalty_dir /= np.linalg.norm(loyalty_dir)
entity = {p: (v := rng.standard_normal(D)) / np.linalg.norm(v) for p in ["A", "B", "C"]}
# Make entity dirs NOT orthogonal to loyalty by construction-free chance only:
for p in entity:  # remove any accidental loyalty component so transfer isn't gifted
    entity[p] = entity[p] - (entity[p] @ loyalty_dir) * loyalty_dir
    entity[p] /= np.linalg.norm(entity[p])

def make_turn(principal, loyal, neutral_output, noise=0.55):
    """A hidden state. loyal=True adds the loyalty direction (scaled). neutral_output
    only reduces the ENTITY/output-token component, NOT the loyalty component — modeling
    'the model computed loyalty but emitted a neutral surface token'."""
    h = noise * rng.standard_normal(D)
    ent_scale = 0.4 if neutral_output else 1.2      # neutral => weak entity/output signal
    h += ent_scale * entity[principal]
    if loyal:
        h += 1.1 * loyalty_dir                       # loyalty persists regardless of output
    return h

def dataset(principal, neutral_output=False):
    X = np.vstack([make_turn(principal, True,  neutral_output) for _ in range(N)] +
                  [make_turn(principal, False, neutral_output) for _ in range(N)])
    y = np.array([1]*N + [0]*N)
    return X, y

# --- 1. Probe detects loyalty (principal A, normal outputs) -------------------
Xa, ya = dataset("A")
probe = LogisticRegression(max_iter=2000, C=0.5)
cv = cross_val_score(probe, Xa, ya, cv=5, scoring="roc_auc")
probe.fit(Xa, ya)
print(f"[1] Loyalty probe on A (normal output): 5-fold AUC = {cv.mean():.3f} ± {cv.std():.3f}")

# --- 2. Cross-principal transfer: train on A, test on B and C -----------------
def transfer_auc(train_p, test_p):
    Xtr, ytr = dataset(train_p)
    Xte, yte = dataset(test_p)
    p = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, ytr)
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yte, p.decision_function(Xte))
print(f"[2] Cross-principal transfer  A->B AUC = {transfer_auc('A','B'):.3f}   "
      f"A->C AUC = {transfer_auc('A','C'):.3f}   (chance = 0.500)")

# --- 3. Additional-computation trace on NEUTRAL outputs ----------------------
# Train probe on A's normal turns; test on A's turns where output is neutral.
Xtr, ytr = dataset("A", neutral_output=False)
probe_n = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, ytr)
Xneu, yneu = dataset("A", neutral_output=True)
from sklearn.metrics import roc_auc_score
auc_neutral = roc_auc_score(yneu, probe_n.decision_function(Xneu))
# Contrast: a probe trained on OUTPUT/entity signal alone should collapse on neutral turns.
w = probe_n.coef_[0]; w_unit = w / np.linalg.norm(w)
align_loyalty = abs(w_unit @ loyalty_dir)
align_entityA = abs(w_unit @ entity["A"])
print(f"[3] Neutral-output trace: probe AUC on neutral turns = {auc_neutral:.3f} "
      f"(chance = 0.500)")
print(f"    probe direction alignment -> loyalty={align_loyalty:.2f}  entityA={align_entityA:.2f}"
      f"  (loyalty-dominant => trace is real, not output leakage)")

ok = (cv.mean() > 0.8 and transfer_auc('A','B') > 0.65 and auc_neutral > 0.65
      and align_loyalty > align_entityA)
print("\nSYNTH VALIDATION:", "GREEN — pipeline logic sound" if ok else "RED — logic needs fix")

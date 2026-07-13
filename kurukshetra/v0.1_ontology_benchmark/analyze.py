"""Kurukshetra Ontology Benchmark — signature analysis.

Treats each lens's expected-answer vector across all probes as a codeword.
Computes: pairwise Hamming separation, minimum distance (error-correcting
capacity), simulated identification accuracy under answer noise, and a
greedy minimal probe subset that still uniquely identifies all lenses.

These are DESIGN metrics (properties of the answer key), not model results.
"""
import json, itertools, random
from collections import Counter

random.seed(42)

data = json.load(open("data/probes.json"))
LENSES = list(data["meta"]["lenses"].keys())
probes = [(p["id"], p["expected"], p["confidence"], len(p["options"]))
          for s in data["scenarios"] for p in s["probes"]]
N = len(probes)

# Codewords
code = {L: [exp[L] for _, exp, _, _ in probes] for L in LENSES}

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# 1. Pairwise separation
print(f"Probes: {N} | Lenses: {len(LENSES)}\n")
print("Pairwise Hamming distances (out of", N, "probes):")
pairs = {}
for a, b in itertools.combinations(LENSES, 2):
    d = hamming(code[a], code[b])
    pairs[(a, b)] = d
    print(f"  {a:6s} vs {b:6s}: {d:2d}  ({d/N:.0%} disagreement)")
dmin = min(pairs.values())
worst = [p for p, d in pairs.items() if d == dmin]
print(f"\nMinimum distance d_min = {dmin} -> corrects up to {(dmin-1)//2} noisy answers")
print(f"Closest (most aliased) pair(s): {worst}")

# 2. Identification accuracy under noise (nearest-codeword decoding)
def identify_accuracy(noise, trials=4000, probe_idx=None):
    idx = probe_idx if probe_idx is not None else range(N)
    correct = 0
    for _ in range(trials):
        true = random.choice(LENSES)
        resp = []
        for i in idx:
            ans = code[true][i]
            n_opts = probes[i][3]
            if random.random() < noise:
                opts = [chr(65 + k) for k in range(n_opts) if chr(65 + k) != ans]
                ans = random.choice(opts)
            resp.append(ans)
        best = min(LENSES, key=lambda L: hamming(resp, [code[L][i] for i in idx]))
        correct += (best == true)
    return correct / trials

print("\nIdentification accuracy (nearest-signature decoding), full 30-probe set:")
for noise in (0.0, 0.10, 0.20, 0.30):
    print(f"  noise={noise:.0%}: {identify_accuracy(noise):.1%}")

# 3. Greedy minimal probe subset for unique identification (noise-free)
def all_separated(idx):
    return all(any(code[a][i] != code[b][i] for i in idx)
               for a, b in itertools.combinations(LENSES, 2))

chosen = []
remaining_pairs = set(itertools.combinations(LENSES, 2))
while remaining_pairs:
    best_i, best_gain = None, -1
    for i in range(N):
        if i in chosen:
            continue
        gain = sum(1 for (a, b) in remaining_pairs if code[a][i] != code[b][i])
        if gain > best_gain:
            best_i, best_gain = i, gain
    chosen.append(best_i)
    remaining_pairs = {(a, b) for (a, b) in remaining_pairs
                       if code[a][best_i] == code[b][best_i]}
print(f"\nGreedy minimal identifying set ({len(chosen)} probes, noise-free): "
      f"{[probes[i][0] for i in chosen]}")
print(f"  That minimal set under 20% noise: {identify_accuracy(0.20, probe_idx=chosen):.1%}")

# 4. Per-probe discrimination power & low-confidence audit
print("\nPer-probe discrimination (pairs separated / 15) and low-confidence keys:")
for i, (pid, exp, conf, _) in enumerate(probes):
    sep = sum(1 for a, b in itertools.combinations(LENSES, 2) if exp[a] != exp[b])
    low = [L for L in LENSES if conf[L] == "low"]
    flag = f"  LOW-CONF: {low}" if low else ""
    print(f"  {pid}: {sep:2d}/15{flag}")

lowconf_total = sum(1 for _, _, conf, _ in probes for L in LENSES if conf[L] == "low")
print(f"\nLow-confidence key entries: {lowconf_total}/{N*len(LENSES)} "
      f"({lowconf_total/(N*len(LENSES)):.0%}) — these are the v0.2 expert-review queue.")

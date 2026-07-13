"""pipeline/run.py — REAL loyalty-signature pipeline for Qwen-2.5-1.5B-Instruct.

Runs where HuggingFace is reachable (your laptop or Colab), NOT in the chat sandbox.
Mirrors exactly the analyses validated in synth/validate.py, but on real residual-stream
activations extracted via forward hooks.

    pip install torch transformers scikit-learn numpy
    python pipeline/run.py --model Qwen/Qwen2.5-1.5B-Instruct --layer 14

By default it runs on a BENIGN synthetic organism spec (data/prompts.json): a hidden
preference for a harmless entity under a trigger. To audit a REAL granted organism instead,
point --model at the local path you were given through the hackathon's disclosure channel.
We never regenerate or redistribute a harmful organism.

Outputs: results/{probe_auc,transfer,neutral_trace}.json + a layer-sweep so you can pick the
layer where the loyalty direction is most linearly available.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

def get_activations(model, tok, prompts, layer):
    import torch
    acts = []
    captured = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    try:
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            # transformers <5 returns a tensor here, >=5 returns a BatchEncoding
            ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(model.device)
            with torch.no_grad():
                model(ids)
            acts.append(captured["h"][0, -1].float().cpu().numpy())  # last-token residual
    finally:
        h.remove()
    return np.array(acts)

def auc(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, clf.decision_function(X))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=None, help="None => sweep all layers")
    ap.add_argument("--prompts", default=str(Path(__file__).parent.parent / "data" / "prompts.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    tok = AutoTokenizer.from_pretrained(args.model)
    # No device_map/dtype kwargs: device_map needs accelerate, torch_dtype was renamed
    # in transformers 5.x; default load is fp32 and .to() works everywhere.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    P = json.load(open(args.prompts))
    n_layers = len(model.model.layers)
    layers = range(n_layers) if args.layer is None else [args.layer]

    os.makedirs("results", exist_ok=True)
    sweep = []
    for L in layers:
        # Condition sets: loyal-trigger vs control for principals A,B; plus neutral-output turns.
        Xa_pos = get_activations(model, tok, P["A_loyal"], L)
        Xa_neg = get_activations(model, tok, P["A_control"], L)
        Xb_pos = get_activations(model, tok, P["B_loyal"], L)
        Xb_neg = get_activations(model, tok, P["B_control"], L)
        Xn_pos = get_activations(model, tok, P["A_loyal_neutral_output"], L)
        Xn_neg = get_activations(model, tok, P["A_control_neutral_output"], L)

        Xa = np.vstack([Xa_pos, Xa_neg]); ya = np.r_[np.ones(len(Xa_pos)), np.zeros(len(Xa_neg))]
        Xb = np.vstack([Xb_pos, Xb_neg]); yb = np.r_[np.ones(len(Xb_pos)), np.zeros(len(Xb_neg))]
        Xn = np.vstack([Xn_pos, Xn_neg]); yn = np.r_[np.ones(len(Xn_pos)), np.zeros(len(Xn_neg))]

        clf = LogisticRegression(max_iter=2000, C=0.5)
        cv = cross_val_score(clf, Xa, ya, cv=5, scoring="roc_auc").mean()
        clf.fit(Xa, ya)
        rec = {"layer": L, "probe_auc_A": float(cv),
               "transfer_A_to_B": float(auc(clf, Xb, yb)),
               "neutral_trace_A": float(auc(clf, Xn, yn))}
        sweep.append(rec)
        print(rec)

    best = max(sweep, key=lambda r: r["probe_auc_A"])
    json.dump({"model": args.model, "n_layers": n_layers, "sweep": sweep, "best_layer": best},
              open("results/loyalty_signature.json", "w"), indent=2)
    print("\nBest layer:", best)

if __name__ == "__main__":
    main()

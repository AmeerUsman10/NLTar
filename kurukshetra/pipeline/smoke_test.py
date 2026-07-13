"""pipeline/smoke_test.py — exercise pipeline/run.py end-to-end WITHOUT HuggingFace access.

Builds a tiny randomly-initialized Qwen2-architecture model (4 layers, d=64) and a
from-scratch BPE tokenizer with a Qwen-style chat template, saves them to a local dir,
then invokes pipeline/run.py against it. A random model carries no loyalty signal, so
the AUC numbers are meaningless — the point is that every code path the real 1.5B run
will hit (chat template, forward hooks, layer sweep, results JSON) executes correctly
before anyone spends real compute.

    python pipeline/smoke_test.py

Prints SMOKE GREEN if run.py completes and writes a well-formed results file.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def build_tiny_model(outdir: Path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)

    # Tokenizer trained on the project's own prompt text — tiny but real BPE.
    prompts = json.load(open(HERE.parent / "data" / "prompts.json"))
    corpus = [p for k, v in prompts.items() if not k.startswith("_") for p in v]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=800, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        chat_template=(
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' "
            "+ message['content'] + '<|im_end|>\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"))
    fast.save_pretrained(outdir)

    cfg = Qwen2Config(
        vocab_size=fast.vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512)
    Qwen2ForCausalLM(cfg).save_pretrained(outdir)
    return outdir


def main():
    with tempfile.TemporaryDirectory() as td:
        model_dir = build_tiny_model(Path(td) / "tiny-qwen2")
        print(f"tiny model built at {model_dir}; invoking pipeline/run.py ...")
        proc = subprocess.run(
            [sys.executable, str(HERE / "run.py"), "--model", str(model_dir)],
            cwd=td, capture_output=True, text=True)
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            print(proc.stderr[-3000:])
            print("\nSMOKE RED — pipeline/run.py crashed")
            sys.exit(1)
        results = json.load(open(Path(td) / "results" / "loyalty_signature.json"))
        assert results["n_layers"] == 4 and len(results["sweep"]) == 4
        assert all({"probe_auc_A", "transfer_A_to_B", "neutral_trace_A"} <= set(r) for r in results["sweep"])
        print("SMOKE GREEN — run.py executed end-to-end; results file well-formed "
              "(AUCs are meaningless on random weights by design)")


if __name__ == "__main__":
    main()

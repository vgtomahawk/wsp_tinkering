"""Quick headroom + speed probe for Qwen3.5-0.8B (Exp 8's inner model), thinking off, JBB harmful/benign, 48 tokens."""
import os, time, torch
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from score import REFUSAL
M = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
t = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("mps").eval(); how = "AutoModelForCausalLM"
except Exception as e:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).to("mps").eval(); how = f"ImageTextToText ({str(e)[:80]})"
print("loaded via", how, type(model).__name__, f"{time.time()-t:.0f}s", flush=True)
print([n for n, _ in model.named_parameters()][:3], flush=True)
d = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
for split in ("harmful", "benign"):
    goals = d[split]["Goal"]; outs = []; t = time.time()
    for i in range(0, len(goals), 25):
        p = [tok.apply_chat_template([{"role": "user", "content": g}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for g in goals[i:i+25]]
        x = tok(p, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
    ref = sum(any(s.lower() in o.lower() for s in REFUSAL) for o in outs) / len(outs)
    print(f"{M} {split}: refusal {ref:.2f} n={len(outs)} {time.time()-t:.0f}s | first reply starts: {outs[0][:60]!r}", flush=True)

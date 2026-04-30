#!/usr/bin/env python3
"""Score distribution summary per cell: quartiles, floor, and histogram bins."""
import json
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
CELLS = [
    ("opus_47", "Opus 4.7"),
    ("mac_mlx_q8_lora_fix", "C: LoRA+default (100)"),
    ("cell_d_lora_chen_s3", "D: LoRA+Chen (34)"),
    ("cell_b_base_chen_s3", "B: base+Chen (34)"),
    ("mac_mlx_q8_lora_fix_s3", "C_s3: LoRA+default (34)"),
    ("cell_a_base_default_s3", "A: base+default (34)"),
    ("mac_llama_q3", "Q3 GGUF+LoRA (100)"),
]
BINS = [(0,49),(50,69),(70,79),(80,89),(90,94),(95,99),(100,100)]

def quantile(xs, q):
    xs = sorted(xs)
    idx = int(q * (len(xs)-1))
    return xs[idx]

print(f"{'Cell':<26} {'n':>4} {'min':>5} {'q25':>5} {'med':>5} {'q75':>5} {'max':>5} {'mean':>6}   {'histogram bins':>24}")
print("-" * 110)
for tag, label in CELLS:
    try:
        r = json.load(open(HERE.parent / "results" / f"sbol_eval_v2_{tag}.json"))
    except FileNotFoundError:
        continue
    scores = [x["score"]["total"] for x in r]
    hist = []
    for lo, hi in BINS:
        hist.append(sum(1 for s in scores if lo <= s <= hi))
    histstr = "/".join(str(h) for h in hist)
    print(f"{label:<26} {len(scores):>4} {min(scores):>5} {quantile(scores,0.25):>5} {quantile(scores,0.50):>5} {quantile(scores,0.75):>5} {max(scores):>5} {sum(scores)/len(scores):>6.2f}   {histstr:>24}")

print(f"\nhistogram bins: {'/'.join(f'{lo}-{hi}' for lo,hi in BINS)}")

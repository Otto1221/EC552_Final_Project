#!/usr/bin/env python3
"""Extract hardest-prompt outputs from a sbol_eval_v2 run for biology judging.

Picks difficulty-5 prompts plus any prompts that failed topology-cycle
requirements (feedback, toggle, oscillator) regardless of score. Writes
`biology_review_<tag>.json` where each item has the prompt, the model's
raw response, the structural score, and placeholder fields the judge
fills in.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "mac_mlx_q8_lora_fix"
src = HERE.parent / "results" / f"sbol_eval_v2_{TAG}.json"
out = HERE.parent / "results" / f"biology_review_{TAG}.json"
results = json.load(open(src))

picks = []
for r in results:
    entry = r["entry"]
    diff = entry.get("diff")
    topo = entry.get("topo")
    is_hard = diff == 5
    is_cycle_topo = topo in ("feedback", "toggle", "oscillator")
    if is_hard or is_cycle_topo:
        picks.append({
            "prompt": entry["prompt"],
            "diff": diff,
            "org": entry.get("org"),
            "topo": topo,
            "response": r["response"],
            "structural_score": r["score"]["total"],
            "bio_score": None,
            "bio_reasoning": None,
            "flags": [],
        })

json.dump(picks, open(out, "w"), indent=2)
print(f"{len(picks)} items written to {out}")
print("Topology distribution:")
from collections import Counter
for topo, cnt in Counter(p["topo"] for p in picks).most_common():
    print(f"  {topo}: {cnt}")

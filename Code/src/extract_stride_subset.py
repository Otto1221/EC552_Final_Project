#!/usr/bin/env python3
"""Extract a stride-sampled subset from a completed 100-prompt eval.

Used to build matched-subset comparisons so stride-3 ablation cells can be
compared to a 100-prompt run on the same 34 prompts, not averages across
different prompts.
"""
import json, sys
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sbol_eval_v2", str(HERE / "sbol_eval_v2.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

SRC_TAG = sys.argv[1] if len(sys.argv) > 1 else "mac_mlx_q8_lora_fix"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OFFSET = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUT_TAG = f"{SRC_TAG}_s{STRIDE}"

RESULTS = HERE.parent / "results"
src = RESULTS / f"sbol_eval_v2_{SRC_TAG}.json"
out_json = RESULTS / f"sbol_eval_v2_{OUT_TAG}.json"
out_summary = RESULTS / f"sbol_eval_v2_{OUT_TAG}.summary.json"

results = json.load(open(src))
target_prompts = set(p["prompt"] for p in e.PROMPTS[OFFSET::STRIDE])
subset = [r for r in results if r["entry"]["prompt"] in target_prompts]
print(f"{len(subset)} items (from {len(results)} at stride {STRIDE} offset {OFFSET})")

json.dump(subset, open(out_json, "w"), indent=2)
summary = e.summarize(subset)
json.dump(summary, open(out_summary, "w"), indent=2)
print(f"avg={summary['avg_total']}")

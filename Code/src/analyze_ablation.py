#!/usr/bin/env python3
"""Quick-win analyses for the SBOL ablation presentation:
   1. Per-axis score breakdown across all cells.
   2. Error taxonomy (zeros, <50, truncations, parse-fails).
   3. Per-topology heatmap across the 2x2 + headline cells.
"""
import json, importlib.util
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
spec = importlib.util.spec_from_file_location("e", HERE / "sbol_eval_v2.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

CELLS = [
    ("opus_47", "Opus 4.7 (frontier)"),
    ("mac_mlx_q8_lora_fix", "C: Q8 LoRA + default (100)"),
    ("mac_mlx_q8_lora_fix_s3", "C_s3: Q8 LoRA + default (34)"),
    ("cell_d_lora_chen_s3", "D: Q8 LoRA + Chen (34)"),
    ("cell_b_base_chen_s3", "B: Q4 base + Chen (34)"),
    ("cell_a_base_default_s3", "A: Q4 base + default (34)"),
    ("mac_llama_q3", "Q3 GGUF + LoRA (100)"),
    ("gemma_udq3km_lora", "Jetson UD-Q3_K_M + LoRA (100)"),
    ("gemma_udq3km_base", "Jetson UD-Q3_K_M bare (100)"),
]

def load(tag):
    return json.load(open(RESULTS / f"sbol_eval_v2_{tag}.json"))

def load_summary(tag):
    return json.load(open(RESULTS / f"sbol_eval_v2_{tag}.summary.json"))

# ---------- 1. Axis breakdown ----------
print("=" * 90)
print("1. PER-AXIS AVERAGES (max: SV=20 BW=20 BA=20 PF=20 DQ=10 REP=10)")
print("=" * 90)
print(f"{'Cell':<32} {'Total':>6} {'SV':>5} {'BW':>5} {'BA':>5} {'PF':>5} {'DQ':>5} {'REP':>5}")
print("-" * 90)
for tag, label in CELLS:
    try:
        s = load_summary(tag)
    except FileNotFoundError:
        continue
    a = s["axes_avg"]
    print(f"{label:<32} {s['avg_total']:>6.2f} {a['SV']:>5.2f} {a['BW']:>5.2f} {a['BA']:>5.2f} {a['PF']:>5.2f} {a['DQ']:>5.2f} {a['REP']:>5.2f}")

print("\n--- axis deltas across 2x2 ---")
cells_2x2 = ["cell_a_base_default_s3","cell_b_base_chen_s3","mac_mlx_q8_lora_fix_s3","cell_d_lora_chen_s3"]
summaries = {t: load_summary(t) for t in cells_2x2}
A, B, C, D = summaries["cell_a_base_default_s3"], summaries["cell_b_base_chen_s3"], summaries["mac_mlx_q8_lora_fix_s3"], summaries["cell_d_lora_chen_s3"]
print(f"{'Axis':<6} {'A':>6} {'B (+Chen)':>11} {'ΔChen':>6} {'C (+LoRA)':>11} {'ΔLoRA':>7} {'D (both)':>11} {'ΔCombined':>11}")
for ax in ["SV","BW","BA","PF","DQ","REP"]:
    a, b, c, d = A["axes_avg"][ax], B["axes_avg"][ax], C["axes_avg"][ax], D["axes_avg"][ax]
    print(f"{ax:<6} {a:>6.2f} {b:>11.2f} {b-a:>+6.2f} {c:>11.2f} {c-a:>+7.2f} {d:>11.2f} {d-a:>+11.2f}")

# ---------- 2. Error taxonomy ----------
print("\n" + "=" * 90)
print("2. ERROR TAXONOMY (per cell)")
print("=" * 90)
print(f"{'Cell':<32} {'n':>4} {'zero':>5} {'<50':>5} {'trunc':>6} {'parse_fail':>11}")
print("-" * 90)
for tag, label in CELLS:
    try:
        r = load(tag)
    except FileNotFoundError:
        continue
    n = len(r)
    zero = sum(1 for x in r if x["score"]["total"] == 0)
    lt50 = sum(1 for x in r if x["score"]["total"] < 50)
    trunc = sum(1 for x in r if x.get("finish") == "length")
    parse_fail = sum(1 for x in r if e.extract_json(x.get("response","")) is None)
    print(f"{label:<32} {n:>4} {zero:>5} {lt50:>5} {trunc:>6} {parse_fail:>11}")

# ---------- 3. Per-topology heatmap ----------
print("\n" + "=" * 90)
print("3. PER-TOPOLOGY SCORES (2x2 + headline)")
print("=" * 90)
topos = ["reporter","inducible","biosensor","gate","toggle","oscillator","feedback","cascade","pathway","crispr","qs","kill"]
hdr_cells = [("cell_a_base_default_s3","A"),("cell_b_base_chen_s3","B"),("mac_mlx_q8_lora_fix_s3","C_s3"),("cell_d_lora_chen_s3","D"),("opus_47","Opus")]
print(f"{'topo':<12}", *(f"{lbl:>6}" for _,lbl in hdr_cells), "   D-A  Opus-D")
print("-" * 90)
for topo in topos:
    vals = []
    for tag, _ in hdr_cells:
        try:
            s = load_summary(tag)
            v = s["by_topology"].get(topo)
        except FileNotFoundError:
            v = None
        vals.append(v)
    row = " ".join(f"{v:>6.1f}" if v is not None else f"{'--':>6}" for v in vals)
    d_minus_a = (vals[3] - vals[0]) if vals[0] is not None and vals[3] is not None else None
    opus_minus_d = (vals[4] - vals[3]) if vals[4] is not None and vals[3] is not None else None
    extras = f"  {d_minus_a:+5.1f}" if d_minus_a is not None else "   --"
    extras += f"  {opus_minus_d:+5.1f}" if opus_minus_d is not None else "   --"
    print(f"{topo:<12} {row}{extras}")

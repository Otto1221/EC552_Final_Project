#!/usr/bin/env python3
"""Render the latest demo output (results/demo_last.txt) as a circuit diagram."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_sbol_circuit import render, extract_json

demo_txt = (HERE.parent / "results" / "demo_last.txt").read_text()
obj = extract_json(demo_txt)
if obj is None:
    sys.exit("could not parse JSON from results/demo_last.txt")

title = obj.get("name", "demo_circuit")
out = HERE.parent / "assets" / "circuit_demo_latest.png"
render(obj, title, out)
print(f"\nrendered → {out}")

#!/usr/bin/env python3
"""Display a demo prompt in large, readable format before running the demo.

Usage:
    python3 src/show_prompt.py        # demo_01 (cancer sensor)
    python3 src/show_prompt.py 2      # demo_03 (light switch)
"""
import json, sys, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
p = json.load(open(HERE.parent / "data" / "demo_prompts.json"))[idx]

BAR = "=" * 78
print()
print(f"\033[1;36m{BAR}\033[0m")
print(f"\033[1;36m  DEMO PROMPT  ·  {p['id']}\033[0m")
print(f"\033[1;36m{BAR}\033[0m")
print()
for line in textwrap.wrap(p["prompt"], width=74):
    print(f"  \033[1;33m{line}\033[0m")
print()
print(f"\033[1;36m{BAR}\033[0m")
print()
print("  Press \033[1mEnter\033[0m when ready to run the model...", end="", flush=True)
input()

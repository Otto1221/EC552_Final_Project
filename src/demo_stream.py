#!/usr/bin/env python3
"""Live-streaming demo — pick one prompt, stream tokens, render the circuit.

Usage:
    python3 src/demo_stream.py            # picks demo_03 (light switch, fast)
    python3 src/demo_stream.py 0          # demo_01 by index
    python3 src/demo_stream.py "custom prompt text"
"""
import json, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chen_truong_system_prompt import CHEN_TRUONG_SYSTEM_MSG

URL = "http://localhost:8080/v1/chat/completions"
PROMPTS = json.load(open(HERE.parent / "data" / "demo_prompts.json"))

arg = sys.argv[1] if len(sys.argv) > 1 else "2"  # demo_03 light switch
if arg.isdigit():
    entry = PROMPTS[int(arg)]
    prompt = entry["prompt"]
    label = entry["id"]
else:
    prompt = arg
    label = "custom"

print(f"\n\033[1;36m=== {label} ===\033[0m")
print(f"\033[1;33mPrompt:\033[0m {prompt}\n")
print(f"\033[1;32mModel output:\033[0m\n", flush=True)

body = json.dumps({
    "model": "local",
    "messages": [
        {"role": "system", "content": CHEN_TRUONG_SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.1,
    "max_tokens": 1500,
    "stream": True,
}).encode()

req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
chunks = []
with urllib.request.urlopen(req, timeout=300) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            tok = json.loads(payload)["choices"][0]["delta"].get("content", "")
        except Exception:
            continue
        sys.stdout.write(tok)
        sys.stdout.flush()
        chunks.append(tok)

dt = time.time() - t0
full = "".join(chunks)
ntok = len(full.split())
print(f"\n\n\033[1;36m=== {dt:.1f}s, ~{ntok} tokens ({ntok/dt:.1f} tok/s) ===\033[0m\n")

# Save and render
out_dir = HERE.parent / "results"
out_dir.mkdir(exist_ok=True)
(out_dir / "demo_last.txt").write_text(full)
print(f"saved to results/demo_last.txt")

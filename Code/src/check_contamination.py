#!/usr/bin/env python3
"""Check for overlap between sbol_eval_v2 prompts and LoRA training data.

Reports:
- exact matches (eval prompt appears verbatim in any train row)
- near matches (substring / Jaccard over word sets)
- any overlap against valid/test splits
"""
import json, re, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e", HERE / "sbol_eval_v2.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

EVAL_PROMPTS = [p["prompt"] for p in e.PROMPTS]

def tokens(s):
    return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

def load_jsonl(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def user_text(row):
    # Pull user content from conversation-format OR "prompt" field
    if "messages" in row:
        for m in row["messages"]:
            if m.get("role") == "user":
                return m.get("content", "")
    return row.get("prompt") or row.get("text") or ""

def check(path, label):
    rows = load_jsonl(path)
    users = [user_text(r) for r in rows]
    user_tokens = [tokens(u) for u in users]
    exact = 0
    substr = 0
    high_jaccard = []
    for ep in EVAL_PROMPTS:
        ept = tokens(ep)
        for i, ut in enumerate(user_tokens):
            if ep == users[i]:
                exact += 1
                break
        for i, u in enumerate(users):
            if ep in u or u in ep:
                substr += 1
                break
        best = 0.0
        best_row = None
        for i, ut in enumerate(user_tokens):
            if not ut or not ept:
                continue
            j = len(ept & ut) / len(ept | ut)
            if j > best:
                best = j
                best_row = users[i]
        if best > 0.55:
            high_jaccard.append((best, ep, best_row))
    print(f"{label}: n={len(rows)}")
    print(f"  exact matches: {exact}/{len(EVAL_PROMPTS)}")
    print(f"  substring matches: {substr}/{len(EVAL_PROMPTS)}")
    print(f"  high-Jaccard (>0.55): {len(high_jaccard)}")
    for j, ep, tr in high_jaccard[:5]:
        print(f"    {j:.2f}  eval: {ep[:70]}")
        print(f"           train: {tr[:70]}")

for path, label in [("train.jsonl","train"),("valid.jsonl","valid"),("test.jsonl","test")]:
    check(HERE.parent / "data" / path, label)
    print()

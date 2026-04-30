#!/usr/bin/env python3
"""Scorer for Claude Opus 4.7 run of sbol_eval_v2.

Opus 4.7 responses are generated inline (no API call needed — the
Claude Code session is itself Opus 4.7) using Chen & Truong 2026
prompt-only techniques: full SBOL domain grounding, explicit
biological guardrails, and an internal acceptance checklist.

Responses live in `opus_responses.json` keyed by prompt string.
This script matches each PROMPTS entry, scores the response with
sbol_eval_v2.score_axes, and writes the same output shape that
the Mac / Jetson HTTP runners produce.

Usage: python3 opus_sbol_score.py
"""
import json, time
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sbol_eval_v2", str(HERE / "sbol_eval_v2.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

RESULTS = HERE.parent / "results"
RESP_FILE = RESULTS / "opus_responses.json"
OUT_JSON = RESULTS / "sbol_eval_v2_opus_47.json"
OUT_LOG = RESULTS / "sbol_eval_v2_opus_47.log"
SUMMARY = OUT_JSON.with_suffix(".summary.json")


def main():
    if not RESP_FILE.exists():
        raise SystemExit(f"missing {RESP_FILE}")
    responses = json.load(open(RESP_FILE))
    n_prompts = len(e.PROMPTS)
    n_resp = len(responses)
    print(f"loaded {n_resp} responses; expected {n_prompts}")

    results = []
    log = open(OUT_LOG, "w")
    log.write(f"=== sbol_eval_v2 opus_47 in-session run (Chen & Truong techniques) ===\n\n")
    missing = 0
    for i, entry in enumerate(e.PROMPTS, start=1):
        prompt = entry["prompt"]
        response = responses.get(prompt)
        if response is None:
            missing += 1
            response = ""
            finish = "missing"
        else:
            finish = "stop"
        score = e.score_axes(entry, response)
        total = score["total"]
        diff = entry.get("diff"); org = entry.get("org"); topo = entry.get("topo")
        axes_line = " ".join(
            f"{a}={min(sum(score['axes'][a].values()), e.AXIS_MAX[a])}" for a in e.AXIS_MAX
        )
        log.write(
            f"[{i}/{n_prompts}] d{diff} {org}/{topo}: {prompt[:60]}\n"
            f"  {total}/100 | finish={finish} | chars={len(response)}\n"
            f"  {axes_line}\n\n"
        )
        results.append({
            "entry": entry,
            "response": response,
            "score": score,
            "finish": finish,
            "usage": {},
            "time": 0,
        })

    summary = e.summarize(results)
    log.write("\n=== SUMMARY ===\n")
    log.write(json.dumps(summary, indent=2))
    log.write(f"\n\nmissing_responses={missing}\n")
    log.close()
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nmissing_responses={missing}")


if __name__ == "__main__":
    main()

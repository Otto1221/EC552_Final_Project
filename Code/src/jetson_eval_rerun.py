#!/usr/bin/env python3
"""Reruns only the failed (timed-out) prompts from a prior jetson eval JSON.

Uses a longer timeout so slow expert prompts can complete. Merges new successes
back into the original results, recomputes FINAL average, rewrites JSON + log.

Usage:
  python3 jetson_eval_rerun.py [tag]     # tag defaults to 'qwen35'
"""
import json, sys, time, urllib.request
from pathlib import Path

import importlib.util
spec = importlib.util.spec_from_file_location("eval100", "./eval100.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen35"
OUT_JSON = Path(f"jetson_eval100_{TAG}.json")
OUT_LOG = Path(f"jetson_eval100_{TAG}.log")
URL = "http://localhost:8080/v1/chat/completions"
TIMEOUT = 1800


def call(prompt):
    body = json.dumps({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": e.SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 1800,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0


def main():
    results = json.load(open(OUT_JSON))
    log = open(OUT_LOG, "a")
    log.write(f"\n--- RERUN with timeout={TIMEOUT}s ---\n\n")
    log.flush()

    failed = [i for i, r in enumerate(results) if r["response"].startswith("<ERROR:")]
    print(f"rerunning {len(failed)} failed prompts out of {len(results)}")

    for idx in failed:
        r = results[idx]
        prompt = r["prompt"]
        diff = r["difficulty"]
        keywords = r["keywords"]
        i = idx + 1
        try:
            payload, dur = call(prompt)
            response = payload["choices"][0]["message"]["content"]
            finish = payload["choices"][0].get("finish_reason", "?")
            usage = payload.get("usage", {})
        except Exception as ex:
            response = f"<ERROR: {ex}>"; finish = "error"; usage = {}; dur = 0

        scores = e.score_output(prompt, response, keywords)
        total = scores.get("total", 0)
        log.write(f"[{i}/{len(results)}] ({diff}) {prompt[:70]}...\n  {dur:.1f}s | {total}/100 | finish={finish} | chars={len(response)}\n\n")
        log.flush()
        results[idx] = {
            "difficulty": diff, "prompt": prompt, "keywords": keywords,
            "response": response, "scores": scores, "time": dur, "usage": usage,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    total_score = sum(r["scores"].get("total", 0) for r in results)
    avg = total_score / len(results)
    log.write(f"\nFINAL average (after rerun): {avg:.1f}/100\n")
    log.close()
    print(f"Done. avg={avg:.1f}/100. results={OUT_JSON}")


if __name__ == "__main__":
    main()

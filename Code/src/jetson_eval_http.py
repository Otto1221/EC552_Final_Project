#!/usr/bin/env python3
"""Jetson-side eval: hits llama-server OpenAI-compatible API at localhost:8080.

Uses the same TEST_PROMPTS + SYSTEM_MSG from eval100.py.
Writes jetson_eval100_<tag>.json + .log in the working directory.

Usage:
  python3 jetson_eval_http.py [tag]     # tag defaults to 'qwen35'
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path

import importlib.util
spec = importlib.util.spec_from_file_location("eval100", "./eval100.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen35"
OUT_JSON = Path(f"jetson_eval100_{TAG}.json")
OUT_LOG = Path(f"jetson_eval100_{TAG}.log")
URL = "http://localhost:8080/v1/chat/completions"


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
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0


def main():
    results = []
    log = open(OUT_LOG, "w")
    log.write(f"START jetson_eval100_{TAG}  ({len(e.TEST_PROMPTS)} prompts)\n\n")
    log.flush()
    total_score = 0
    for i, (diff, prompt, keywords) in enumerate(e.TEST_PROMPTS, 1):
        try:
            payload, dur = call(prompt)
            response = payload["choices"][0]["message"]["content"]
            finish = payload["choices"][0].get("finish_reason", "?")
            usage = payload.get("usage", {})
        except Exception as ex:
            response = f"<ERROR: {ex}>"; finish = "error"; usage = {}; dur = 0

        scores = e.score_output(prompt, response, keywords)
        total = scores.get("total", 0)
        total_score += total
        log.write(f"[{i}/{len(e.TEST_PROMPTS)}] ({diff}) {prompt[:70]}...\n  {dur:.1f}s | {total}/100 | finish={finish} | chars={len(response)}\n\n")
        log.flush()
        results.append({
            "difficulty": diff, "prompt": prompt, "keywords": keywords,
            "response": response, "scores": scores, "time": dur, "usage": usage,
        })
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    avg = total_score / len(e.TEST_PROMPTS)
    log.write(f"\nFINAL average: {avg:.1f}/100\n")
    log.close()
    print(f"Done. avg={avg:.1f}/100. results={OUT_JSON}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the demo_prompts.json set against a running llama-server on Jetson.

Usage (from Mac, with SSH tunnel forwarding Jetson :8080 to localhost:18080):
    python3 run_demo_prompts.py --url http://localhost:18080 --out demo_results.json

Or directly on Jetson:
    python3 run_demo_prompts.py --url http://localhost:8080 --out demo_results.json
"""
import json, sys, time, argparse, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chen_truong_system_prompt import CHEN_TRUONG_SYSTEM_MSG as SYSTEM_MSG

def call(url, user_prompt, max_tok=2800, timeout=1800):
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": max_tok,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    ch = out["choices"][0]
    return ch["message"]["content"], ch.get("finish_reason"), dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--prompts", default=str(HERE.parent / "data" / "demo_prompts.json"))
    ap.add_argument("--out", default=str(HERE.parent / "results" / "demo_results.json"))
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    results = []
    for p in prompts:
        print(f"[{p['id']}] {p['prompt'][:70]}...")
        try:
            content, finish, dt = call(args.url, p["prompt"])
            results.append({"id": p["id"], "entry": p, "response": content, "finish": finish, "time": dt})
            print(f"  ok  finish={finish}  {dt:.1f}s  {len(content)} chars")
        except Exception as exc:
            results.append({"id": p["id"], "entry": p, "error": str(exc)})
            print(f"  ERR: {exc}")
        json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()

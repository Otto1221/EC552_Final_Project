#!/usr/bin/env python3
"""HTTP runner for sbol_eval_v2 benchmark against a local llama-server.

Invokes the 100-prompt stratified SBOL benchmark defined in sbol_eval_v2.py,
calling an OpenAI-compatible /v1/chat/completions endpoint with a long per-
request timeout. Writes incremental results so a crash or ctrl-C does not
lose completed evaluations. Prints a stratified summary at the end.

Usage:
  python3 jetson_sbol_eval_v2_http.py [tag]     # tag defaults to 'run1'
"""
import json, os, sys, time, urllib.request
from pathlib import Path

import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sbol_eval_v2", str(HERE / "sbol_eval_v2.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

TAG = sys.argv[1] if len(sys.argv) > 1 else "run1"
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / f"sbol_eval_v2_{TAG}.json"
OUT_LOG = RESULTS / f"sbol_eval_v2_{TAG}.log"
URL = os.environ.get("LLAMA_URL", "http://localhost:8080/v1/chat/completions")
MODEL = os.environ.get("LLAMA_MODEL", "qwen")
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "1800"))
MAX_TOK = int(os.environ.get("MAX_TOK", "1800"))  # 0 or negative = no cap (uses ctx)
REP_PENALTY = float(os.environ.get("REP_PENALTY", "1.0"))  # 1.05+ curbs pathological loops
SCHEMA_RETRY = int(os.environ.get("SCHEMA_RETRY", "0"))  # 1 = retry once on invalid JSON
CHEN_PROMPT = int(os.environ.get("CHEN_PROMPT", "0"))  # 1 = use Chen & Truong system prompt
if CHEN_PROMPT:
    _cps = importlib.util.spec_from_file_location(
        "chen_truong_system_prompt", str(HERE / "chen_truong_system_prompt.py")
    )
    _cpm = importlib.util.module_from_spec(_cps); _cps.loader.exec_module(_cpm)
    e.SYSTEM_MSG = _cpm.CHEN_TRUONG_SYSTEM_MSG


def _post(messages):
    payload_dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if REP_PENALTY and REP_PENALTY != 1.0:
        payload_dict["repetition_penalty"] = REP_PENALTY
    if MAX_TOK > 0:
        payload_dict["max_tokens"] = MAX_TOK
    else:
        backend = os.environ.get("BACKEND", "llama").lower()
        payload_dict["max_tokens"] = -1 if backend == "llama" else 16384
    body = json.dumps(payload_dict).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0


def call(prompt):
    messages = [
        {"role": "system", "content": e.SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    payload, dur = _post(messages)
    response = payload["choices"][0]["message"]["content"]
    # Optional single retry with validator feedback if JSON can't be extracted
    if SCHEMA_RETRY and e.extract_json(response) is None:
        retry_messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": (
                "Your previous response did not contain a valid, parseable JSON object. "
                "Return ONLY a single JSON object with keys name/components/interactions/"
                "behavior/organism. No prose, no markdown fences."
            )},
        ]
        try:
            payload2, dur2 = _post(retry_messages)
            response2 = payload2["choices"][0]["message"]["content"]
            if e.extract_json(response2) is not None:
                # Use the successful retry; accumulate duration
                return payload2, dur + dur2
        except Exception:
            pass
    return payload, dur


def main():
    results = []
    if OUT_JSON.exists():
        try:
            results = json.load(open(OUT_JSON))
            print(f"resuming from {OUT_JSON} with {len(results)} existing results")
        except Exception:
            results = []

    done_prompts = {r["entry"]["prompt"] for r in results}
    log = open(OUT_LOG, "a")
    log.write(f"\n=== sbol_eval_v2 run tag={TAG} timeout={TIMEOUT}s model={MODEL} ===\n\n")
    log.flush()

    stride = int(os.environ.get("SAMPLE_EVERY", "1"))
    offset = int(os.environ.get("SAMPLE_OFFSET", "0"))
    prompts = e.PROMPTS[offset::stride] if stride > 1 else e.PROMPTS
    n = len(prompts)
    if stride > 1:
        log.write(f"SAMPLE_EVERY={stride} SAMPLE_OFFSET={offset} -> {n} prompts\n")
        log.flush()
    for i, entry in enumerate(prompts, start=1):
        if entry["prompt"] in done_prompts:
            continue
        try:
            payload, dur = call(entry["prompt"])
            response = payload["choices"][0]["message"]["content"]
            finish = payload["choices"][0].get("finish_reason", "?")
            usage = payload.get("usage", {})
        except Exception as ex:
            response = f"<ERROR: {ex}>"; finish = "error"; usage = {}; dur = 0

        score = e.score_axes(entry, response)
        total = score["total"]
        diff = entry.get("diff"); org = entry.get("org"); topo = entry.get("topo")
        log.write(
            f"[{i}/{n}] d{diff} {org}/{topo}: {entry['prompt'][:60]}\n"
            f"  {dur:.1f}s | {total}/100 | finish={finish} | chars={len(response)}\n"
        )
        axes_line = " ".join(
            f"{a}={min(sum(score['axes'][a].values()), e.AXIS_MAX[a])}" for a in e.AXIS_MAX
        )
        log.write(f"  {axes_line}\n\n")
        log.flush()

        results.append({
            "entry": entry,
            "response": response,
            "score": score,
            "finish": finish,
            "usage": usage,
            "time": dur,
        })
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    summary = e.summarize(results)
    log.write("\n=== SUMMARY ===\n")
    log.write(json.dumps(summary, indent=2))
    log.write("\n")
    log.close()
    with open(OUT_JSON.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Live-streaming demo — stream tokens, self-correct on validation errors, score, emit SBOL3.

Usage:
    python3 src/demo_stream.py                      # interactive — type your prompt
    python3 src/demo_stream.py 0                    # demo_01 by index
    python3 src/demo_stream.py "custom prompt text"
"""
import json, os, re, sys, time, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chen_truong_system_prompt import CHEN_TRUONG_SYSTEM_MSG

URL = "http://localhost:8080/v1/chat/completions"
MODEL_PATH = os.getenv("NG_MODEL_PATH", "qwen35-27b-lora")
PROMPTS = json.load(open(HERE.parent / "data" / "demo_prompts.json"))
MAX_RETRIES = 2   # up to MAX_RETRIES + 1 total attempts

# ---------- prompt intake ----------
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg.isdigit():
        entry = PROMPTS[int(arg)]
        prompt = entry["prompt"]
        label = entry["id"]
    else:
        prompt = arg
        label = "custom"
else:
    BAR = "=" * 78
    print(f"\n\033[1;36m{BAR}\033[0m")
    print(f"\033[1;36m  DESCRIBE YOUR CIRCUIT  (plain English is fine)\033[0m")
    print(f"\033[0;37m  paste or type prompt — press Enter on an empty line to submit\033[0m")
    print(f"\033[1;36m{BAR}\033[0m\n")
    lines = []
    leader = "  \033[1;33m> \033[0m"
    while True:
        try:
            line = input(leader)
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
        leader = "  \033[1;33m. \033[0m"
    prompt = " ".join(l.strip() for l in lines).strip()
    if not prompt:
        sys.exit("no prompt given")
    label = "custom"


# ---------- helpers ----------
def stream_chat(messages):
    body = json.dumps({
        "model": MODEL_PATH,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 8000,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                    tok = delta.get("content") or delta.get("reasoning") or ""
                except Exception:
                    continue
                if not tok:
                    continue
                sys.stdout.write(tok)
                sys.stdout.flush()
                chunks.append(tok)
    except urllib.error.URLError as e:
        sys.exit(f"\n\033[1;31m✗ MLX server unreachable at {URL}\033[0m\n"
                 f"  reason: {e.reason}\n"
                 f"  start with: mlx_lm.server --model {MODEL_PATH} --port 8080")
    dt = time.time() - t0
    full = "".join(chunks)
    ntok = len(chunks)
    nchars = len(full)
    rate = ntok / dt if dt > 0 else 0.0
    print(f"\n\n\033[1;36m=== {dt:.1f}s · ~{ntok} tokens · {nchars} chars ({rate:.1f} tok/s) ===\033[0m")
    return full


def check_output(full):
    """Return (parsed_obj, list_of_error_strings). Empty list = clean."""
    from render_sbol_circuit import extract_json
    obj = extract_json(full)
    errs = []
    if obj is None:
        return None, ["output is not valid JSON (parse failed)"]
    if not isinstance(obj.get("components"), list) or not obj.get("components"):
        errs.append("missing or empty 'components' list")
    if not isinstance(obj.get("interactions"), list):
        errs.append("missing 'interactions' list")
    comp_names = {c.get("name") for c in obj.get("components", []) if isinstance(c, dict)}
    for i, ix in enumerate(obj.get("interactions", []) or []):
        if not isinstance(ix, dict):
            continue
        if not ix.get("type"):
            errs.append(f"interaction[{i}] missing 'type' field")
        for k in ("from", "to"):
            v = ix.get(k)
            if not v:
                errs.append(f"interaction[{i}] missing '{k}' field")
            elif v not in comp_names:
                errs.append(f"interaction[{i}] references unknown component: '{v}'")
    if errs:
        return obj, errs
    try:
        from json_to_sbol3 import circuit_to_sbol3
        import sbol3
        xml = circuit_to_sbol3(obj, circuit_name=obj.get("name", "circuit"))
        tmp = Path("/tmp/validate_demo.xml")
        tmp.write_text(xml)
        doc = sbol3.Document()
        doc.read(str(tmp), sbol3.RDF_XML)
        report = doc.validate()
        # Only treat real errors as retry-triggering — warnings are informational
        report_errs = getattr(report, 'errors', None)
        if report_errs is None:
            report_errs = list(report)  # older pysbol3 API: no .errors attr
        for e in report_errs:
            errs.append(str(e))
    except ImportError:
        pass
    except Exception as e:
        errs.append(f"SBOL3 conversion error: {e}")
    return obj, errs


# ---------- startup banner (provenance — preempts "is this ChatGPT?") ----------
print(f"\n\033[1;36m=== PIPELINE CONFIG ===\033[0m")
print(f"  \033[1;33mModel:\033[0m    Qwen3.5-27B + LoRA (8-bit)")
print(f"  \033[1;33mServer:\033[0m   {URL}  \033[0;37m(MLX, local, offline)\033[0m")
print(f"  \033[1;33mSystem:\033[0m   Chen-Truong prompt ({len(CHEN_TRUONG_SYSTEM_MSG):,} chars)")
print(f"  \033[1;33mRubric:\033[0m   deterministic Python, 6 axes, 100 pts")
print(f"  \033[1;33mValidator:\033[0m pysbol3 reference implementation")

# ---------- header ----------
print(f"\n\033[1;36m=== {label} ===\033[0m")
print(f"\033[1;33mPrompt:\033[0m {prompt}\n")

# ---------- self-correction loop ----------
messages = [
    {"role": "system", "content": CHEN_TRUONG_SYSTEM_MSG},
    {"role": "user", "content": prompt},
]

full = ""
final_obj = None
final_errs = []
for attempt in range(MAX_RETRIES + 1):
    if attempt == 0:
        print(f"\033[1;32mModel output:\033[0m\n", flush=True)
    else:
        print(f"\n\033[1;35m↻ RETRY {attempt}/{MAX_RETRIES}\033[0m  — asking model to fix the issues above\n", flush=True)

    full = stream_chat(messages)
    final_obj, final_errs = check_output(full)

    if not final_errs:
        if attempt > 0:
            print(f"\n\033[1;32m✓ validation passed on attempt {attempt + 1}\033[0m")
        break

    print(f"\n\033[1;33m⚠ {len(final_errs)} validation issue(s) found:\033[0m")
    for e in final_errs[:5]:
        print(f"    • {e}")
    if len(final_errs) > 5:
        print(f"    ... and {len(final_errs) - 5} more")

    if attempt < MAX_RETRIES:
        messages.append({"role": "assistant", "content": full})
        correction = (
            "The JSON you just produced has these validation issues:\n"
            + "\n".join(f"- {e}" for e in final_errs[:10])
            + "\n\nRegenerate the ENTIRE corrected JSON object. Output only the JSON — no explanation, no markdown fences."
        )
        messages.append({"role": "user", "content": correction})
    else:
        print(f"\n\033[1;31m✗ still has {len(final_errs)} issues after {MAX_RETRIES} retries — proceeding with last output\033[0m")

# ---------- save ----------
out_dir = Path.home() / "Desktop"
out_dir.mkdir(exist_ok=True)
(out_dir / "demo_last.txt").write_text(full)
print(f"\nsaved → ~/Desktop/demo_last.txt")

if final_obj is None:
    print("\n\033[1;31m⚠ final output is not parseable JSON — cannot score or emit SBOL3\033[0m")
    sys.exit(1)

# ---------- rubric score ----------
try:
    from sbol_eval_v2 import score_axes, AXIS_MAX
    org_raw = (final_obj.get("organism") or "").lower()
    # Word-bounded matches to avoid false positives like "borrelia" matching "coli"
    def _has(*tokens):
        return any(re.search(rf"\b{re.escape(t)}\b", org_raw) for t in tokens)
    if _has("e. coli", "ecoli", "escherichia coli", "coli"):
        org = "ecoli"
    elif _has("yeast", "cerevisiae", "saccharomyces"):
        org = "yeast"
    elif _has("hek", "hek293", "mammalian", "cho", "human"):
        org = "mammalian"
    elif _has("subtilis", "bacillus"):
        org = "bacillus"  # rubric expects 'bacillus' (not 'bsubtilis')
    elif _has("arabidopsis", "tobacco", "plant", "thaliana"):
        org = "plant"
    elif _has("cell-free", "cell free", "cellfree", "in vitro"):
        org = "cellfree"
    else:
        org = "ecoli"  # conservative default for the rubric
    # Use the canonical entry's metadata when running an indexed prompt; fall
    # back to env-var-overridable defaults for free-text custom prompts.
    if 'entry' in locals() and isinstance(entry, dict) and entry.get("diff") is not None:
        diff = entry["diff"]
        topo = entry.get("topo", "reporter")
        org = entry.get("org", org)
        kw = entry.get("kw", [])
    else:
        try:
            diff = int(__import__("os").environ.get("DEMO_DIFF", "3"))
        except ValueError:
            diff = 3
        diff = max(1, min(5, diff))  # clamp to rubric's range
        topo = __import__("os").environ.get("DEMO_TOPO", "reporter")
        kw = []
    entry = {"id": "live", "diff": diff, "org": org, "topo": topo,
             "prompt": prompt, "kw": kw, "must_have": []}
    score = score_axes(entry, full)
    print(f"\n\033[1;36m=== RUBRIC SCORE (deterministic Python) ===\033[0m")
    print(f"\033[1;32mTOTAL: {score['total']}/100\033[0m")
    for axis, breakdown in score["axes"].items():
        pts = min(sum(breakdown.values()), AXIS_MAX[axis])
        bar = "█" * pts + "·" * (AXIS_MAX[axis] - pts)
        print(f"  {axis:3s}  {pts:2d}/{AXIS_MAX[axis]:2d}  \033[0;32m{bar}\033[0m")
except Exception as exc:
    print(f"\n\033[1;31m⚠ score failed: {exc}\033[0m")

# ---------- SBOL3 RDF/XML ----------
xml_path = None
try:
    from json_to_sbol3 import circuit_to_sbol3, SBO_INTERACTION_TYPES, SO_ROLES
    xml = circuit_to_sbol3(final_obj, circuit_name=final_obj.get("name", "live_circuit"))
    xml_path = out_dir / "demo_last.xml"
    xml_path.write_text(xml)
    print(f"\n\033[1;36m=== SBOL3 RDF/XML  ({len(xml)} bytes → ~/Desktop/demo_last.xml) ===\033[0m")
    for line in xml.splitlines():
        print(f"\033[0;37m{line}\033[0m")

    # ---------- ontology mappings used in THIS circuit (Python lookup) ----------
    ix_types_used = {ix.get("type") for ix in final_obj.get("interactions", []) if isinstance(ix, dict)}
    comp_types_used = {c.get("type") for c in final_obj.get("components", []) if isinstance(c, dict)}
    print(f"\n\033[1;36m=== ONTOLOGY MAPPINGS  (deterministic Python lookup) ===\033[0m")
    print(f"  \033[1;33mSBO (interactions):\033[0m")
    for t in sorted(x for x in ix_types_used if x in SBO_INTERACTION_TYPES):
        print(f"    {t:18s} → \033[0;32m{SBO_INTERACTION_TYPES[t]}\033[0m")
    print(f"  \033[1;33mSO  (components):\033[0m")
    from json_to_sbol3 import SO_ROLES as _SO
    for t in sorted(x for x in comp_types_used if x in _SO):
        print(f"    {t:18s} → \033[0;32m{_SO[t]}\033[0m")
except Exception as exc:
    print(f"\n\033[1;31m⚠ SBOL3 conversion failed: {exc}\033[0m")

# ---------- pysbol3 validation ----------
if xml_path is not None:
    try:
        import sbol3
        doc = sbol3.Document()
        doc.read(str(xml_path), sbol3.RDF_XML)
        report = doc.validate()
        err_list = list(getattr(report, 'errors', []) or [])
        warn_list = list(getattr(report, 'warnings', []) or [])
        if not err_list and not warn_list:
            # Older pysbol3 API: no .errors/.warnings split — treat all as errors
            all_issues = list(report)
            if all_issues:
                err_list = all_issues
        status = ("\033[1;32m✓ PASSED\033[0m" if not err_list
                  else f"\033[1;31m✗ {len(err_list)} errors\033[0m")
        print(f"\n\033[1;36m=== SBOL3 VALIDATION (pysbol3 reference implementation) ===\033[0m")
        print(f"{status}   {len(doc.objects)} objects parsed, {len(err_list)} errors, {len(warn_list)} warnings")
        type_counts = {}
        for obj in doc.objects:
            t = type(obj).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, n in sorted(type_counts.items()):
            print(f"    {n}× \033[0;32m{t}\033[0m")
        for e in err_list[:5]:
            print(f"    \033[0;31mERR: \033[0m {e}")
        for w in warn_list[:5]:
            print(f"    \033[0;33mWARN:\033[0m {w}")
    except ImportError:
        print("\n\033[0;33m(pysbol3 not installed — run: python3 -m pip install --user --break-system-packages sbol3)\033[0m")
    except Exception as exc:
        print(f"\n\033[1;31m⚠ SBOL3 validation failed: {exc}\033[0m")


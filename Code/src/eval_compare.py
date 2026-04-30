#!/usr/bin/env python3
"""
Before/after fine-tuning comparison for Newgenes.
Usage:
  python eval_compare.py before    # run base model, save results
  python eval_compare.py after     # run fine-tuned model, save results
  python eval_compare.py compare   # compare both side by side
"""
import json, sys, os, time, re
from pathlib import Path

MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
ADAPTER = "./adapters/gemma4-newgenes"
RESULTS_DIR = Path("./eval_results")
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_MSG = (
    "You are a synthetic biology assistant that converts natural language "
    "descriptions of genetic circuits into structured JSON. Return a single "
    "JSON object with these fields:\n\n"
    "- \"name\": short circuit name\n"
    "- \"components\": array of {\"name\", \"type\", \"description\"} — each part in the circuit\n"
    "- \"interactions\": array of {\"from\", \"to\", \"type\"} — wiring between parts\n"
    "- \"behavior\": what the circuit does\n"
    "- \"organism\": host organism\n\n"
    "Component types: promoter, rbs, cds, terminator, operator, other.\n"
    "Interaction types: transcription (promoter→cds), translation (rbs→cds), "
    "activation (cds→promoter/operator), repression (cds→promoter/operator).\n\n"
    "Rules:\n"
    "- Every gene needs: promoter, rbs, cds, terminator\n"
    "- Use snake_case names\n"
    "- Every from/to must reference an existing component name\n"
    "- Respond with valid JSON only, no explanation"
)

# 10 test prompts — none from training data, spanning easy to hard
TEST_PROMPTS = [
    # 1. Simple (single gene expression)
    "Express GFP from a constitutive promoter in E. coli",

    # 2. Simple inducible
    "IPTG-inducible mCherry reporter in E. coli",

    # 3. Two-gene regulation
    "Design a circuit where arabinose induces expression of a T7 RNA polymerase, which then drives GFP from a T7 promoter",

    # 4. Repression logic (NOT gate)
    "Build a NOT gate: constitutive TetR represses GFP, so adding aTc turns GFP ON",

    # 5. Toggle switch
    "Create a toggle switch using LacI and TetR that can flip between two stable states",

    # 6. Biosensor
    "Design an arsenic biosensor where ArsR detects arsenite and derepresses a GFP reporter, include a constitutive mCherry normalization control",

    # 7. Metabolic pathway
    "Engineer a violacein biosynthesis pathway with VioA, VioB, VioC, VioD, and VioE enzymes driven from a T7 promoter as an operon",

    # 8. Quorum sensing
    "Build a quorum-sensing relay: LuxI produces AHL, which activates LuxR to turn on a downstream GFP reporter in a second cell population",

    # 9. Complex therapeutic
    "Design a tumor-targeting circuit in E. coli Nissle: hypoxia-responsive FNR promoter drives invasion gene InvA, while a second constitutive module produces anti-HER2 nanobody for surface display via INP anchor",

    # 10. Multi-input logic
    "Create an AND gate where both IPTG and arabinose are required for GFP output. Use LacI repression and AraC activation as the two inputs converging on a hybrid promoter",
]


ALLOWED_TYPES = {'promoter', 'rbs', 'cds', 'terminator', 'operator', 'other'}

TIER_LABELS = {
    'valid_json':       ('T1 Format',   2),
    'correct_keys':     ('T1 Format',   2),
    'has_components':   ('T1 Format',   2),
    'has_interactions': ('T1 Format',   2),
    'snake_case':       ('T2 Schema',   2),
    'no_orphans':       ('T2 Schema',   2),
    'no_dup_ix':        ('T2 Schema',   1),
    'all_gene_parts':   ('T2 Schema',   2),
    'valid_types':      ('T2 Schema',   1),
    'has_descriptions': ('T2 Schema',   2),
    'tx_from_promoter': ('T3 Biology',  3),
    'tl_from_rbs':      ('T3 Biology',  3),
    'cds_wiring':       ('T3 Biology',  3),
    'behavior_quality': ('T3 Biology',  2),
    'no_self_loops':    ('T3 Biology',  1),
}

def _pct_score(passing, total, max_pts):
    """Partial credit: full if 100%, half if >75%, else 0."""
    if total == 0:
        return max_pts
    ratio = passing / total
    if ratio >= 1.0:
        return max_pts
    elif ratio >= 0.75:
        return max_pts // 2 or 1
    return 0

def _extract_json(raw_text):
    """Extract JSON object from model output, handling thinking blocks and fences."""
    text = raw_text.strip()
    text = re.sub(r'<\|channel\>thought.*?<channel\|>', '', text, flags=re.DOTALL)
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        text = text.strip()
    match = re.search(r'\{[^{}]*"name".*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def score_output(prompt, raw_text):
    """Score model output on 15 criteria across 3 tiers (max 30 pts).

    Tier 1 — Format (8 pts): Can it output valid structured JSON?
    Tier 2 — Schema (10 pts): Does it follow the naming/structural rules?
    Tier 3 — Biology (12 pts): Does it understand genetic circuit wiring?
    """
    scores = {}
    zero = {k: 0 for k in TIER_LABELS}
    zero['total'] = 0

    # --- TIER 1: FORMAT (8 pts) ---

    # valid_json (2): parses as JSON
    try:
        r = _extract_json(raw_text)
        scores['valid_json'] = 2
    except Exception:
        return zero

    # correct_keys (2): exact {name, components, interactions, behavior, organism}
    expected = {'name', 'components', 'interactions', 'behavior', 'organism'}
    present = set(r.keys()) & expected
    if present == expected:
        scores['correct_keys'] = 2
    elif len(present) >= 4:
        scores['correct_keys'] = 1
    else:
        scores['correct_keys'] = 0

    # has_components (2): non-empty array of dicts
    comps = r.get('components', [])
    if not isinstance(comps, list):
        comps = []
    comps = [c for c in comps if isinstance(c, dict)]
    if len(comps) >= 3:
        scores['has_components'] = 2
    elif len(comps) >= 1:
        scores['has_components'] = 1
    else:
        scores['has_components'] = 0

    # has_interactions (2): non-empty array of dicts
    ixs = r.get('interactions', [])
    if not isinstance(ixs, list):
        ixs = []
    ixs = [ix for ix in ixs if isinstance(ix, dict)]
    if len(ixs) >= 3:
        scores['has_interactions'] = 2
    elif len(ixs) >= 1:
        scores['has_interactions'] = 1
    else:
        scores['has_interactions'] = 0

    if not comps or not ixs:
        for k in TIER_LABELS:
            scores.setdefault(k, 0)
        scores['total'] = sum(scores.values())
        return scores

    # --- TIER 2: SCHEMA COMPLIANCE (10 pts) ---

    names = {c['name'] for c in comps}
    ct = {c['name']: c.get('type', '') for c in comps}

    # snake_case (2): all names match ^[a-z0-9][a-z0-9_]*$
    n_snake = sum(1 for c in comps if re.match(r'^[a-z0-9][a-z0-9_]*$', c.get('name', 'X')))
    scores['snake_case'] = _pct_score(n_snake, len(comps), 2)

    # no_orphans (2): every from/to references an existing component
    n_refs = 0
    n_valid = 0
    for ix in ixs:
        for side in ('from', 'to'):
            n_refs += 1
            if ix.get(side, '') in names:
                n_valid += 1
    scores['no_orphans'] = _pct_score(n_valid, n_refs, 2)

    # no_dup_ix (1): no duplicate (from, to, type) triples
    seen = set()
    dups = 0
    for ix in ixs:
        k = (ix.get('from'), ix.get('to'), ix.get('type'))
        if k in seen:
            dups += 1
        seen.add(k)
    scores['no_dup_ix'] = 1 if dups == 0 else 0

    # all_gene_parts (2): at least 1 promoter, rbs, cds, terminator
    types_present = {c.get('type') for c in comps}
    required = {'promoter', 'rbs', 'cds', 'terminator'}
    n_have = len(required & types_present)
    if n_have == 4:
        scores['all_gene_parts'] = 2
    elif n_have >= 3:
        scores['all_gene_parts'] = 1
    else:
        scores['all_gene_parts'] = 0

    # valid_types (1): all component types in allowed set
    all_valid_types = all(c.get('type', '') in ALLOWED_TYPES for c in comps)
    scores['valid_types'] = 1 if all_valid_types else 0

    # has_descriptions (2): non-terminator components have descriptions >= 10 chars
    desc_comps = [c for c in comps if c.get('type') != 'terminator']
    n_good_desc = sum(1 for c in desc_comps if len(c.get('description', '').strip()) >= 10)
    scores['has_descriptions'] = _pct_score(n_good_desc, len(desc_comps), 2) if desc_comps else 2

    # --- TIER 3: BIOLOGICAL CORRECTNESS (12 pts) ---

    # tx_from_promoter (3): transcription edges only from promoters
    tx_edges = [ix for ix in ixs if ix.get('type') == 'transcription']
    n_tx_ok = sum(1 for ix in tx_edges if ct.get(ix['from'], '') == 'promoter')
    scores['tx_from_promoter'] = _pct_score(n_tx_ok, len(tx_edges), 3)

    # tl_from_rbs (3): translation edges from rbs/other, targeting cds
    tl_edges = [ix for ix in ixs if ix.get('type') == 'translation']
    n_tl_ok = sum(1 for ix in tl_edges
                  if ct.get(ix['from'], '') in ('rbs', 'other')
                  and ct.get(ix['to'], '') == 'cds')
    scores['tl_from_rbs'] = _pct_score(n_tl_ok, len(tl_edges), 3)

    # cds_wiring (3): every CDS has both a transcription input AND a translation input
    cds_names = {c['name'] for c in comps if c.get('type') == 'cds'}
    tx_targets = {ix['to'] for ix in ixs if ix.get('type') == 'transcription'}
    tl_targets = {ix['to'] for ix in ixs if ix.get('type') == 'translation'}
    if cds_names:
        n_fully_wired = sum(1 for cds in cds_names
                            if cds in tx_targets and cds in tl_targets)
        scores['cds_wiring'] = _pct_score(n_fully_wired, len(cds_names), 3)
    else:
        scores['cds_wiring'] = 0

    # behavior_quality (2): substantive behavior string
    beh = r.get('behavior', '').strip()
    if len(beh) >= 60:
        scores['behavior_quality'] = 2
    elif len(beh) >= 25:
        scores['behavior_quality'] = 1
    else:
        scores['behavior_quality'] = 0

    # no_self_loops (1): no interaction where from == to
    has_self = any(ix.get('from') == ix.get('to') for ix in ixs)
    scores['no_self_loops'] = 0 if has_self else 1

    scores['total'] = sum(scores.values())
    return scores


def run_inference(mode):
    """Run model inference on test prompts."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"\nLoading model ({mode})...")
    t0 = time.time()
    if mode == 'after':
        model, tokenizer = load(MODEL, adapter_path=ADAPTER)
    else:
        model, tokenizer = load(MODEL)
    print(f"Loaded in {time.time()-t0:.1f}s")

    results = []
    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{idx+1}/{len(TEST_PROMPTS)}] {prompt[:70]}...")

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t1 = time.time()
        sampler = make_sampler(temp=0.1, top_p=0.95)
        max_tok = 4096 if mode == 'before' else 2048
        response = generate(
            model, tokenizer, prompt=chat_prompt,
            max_tokens=max_tok, sampler=sampler,
        )
        elapsed = time.time() - t1

        # Score
        scores = score_output(prompt, response)
        results.append({
            'prompt': prompt,
            'response': response,
            'scores': scores,
            'time': round(elapsed, 1),
        })
        print(f"  {elapsed:.1f}s | score={scores['total']}/30 | json={'OK' if scores['valid_json'] else 'FAIL'}")

    # Save
    out_path = RESULTS_DIR / f"{mode}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Summary by tier
    n = len(results)
    print(f"\n{'='*60}")
    print(f"{mode.upper()} SUMMARY  ({n} prompts)")
    print(f"{'='*60}")

    for tier_name in ('T1 Format', 'T2 Schema', 'T3 Biology'):
        tier_criteria = [(k, mx) for k, (t, mx) in TIER_LABELS.items() if t == tier_name]
        tier_max = sum(mx for _, mx in tier_criteria)
        tier_sum = sum(sum(r['scores'].get(k, 0) for r in results) for k, _ in tier_criteria)
        print(f"\n  {tier_name} (max {tier_max} pts):")
        for k, mx in tier_criteria:
            s = sum(r['scores'].get(k, 0) for r in results)
            print(f"    {k:<20} {s:>4}/{mx*n}")
        print(f"    {'SUBTOTAL':<20} {tier_sum:>4}/{tier_max*n}")

    total_scores = [r['scores']['total'] for r in results]
    print(f"\n  Avg total: {sum(total_scores)/n:.1f}/30")
    print(f"  Avg time:  {sum(r['time'] for r in results)/n:.1f}s")


def compare():
    """Compare before and after results."""
    before_path = RESULTS_DIR / "before.json"
    after_path = RESULTS_DIR / "after.json"

    if not before_path.exists():
        print("No before.json — run: python eval_compare.py before")
        return
    if not after_path.exists():
        print("No after.json — run: python eval_compare.py after")
        return

    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    n = len(before)

    print(f"\n{'='*72}")
    print("BEFORE vs AFTER COMPARISON")
    print(f"{'='*72}")

    for tier_name in ('T1 Format', 'T2 Schema', 'T3 Biology'):
        tier_criteria = [(k, mx) for k, (t, mx) in TIER_LABELS.items() if t == tier_name]
        tier_max = sum(mx for _, mx in tier_criteria)
        print(f"\n  {tier_name} (max {tier_max} pts per prompt)")
        print(f"  {'Criterion':<20} {'Max':>4} {'Before':>8} {'After':>8} {'Delta':>8}")
        print(f"  {'-'*52}")
        t_b = t_a = 0
        for k, mx in tier_criteria:
            b = sum(r['scores'].get(k, 0) for r in before)
            a = sum(r['scores'].get(k, 0) for r in after)
            d = a - b
            t_b += b; t_a += a
            sign = '+' if d >= 0 else ''
            print(f"  {k:<20} {mx*n:>4} {b:>8} {a:>8} {sign}{d:>7}")
        print(f"  {'-'*52}")
        d = t_a - t_b
        sign = '+' if d >= 0 else ''
        print(f"  {'SUBTOTAL':<20} {tier_max*n:>4} {t_b:>8} {t_a:>8} {sign}{d:>7}")

    b_avg = sum(r['scores']['total'] for r in before) / n
    a_avg = sum(r['scores']['total'] for r in after) / n
    d = a_avg - b_avg
    sign = '+' if d >= 0 else ''
    print(f"\n  {'AVG TOTAL':<20} {'/30':>4} {b_avg:>7.1f} {a_avg:>8.1f} {sign}{d:>7.1f}")

    b_time = sum(r['time'] for r in before) / n
    a_time = sum(r['time'] for r in after) / n
    print(f"  {'AVG TIME':<20} {'s':>4} {b_time:>7.1f} {a_time:>8.1f} {a_time-b_time:>+8.1f}")

    # Per-prompt comparison
    print(f"\n{'='*72}")
    print("PER-PROMPT SCORES")
    print(f"{'='*72}")
    print(f"\n{'#':<3} {'Prompt':<42} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 72)
    for i in range(n):
        p = before[i]['prompt'][:40]
        bs = before[i]['scores']['total']
        a_s = after[i]['scores']['total']
        d = a_s - bs
        marker = ' +' if d > 0 else (' -' if d < 0 else '  ')
        print(f"{i+1:<3} {p:<42} {bs:>5}/30 {a_s:>6}/30 {'+' if d>=0 else ''}{d:>5}{marker}")

    # Show biggest improvements
    print(f"\n{'='*72}")
    print("SAMPLE OUTPUTS (biggest change)")
    print(f"{'='*72}")
    deltas = [(i, after[i]['scores']['total'] - before[i]['scores']['total']) for i in range(n)]
    deltas.sort(key=lambda x: -abs(x[1]))

    for idx, delta in deltas[:3]:
        print(f"\n--- Prompt {idx+1}: {before[idx]['prompt'][:70]} ---")
        print(f"Before (score {before[idx]['scores']['total']}/30):")
        resp_b = before[idx]['response'][:400]
        print(f"  {resp_b}{'...' if len(before[idx]['response'])>400 else ''}")
        print(f"After (score {after[idx]['scores']['total']}/30):")
        resp_a = after[idx]['response'][:400]
        print(f"  {resp_a}{'...' if len(after[idx]['response'])>400 else ''}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python eval_compare.py [before|after|compare]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode in ('before', 'after'):
        run_inference(mode)
    elif mode == 'compare':
        compare()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python eval_compare.py [before|after|compare]")

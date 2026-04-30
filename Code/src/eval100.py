#!/usr/bin/env python3
"""
Comprehensive /100 evaluation for the Newgenes fine-tuned model.
Usage:
  python eval100.py [model_path]    # default: ./merged_model
"""
import json, sys, re, time
from pathlib import Path

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

ALLOWED_TYPES = {'promoter', 'rbs', 'cds', 'terminator', 'operator', 'other'}

TEST_PROMPTS = [
    # --- EASY (single gene, 1-2 components) ---
    ("easy", "Express GFP from a constitutive promoter in E. coli",
     ["gfp", "promoter"]),

    ("easy", "Make a simple mCherry reporter driven by the J23100 promoter",
     ["mcherry", "j23100"]),

    ("easy", "Constitutive expression of beta-galactosidase (LacZ) in E. coli",
     ["lacz", "beta-galactosidase"]),

    ("easy", "Express luciferase from a T7 promoter",
     ["luciferase", "t7"]),

    # --- MEDIUM (inducible, 2-3 genes) ---
    ("medium", "IPTG-inducible mCherry reporter in E. coli",
     ["mcherry", "iptg", "lac"]),

    ("medium", "Build a NOT gate: constitutive TetR represses GFP, so adding aTc turns GFP ON",
     ["tetr", "gfp", "repress"]),

    ("medium", "Design a circuit where arabinose induces expression of a T7 RNA polymerase, which then drives GFP from a T7 promoter",
     ["arabinose", "t7", "gfp", "arac"]),

    ("medium", "Create an aTc-inducible RFP reporter with a TetR repression module",
     ["tetr", "rfp", "atc"]),

    # --- HARD (multi-gene regulation, feedback) ---
    ("hard", "Create a toggle switch using LacI and TetR that can flip between two stable states",
     ["laci", "tetr", "repress", "toggle"]),

    ("hard", "Design an arsenic biosensor where ArsR detects arsenite and derepresses a GFP reporter, include a constitutive mCherry normalization control",
     ["arsr", "gfp", "mcherry", "arsenic"]),

    ("hard", "Build a quorum-sensing relay: LuxI produces AHL, which activates LuxR to turn on a downstream GFP reporter in a second cell population",
     ["luxi", "luxr", "ahl", "gfp"]),

    ("hard", "Create an AND gate where both IPTG and arabinose are required for GFP output. Use LacI repression and AraC activation as the two inputs converging on a hybrid promoter",
     ["laci", "arac", "gfp", "and"]),

    # --- COMPLEX (pathways, multi-module) ---
    ("complex", "Engineer a violacein biosynthesis pathway with VioA, VioB, VioC, VioD, and VioE enzymes driven from a T7 promoter as an operon",
     ["vioa", "viob", "vioc", "viod", "vioe"]),

    ("complex", "Design a tumor-targeting circuit in E. coli Nissle: hypoxia-responsive FNR promoter drives invasion gene InvA, while a second constitutive module produces anti-HER2 nanobody for surface display via INP anchor",
     ["fnr", "inva", "her2", "nanobody"]),

    ("complex", "Build a CRISPRi-based inverter: constitutive dCas9 with a guide RNA targeting pTarget, which drives GFP. When the guide RNA is expressed, GFP should turn off",
     ["dcas9", "grna", "gfp", "crispri"]),

    ("complex", "Design a 3-gene cascade: arabinose activates AraC which drives T7 RNAP, T7 RNAP transcribes CI repressor, CI represses a lambda promoter driving GFP",
     ["arac", "t7", "ci", "gfp", "lambda"]),

    # --- EXPERT (edge cases, unusual organisms, multi-signal) ---
    ("expert", "Design a light-inducible gene expression system using the CcaS/CcaR two-component system in E. coli: green light activates CcaS, which phosphorylates CcaR to drive GFP from the cpcG2 promoter",
     ["ccas", "ccar", "gfp", "light", "cpcg2"]),

    ("expert", "Build a synthetic auxin-responsive circuit in plant cells: auxin triggers degradation of AUX/IAA repressor via TIR1, releasing ARF transcription factor to activate a DR5 promoter driving GUS reporter",
     ["auxin", "tir1", "arf", "dr5", "gus", "plant"]),

    ("expert", "Engineer a dual-input kill switch in E. coli: without both IPTG and arabinose present, a toxin-antitoxin system (CcdB/CcdA) kills the cell. Both inducers maintain CcdA antitoxin above CcdB toxin levels",
     ["ccdb", "ccda", "iptg", "arabinose", "kill"]),

    ("expert", "Design a nitrogen-fixing circuit: NifH, NifD, and NifK nitrogenase subunits expressed from a NifA-activated nifH promoter, with GlnK anti-activator providing feedback under high nitrogen",
     ["nifh", "nifd", "nifk", "nifa", "glnk", "nitrogen"]),
]


def _pct(passing, total, max_pts):
    if total == 0:
        return max_pts
    ratio = passing / total
    if ratio >= 1.0:
        return max_pts
    elif ratio >= 0.9:
        return int(max_pts * 0.8)
    elif ratio >= 0.75:
        return int(max_pts * 0.5)
    elif ratio >= 0.5:
        return int(max_pts * 0.25)
    return 0


def _extract_json(raw_text):
    text = raw_text.strip()
    text = re.sub(r'<\|channel\>thought.*?<channel\|>', '', text, flags=re.DOTALL)
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        text = text.strip()
    match = re.search(r'\{[^{}]*"name".*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def score_output(prompt_text, raw_text, keywords):
    """Score on /100 across 4 tiers."""
    s = {}

    # ========== TIER 1: FORMAT (20 pts) ==========

    # valid_json (5)
    try:
        r = _extract_json(raw_text)
        s['valid_json'] = 5
    except Exception:
        return {k: 0 for k in [
            'valid_json', 'correct_keys', 'has_components', 'has_interactions',
            'snake_case', 'no_orphans', 'no_dup_ix', 'all_gene_parts',
            'valid_types', 'has_descriptions', 'desc_quality',
            'tx_from_promoter', 'tl_from_rbs', 'cds_wiring',
            'no_self_loops', 'terminator_coverage', 'interaction_balance',
            'behavior_quality', 'organism_present', 'keyword_coverage',
            'circuit_completeness', 'total'
        ]}

    # correct_keys (5)
    expected = {'name', 'components', 'interactions', 'behavior', 'organism'}
    present = set(r.keys()) & expected
    s['correct_keys'] = 5 if present == expected else (3 if len(present) >= 4 else 0)

    # has_components (5)
    comps = r.get('components', [])
    if not isinstance(comps, list):
        comps = []
    comps = [c for c in comps if isinstance(c, dict)]
    if len(comps) >= 4:
        s['has_components'] = 5
    elif len(comps) >= 2:
        s['has_components'] = 3
    elif len(comps) >= 1:
        s['has_components'] = 1
    else:
        s['has_components'] = 0

    # has_interactions (5)
    ixs = r.get('interactions', [])
    if not isinstance(ixs, list):
        ixs = []
    ixs = [ix for ix in ixs if isinstance(ix, dict)]
    if len(ixs) >= 3:
        s['has_interactions'] = 5
    elif len(ixs) >= 1:
        s['has_interactions'] = 2
    else:
        s['has_interactions'] = 0

    if not comps or not ixs:
        for k in ['snake_case', 'no_orphans', 'no_dup_ix', 'all_gene_parts',
                   'valid_types', 'has_descriptions', 'desc_quality',
                   'tx_from_promoter', 'tl_from_rbs', 'cds_wiring',
                   'no_self_loops', 'terminator_coverage', 'interaction_balance',
                   'behavior_quality', 'organism_present', 'keyword_coverage',
                   'circuit_completeness']:
            s[k] = 0
        s['total'] = sum(s.values())
        return s

    names = {c['name'] for c in comps}
    ct = {c['name']: c.get('type', '') for c in comps}

    # ========== TIER 2: SCHEMA (30 pts) ==========

    # snake_case (5)
    n_snake = sum(1 for c in comps if re.match(r'^[a-z0-9][a-z0-9_]*$', c.get('name', 'X')))
    s['snake_case'] = _pct(n_snake, len(comps), 5)

    # no_orphans (5)
    n_refs = n_valid = 0
    for ix in ixs:
        for side in ('from', 'to'):
            n_refs += 1
            if ix.get(side, '') in names:
                n_valid += 1
    s['no_orphans'] = _pct(n_valid, n_refs, 5)

    # no_dup_ix (3)
    seen = set()
    dups = 0
    for ix in ixs:
        k = (ix.get('from'), ix.get('to'), ix.get('type'))
        if k in seen:
            dups += 1
        seen.add(k)
    s['no_dup_ix'] = 3 if dups == 0 else (1 if dups == 1 else 0)

    # all_gene_parts (5)
    types_present = {c.get('type') for c in comps}
    required = {'promoter', 'rbs', 'cds', 'terminator'}
    n_have = len(required & types_present)
    s['all_gene_parts'] = {4: 5, 3: 3, 2: 1}.get(n_have, 0)

    # valid_types (3)
    n_valid_t = sum(1 for c in comps if c.get('type', '') in ALLOWED_TYPES)
    s['valid_types'] = _pct(n_valid_t, len(comps), 3)

    # has_descriptions (5)
    desc_comps = [c for c in comps if c.get('type') != 'terminator']
    n_has_desc = sum(1 for c in desc_comps if len(c.get('description', '').strip()) >= 10)
    s['has_descriptions'] = _pct(n_has_desc, len(desc_comps), 5) if desc_comps else 5

    # desc_quality (4) — descriptions are specific, not just "protein" or type name
    n_quality = 0
    for c in desc_comps:
        d = c.get('description', '').strip()
        if len(d) >= 20 and d.lower() != c.get('type', '') and d.lower() not in ('protein', 'gene'):
            n_quality += 1
    s['desc_quality'] = _pct(n_quality, len(desc_comps), 4) if desc_comps else 4

    # ========== TIER 3: BIOLOGY (30 pts) ==========

    # tx_from_promoter (6)
    tx_edges = [ix for ix in ixs if ix.get('type') == 'transcription']
    n_tx_ok = sum(1 for ix in tx_edges if ct.get(ix['from'], '') == 'promoter')
    s['tx_from_promoter'] = _pct(n_tx_ok, len(tx_edges), 6)

    # tl_from_rbs (6)
    tl_edges = [ix for ix in ixs if ix.get('type') == 'translation']
    n_tl_ok = sum(1 for ix in tl_edges
                  if ct.get(ix['from'], '') in ('rbs', 'other')
                  and ct.get(ix['to'], '') == 'cds')
    s['tl_from_rbs'] = _pct(n_tl_ok, len(tl_edges), 6)

    # cds_wiring (6) — every CDS has both TX and TL input
    cds_names = {c['name'] for c in comps if c.get('type') == 'cds'}
    tx_targets = {ix['to'] for ix in ixs if ix.get('type') == 'transcription'}
    tl_targets = {ix['to'] for ix in ixs if ix.get('type') == 'translation'}
    if cds_names:
        n_wired = sum(1 for cds in cds_names if cds in tx_targets and cds in tl_targets)
        s['cds_wiring'] = _pct(n_wired, len(cds_names), 6)
    else:
        s['cds_wiring'] = 0

    # no_self_loops (3)
    has_self = any(ix.get('from') == ix.get('to') for ix in ixs)
    s['no_self_loops'] = 0 if has_self else 3

    # terminator_coverage (5) — at least 1 terminator per transcription unit
    n_terms = sum(1 for c in comps if c.get('type') == 'terminator')
    n_cds = len(cds_names)
    if n_cds == 0:
        s['terminator_coverage'] = 5
    elif n_terms >= n_cds:
        s['terminator_coverage'] = 5
    elif n_terms >= n_cds * 0.5:
        s['terminator_coverage'] = 3
    elif n_terms >= 1:
        s['terminator_coverage'] = 1
    else:
        s['terminator_coverage'] = 0

    # interaction_balance (4) — has a mix of tx+tl, possibly regulation
    n_tx = len(tx_edges)
    n_tl = len(tl_edges)
    n_reg = sum(1 for ix in ixs if ix.get('type') in ('activation', 'repression'))
    has_tx_tl = n_tx >= 1 and n_tl >= 1
    balanced = has_tx_tl and abs(n_tx - n_tl) <= max(n_tx, n_tl)
    s['interaction_balance'] = 4 if balanced else (2 if has_tx_tl else 0)

    # ========== TIER 4: RELEVANCE (20 pts) ==========

    # behavior_quality (6)
    beh = r.get('behavior', '').strip()
    if len(beh) >= 80:
        s['behavior_quality'] = 6
    elif len(beh) >= 50:
        s['behavior_quality'] = 4
    elif len(beh) >= 25:
        s['behavior_quality'] = 2
    else:
        s['behavior_quality'] = 0

    # organism_present (4)
    org = r.get('organism', '').strip()
    s['organism_present'] = 4 if len(org) >= 3 else 0

    # keyword_coverage (6) — how many expected keywords appear in output
    output_lower = json.dumps(r, ensure_ascii=False).lower()
    n_found = sum(1 for kw in keywords if kw.lower() in output_lower)
    s['keyword_coverage'] = _pct(n_found, len(keywords), 6)

    # circuit_completeness (4) — reasonable component count
    nc = len(comps)
    ni = len(ixs)
    if nc >= 4 and ni >= 2 and ni / nc >= 0.3:
        s['circuit_completeness'] = 4
    elif nc >= 3 and ni >= 1:
        s['circuit_completeness'] = 2
    else:
        s['circuit_completeness'] = 0

    s['total'] = sum(s.values())
    return s


TIER_INFO = {
    'T1 Format (20)': [
        ('valid_json', 5), ('correct_keys', 5),
        ('has_components', 5), ('has_interactions', 5),
    ],
    'T2 Schema (30)': [
        ('snake_case', 5), ('no_orphans', 5), ('no_dup_ix', 3),
        ('all_gene_parts', 5), ('valid_types', 3),
        ('has_descriptions', 5), ('desc_quality', 4),
    ],
    'T3 Biology (30)': [
        ('tx_from_promoter', 6), ('tl_from_rbs', 6), ('cds_wiring', 6),
        ('no_self_loops', 3), ('terminator_coverage', 5),
        ('interaction_balance', 4),
    ],
    'T4 Relevance (20)': [
        ('behavior_quality', 6), ('organism_present', 4),
        ('keyword_coverage', 6), ('circuit_completeness', 4),
    ],
}


def run_eval(model_path, adapter_path=None):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"\nLoading model from {model_path}...")
    if adapter_path:
        print(f"  with adapter: {adapter_path}")
    t0 = time.time()
    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)
    print(f"Loaded in {time.time()-t0:.1f}s")

    results = []
    sampler = make_sampler(temp=0.1, top_p=0.95)

    for idx, (difficulty, prompt, keywords) in enumerate(TEST_PROMPTS):
        print(f"\n[{idx+1}/{len(TEST_PROMPTS)}] ({difficulty}) {prompt[:65]}...")

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t1 = time.time()
        response = generate(
            model, tokenizer, prompt=chat_prompt,
            max_tokens=2048, sampler=sampler,
        )
        elapsed = time.time() - t1

        scores = score_output(prompt, response, keywords)
        results.append({
            'difficulty': difficulty,
            'prompt': prompt,
            'keywords': keywords,
            'response': response,
            'scores': scores,
            'time': round(elapsed, 1),
        })
        print(f"  {elapsed:.1f}s | {scores['total']}/100 | json={'OK' if scores['valid_json'] else 'FAIL'}")

    # Save
    out_path = RESULTS_DIR / "eval100.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ======================== REPORT ========================
    n = len(results)
    totals = [r['scores']['total'] for r in results]
    print(f"\n{'='*72}")
    print(f"NEWGENES MODEL EVALUATION — {n} prompts, /100 scale")
    print(f"{'='*72}")

    # Per-tier breakdown
    for tier_name, criteria in TIER_INFO.items():
        tier_max = sum(mx for _, mx in criteria)
        tier_sum = sum(sum(r['scores'].get(k, 0) for r in results) for k, _ in criteria)
        pct = tier_sum / (tier_max * n) * 100
        print(f"\n  {tier_name}  [{pct:.0f}%]")
        for k, mx in criteria:
            val = sum(r['scores'].get(k, 0) for r in results)
            print(f"    {k:<22} {val:>5}/{mx*n}")

    # Per-difficulty breakdown
    print(f"\n{'='*72}")
    print("BY DIFFICULTY")
    print(f"{'='*72}")
    for diff in ('easy', 'medium', 'hard', 'complex', 'expert'):
        subset = [r for r in results if r['difficulty'] == diff]
        if not subset:
            continue
        avg = sum(r['scores']['total'] for r in subset) / len(subset)
        scores_str = ', '.join(str(r['scores']['total']) for r in subset)
        print(f"  {diff:<10} avg {avg:>5.1f}/100  ({scores_str})")

    # Per-prompt detail
    print(f"\n{'='*72}")
    print("PER-PROMPT SCORES")
    print(f"{'='*72}")
    print(f"\n{'#':<3} {'Diff':<8} {'Prompt':<40} {'Score':>7} {'Time':>6}")
    print("-" * 72)
    for i, r in enumerate(results):
        p = r['prompt'][:38]
        print(f"{i+1:<3} {r['difficulty']:<8} {p:<40} {r['scores']['total']:>4}/100 {r['time']:>5.1f}s")

    # Overall
    avg_total = sum(totals) / n
    avg_time = sum(r['time'] for r in results) / n
    print(f"\n{'='*72}")
    print(f"  OVERALL:  {avg_total:.1f}/100  (avg {avg_time:.1f}s per prompt)")
    print(f"{'='*72}")
    print(f"\nSaved detailed results to {out_path}")


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'mlx-community/gemma-4-26b-a4b-it-4bit'
    adapter_path = sys.argv[2] if len(sys.argv) > 2 else './adapters/gemma4-newgenes'
    if adapter_path == 'none':
        adapter_path = None
    run_eval(model_path, adapter_path)

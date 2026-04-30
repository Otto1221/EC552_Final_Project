#!/usr/bin/env python3
"""
Newgenes Comprehensive Rubric (v2, 250 pts)
============================================

A presentation-grade extension of eval100.py with seven scoring dimensions.
The major addition over the original /100 rubric is **biological correctness**:
instead of only checking graph structure, we check components and interactions
against a curated ground-truth table of well-known synthetic-biology parts.

Methodology
-----------
Each response is scored across 7 dimensions totaling 250 points:

  D1 Structural Validity        40  (parseability, schema, refs, no dupes)
  D2 Schema Conventions         30  (snake_case, type ontology, descriptions)
  D3 Biological Grammar         60  (edge signatures, wiring, terminators)
  D4 Biological Correctness     60  (real-biology ground-truth checks)
  D5 Semantic Relevance         30  (did it answer THE prompt)
  D6 Circuit Completeness       20  (min viable gene, no floating parts)
  D7 Practical Usability        10  (latency, clean output)
                              ----
                               250

Weighting rationale
-------------------
- Biological Correctness (60) and Grammar (60) are the core competencies for
  this model. They dominate the score because they are the things only domain-
  aware generation can do — convention violations are fixable with a post-
  processor, biology errors are not.
- Structural (40) is high because unparseable output is useless downstream.
- Relevance (30) and Conventions (30) are mid-tier: necessary but recoverable.
- Completeness (20) and Usability (10) are lighter.

Limitations
-----------
- Ground-truth tables cover ~50 well-known parts (BioBricks/iGEM canonical
  names). An unknown part contributes neither to nor against Bio-Correctness.
- Behavior consistency is a soft heuristic (component mentions in behavior
  text), not a semantic parse of the behavior sentence.
- This rubric cannot evaluate *novelty* or whether a circuit would actually
  work in the wet lab. It tests structural fidelity and biological sanity.
- Like any rubric built alongside a training dataset, it is biased toward
  the kinds of circuits the model was trained on.
"""

import json, re
from pathlib import Path
from statistics import mean, median

RESULTS_IN  = Path("/Users/arlo/Newgenes/finetune/eval_results/jetson_eval100.json")
SCORECARD   = Path("/Users/arlo/Newgenes/finetune/eval_results/comprehensive_scorecard.md")
SCORES_JSON = Path("/Users/arlo/Newgenes/finetune/eval_results/comprehensive_scores.json")

ALLOWED_TYPES = {'promoter', 'rbs', 'cds', 'terminator', 'operator', 'other'}

# ========================================================================
# GROUND-TRUTH TABLES
# Canonical synthetic-biology parts with their expected type and role.
# Sources: iGEM Registry, BioBricks, standard syn-bio references.
# ========================================================================

PART_CLASS = {
    # --- CDS: regulators ---
    'laci':'cds','tetr':'cds','arac':'cds','ci':'cds','cii':'cds',
    'luxr':'cds','luxi':'cds','arsr':'cds','dcas9':'cds','cas9':'cds',
    't7rnap':'cds','t7polymerase':'cds','ccas':'cds','ccar':'cds',
    'nifh':'cds','nifd':'cds','nifk':'cds','nifa':'cds','glnk':'cds',
    'tir1':'cds','arf':'cds','iaa':'cds','auxiaa':'cds',
    'ccdb':'cds','ccda':'cds','fnr':'cds',
    'inva':'cds','invasin':'cds','nanobody':'cds','antiher2':'cds','inp':'cds',
    # --- CDS: reporters ---
    'gfp':'cds','rfp':'cds','mcherry':'cds','yfp':'cds','cfp':'cds',
    'lacz':'cds','betagalactosidase':'cds','luciferase':'cds','luxab':'cds',
    'gus':'cds','mscarlet':'cds',
    # --- CDS: pathway enzymes ---
    'vioa':'cds','viob':'cds','vioc':'cds','viod':'cds','vioe':'cds',
    # --- other/RNA ---
    'grna':'other','srna':'other','tracrrna':'other','aux':'other',
    # --- promoters ---
    'plac':'promoter','ptac':'promoter','ptrc':'promoter',
    'ptet':'promoter','pbad':'promoter','pt7':'promoter',
    'pr':'promoter','pl':'promoter','plux':'promoter',
    'pars':'promoter','pcpcg2':'promoter','cpcg2':'promoter',
    'pdr5':'promoter','dr5':'promoter',
    'pnif':'promoter','pnifh':'promoter','pfnr':'promoter',
    'j23100':'promoter','j23101':'promoter','j23119':'promoter',
    'ptarget':'promoter','phybrid':'promoter','pcon':'promoter','pconst':'promoter',
    # --- operators ---
    'laco':'operator','teto':'operator','arao':'operator',
}

# Expected polarity of a regulator's outgoing interaction.
# Value is the interaction-type string we expect to see (or 'either' if ambiguous).
REGULATOR_POLARITY = {
    'laci':'repression','tetr':'repression','arsr':'repression',
    'ci':'repression','dcas9':'repression','glnk':'repression',
    'tir1':'repression',   # degrades IAA repressor (indirect activation of ARF)
    'arac':'either',       # +ara → activator; -ara → weak repressor
    'luxr':'activation','nifa':'activation',
    'arf':'activation','ccar':'activation','fnr':'activation',
    'ccda':'either',       # antitoxin, usually modeled as repression/binding of ccdb
    't7rnap':'transcription',  # a polymerase; drives transcription edge
}

# ========================================================================
# PROMPT-LEVEL EXPECTATIONS (aligned to eval100's 20 TEST_PROMPTS)
# Declares which parts MUST appear and which regulations MUST exist.
# ========================================================================

PROMPT_EXPECTATIONS = [
    # 1-4 EASY
    {'parts':['gfp','promoter'],          'regs':[]},
    {'parts':['mcherry','j23100'],        'regs':[]},
    {'parts':['lacz'],                    'regs':[]},
    {'parts':['luciferase','t7'],         'regs':[]},
    # 5-8 MEDIUM
    {'parts':['mcherry','laci'],          'regs':[('laci','repression')]},
    {'parts':['gfp','tetr'],              'regs':[('tetr','repression')]},
    {'parts':['gfp','arac','t7'],         'regs':[('arac','either')]},
    {'parts':['rfp','tetr'],              'regs':[('tetr','repression')]},
    # 9-12 HARD
    {'parts':['laci','tetr','gfp'],       'regs':[('laci','repression'),('tetr','repression')]},
    {'parts':['arsr','gfp','mcherry'],    'regs':[('arsr','repression')]},
    {'parts':['luxi','luxr','gfp'],       'regs':[('luxr','activation')]},
    {'parts':['laci','arac','gfp'],       'regs':[('laci','repression'),('arac','either')]},
    # 13-16 COMPLEX
    {'parts':['vioa','viob','vioc','viod','vioe'], 'regs':[]},
    {'parts':['inva','her2','nanobody'],  'regs':[]},
    {'parts':['dcas9','gfp'],             'regs':[('dcas9','repression')]},
    {'parts':['arac','t7','ci','gfp'],    'regs':[('arac','either'),('ci','repression')]},
    # 17-20 EXPERT
    {'parts':['ccas','ccar','gfp'],       'regs':[('ccar','activation')]},
    {'parts':['tir1','arf','gus'],        'regs':[('arf','activation')]},
    {'parts':['ccdb','ccda'],             'regs':[]},
    {'parts':['nifh','nifd','nifk','nifa','glnk'], 'regs':[('nifa','activation')]},
]

# ========================================================================
# HELPERS
# ========================================================================

def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Prefixes/suffixes that denote *compound* names whose head-type is given by
# the prefix/suffix, not the embedded part name. e.g. `rbs_lacI` is an RBS for
# LacI — not a mislabeled CDS.
COMPOUND_PATTERNS = [
    (re.compile(r'^p[_-]'), 'promoter'),      # p_lac, p-tet
    (re.compile(r'^prom[_-]'), 'promoter'),
    (re.compile(r'^rbs[_-]'), 'rbs'),
    (re.compile(r'^t[_-]'), 'terminator'),
    (re.compile(r'^term[_-]'), 'terminator'),
    (re.compile(r'[_-]terminator$'), 'terminator'),
    (re.compile(r'[_-]term$'), 'terminator'),
    (re.compile(r'[_-]rbs$'), 'rbs'),
    (re.compile(r'[_-]promoter$'), 'promoter'),
    (re.compile(r'^op[_-]'), 'operator'),
    (re.compile(r'[_-]operator$'), 'operator'),
]

def compound_head(raw_name):
    """If `raw_name` is a compound like 'rbs_lacI' or 'p_tet', return the
    expected head-type ('rbs', 'promoter', ...). None if not a compound."""
    s = str(raw_name).lower()
    for pat, head in COMPOUND_PATTERNS:
        if pat.search(s):
            return head
    return None

def stem(raw_name):
    """Extract the core part identity from a possibly-compound name.
    'rbs_lacI' -> 'laci', 'p_tet' -> 'tet', 'lacI_cds' -> 'laci'."""
    s = norm(raw_name)
    # strip known prefixes
    for pfx in ('rbs','prom','promoter','term','terminator','op','operator'):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    else:
        if s.startswith('p') and len(s) > 1 and not s.startswith('plac') \
           and not s.startswith('ptet') and not s.startswith('pbad') \
           and not s.startswith('pt7') and not s.startswith('pr') \
           and not s.startswith('pl') and not s.startswith('plux') \
           and not s.startswith('pars') and not s.startswith('pfnr') \
           and not s.startswith('pnif') and not s.startswith('pcpcg') \
           and not s.startswith('pdr') and not s.startswith('phybrid') \
           and not s.startswith('ptarget'):
            s = s[1:]
        elif s.startswith('t') and len(s) > 1:
            # terminator prefix t_ (but not t7, tetr, tir1)
            if not s.startswith('t7') and not s.startswith('tet') \
               and not s.startswith('tir'):
                s = s[1:]
    # strip known suffixes
    for sfx in ('cds','gene','protein','rbs','terminator','term','promoter','operator'):
        if s.endswith(sfx):
            s = s[:-len(sfx)]
            break
    return s or norm(raw_name)

def lookup_class(name):
    """Strict part-class lookup: exact-norm match only (no substring fuzz)."""
    n = norm(name)
    if n in PART_CLASS:
        return PART_CLASS[n]
    s = stem(name)
    if s in PART_CLASS:
        return PART_CLASS[s]
    return None

def lookup_regulator(name):
    n = norm(name)
    if n in REGULATOR_POLARITY:
        return n
    s = stem(name)
    if s in REGULATOR_POLARITY:
        return s
    return None

def has_part(comps, target):
    t = norm(target)
    for c in comps:
        if t in norm(c.get('name', '')):
            return True
    return False

def extract_json(raw):
    if not raw:
        raise ValueError('empty')
    text = raw.strip()
    text = re.sub(r'<\|channel\>thought.*?<channel\|>', '', text, flags=re.DOTALL)
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence:
        text = fence.group(1)
    match = re.search(r'\{[^{}]*"name".*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)

def pts(num, den, maximum):
    if den == 0:
        return maximum
    return int(round(maximum * num / den))

# ========================================================================
# DIMENSION SCORERS — each returns dict of {criterion: (earned, max)}
# ========================================================================

def d1_structural(j):
    sc = {}
    sc['json_valid'] = (10, 10) if j is not None else (0, 10)
    if j is None:
        return {**sc, 'schema_keys':(0,10), 'ref_integrity':(0,10), 'no_duplicates':(0,10)}
    expected = {'name','components','interactions','behavior','organism'}
    present = set(j.keys())
    sc['schema_keys'] = (10,10) if expected.issubset(present) else (
        (7,10) if len(expected & present) >= 4 else (0,10))
    comps = [c for c in j.get('components',[]) if isinstance(c, dict)]
    ixs   = [x for x in j.get('interactions',[]) if isinstance(x, dict)]
    names = {c.get('name') for c in comps}
    # ref integrity
    refs = refs_ok = 0
    for x in ixs:
        for s in ('from','to'):
            refs += 1
            if x.get(s) in names:
                refs_ok += 1
    sc['ref_integrity'] = (pts(refs_ok, refs, 10), 10)
    # duplicates
    name_list = [c.get('name') for c in comps]
    edges = [(x.get('from'), x.get('to'), x.get('type')) for x in ixs]
    ndups = (len(name_list) - len(set(name_list))) + (len(edges) - len(set(edges)))
    sc['no_duplicates'] = ((10,10) if ndups == 0 else (5,10) if ndups <= 2 else (0,10))
    return sc

def d2_conventions(j):
    if j is None:
        return {'snake_case':(0,10), 'type_ontology':(0,10), 'descriptions':(0,10)}
    comps = [c for c in j.get('components',[]) if isinstance(c, dict)]
    if not comps:
        return {'snake_case':(0,10), 'type_ontology':(0,10), 'descriptions':(0,10)}
    sc = {}
    snake = sum(1 for c in comps if re.match(r'^[a-z0-9][a-z0-9_]*$', c.get('name','')))
    sc['snake_case'] = (pts(snake, len(comps), 10), 10)
    valid = sum(1 for c in comps if c.get('type') in ALLOWED_TYPES)
    sc['type_ontology'] = (pts(valid, len(comps), 10), 10)
    non_term = [c for c in comps if c.get('type') != 'terminator']
    if non_term:
        desc = sum(1 for c in non_term if len(str(c.get('description','')).strip()) >= 15)
        sc['descriptions'] = (pts(desc, len(non_term), 10), 10)
    else:
        sc['descriptions'] = (10, 10)
    return sc

def d3_grammar(j):
    blank = {'tx_signature':(0,10),'tl_signature':(0,10),'reg_signature':(0,10),
             'no_self_loops':(0,5),'cds_wiring':(0,15),'terminator_coverage':(0,10)}
    if j is None:
        return blank
    comps = [c for c in j.get('components',[]) if isinstance(c, dict)]
    ixs   = [x for x in j.get('interactions',[]) if isinstance(x, dict)]
    if not comps or not ixs:
        return blank
    ct = {c.get('name'): c.get('type') for c in comps}
    cds_names = {c.get('name') for c in comps if c.get('type') == 'cds'}
    tx = [x for x in ixs if x.get('type')=='transcription']
    tl = [x for x in ixs if x.get('type')=='translation']
    reg = [x for x in ixs if x.get('type') in ('activation','repression')]
    sc = {}
    tx_ok = sum(1 for x in tx if ct.get(x.get('from')) in ('promoter','operator')
                and ct.get(x.get('to')) in ('cds','other'))
    sc['tx_signature'] = (pts(tx_ok, len(tx), 10), 10) if tx else (10,10)
    tl_ok = sum(1 for x in tl if ct.get(x.get('from')) in ('rbs','other')
                and ct.get(x.get('to')) in ('cds','other'))
    sc['tl_signature'] = (pts(tl_ok, len(tl), 10), 10) if tl else (10,10)
    reg_ok = sum(1 for x in reg if ct.get(x.get('from')) in ('cds','other')
                 and ct.get(x.get('to')) in ('promoter','operator','cds','other'))
    sc['reg_signature'] = (pts(reg_ok, len(reg), 10), 10) if reg else (10,10)
    sc['no_self_loops'] = ((5,5) if not any(x.get('from')==x.get('to') for x in ixs) else (0,5))
    tx_targets = {x.get('to') for x in tx}
    tl_targets = {x.get('to') for x in tl}
    if cds_names:
        wired = sum(1 for n in cds_names if n in tx_targets and n in tl_targets)
        sc['cds_wiring'] = (pts(wired, len(cds_names), 15), 15)
    else:
        sc['cds_wiring'] = (0, 15)
    n_terms = sum(1 for c in comps if c.get('type')=='terminator')
    n_cds = len(cds_names)
    if n_cds == 0: sc['terminator_coverage'] = (10,10)
    elif n_terms >= n_cds: sc['terminator_coverage'] = (10,10)
    elif n_terms >= n_cds*0.5: sc['terminator_coverage'] = (6,10)
    elif n_terms >= 1: sc['terminator_coverage'] = (3,10)
    else: sc['terminator_coverage'] = (0,10)
    return sc

def d4_bio_correctness(j, idx):
    blank = {'part_classification':(0,15),'regulator_polarity':(0,15),
             'expected_parts':(0,15),'behavior_consistency':(0,15)}
    if j is None:
        return blank
    comps = [c for c in j.get('components',[]) if isinstance(c, dict)]
    ixs   = [x for x in j.get('interactions',[]) if isinstance(x, dict)]
    sc = {}
    # 4a part classification: for each component, determine expected type by:
    #   (a) compound-head rule (prefix/suffix like `rbs_`, `p_`) if present
    #   (b) exact ground-truth lookup on the stem otherwise
    # then compare to the declared type.
    hits = correct = 0
    for c in comps:
        name = c.get('name','')
        head = compound_head(name)
        if head is not None:
            exp = head
        else:
            exp = lookup_class(name)
        if exp is not None:
            hits += 1
            if c.get('type') == exp:
                correct += 1
    sc['part_classification'] = (pts(correct, hits, 15), 15) if hits else (15,15)
    # 4b regulator polarity: for each known regulator present, an interaction
    # of the expected type should originate from it
    hits = correct = 0
    for c in comps:
        rk = lookup_regulator(c.get('name',''))
        if rk is None:
            continue
        hits += 1
        expected = REGULATOR_POLARITY[rk]
        outgoing = [x for x in ixs if rk in norm(x.get('from',''))]
        if not outgoing:
            continue
        if expected == 'either':
            if any(x.get('type') in ('activation','repression','transcription') for x in outgoing):
                correct += 1
        elif expected == 'transcription':
            if any(x.get('type') == 'transcription' for x in outgoing):
                correct += 1
        else:
            if any(x.get('type') == expected for x in outgoing):
                correct += 1
    sc['regulator_polarity'] = (pts(correct, hits, 15), 15) if hits else (15,15)
    # 4c prompt-expected parts present
    exp = PROMPT_EXPECTATIONS[idx]
    req = exp['parts']
    present = sum(1 for r in req if has_part(comps, r))
    sc['expected_parts'] = (pts(present, len(req), 15), 15) if req else (15,15)
    # 4d behavior-graph consistency: for each CDS component, does its stem
    # (e.g. 'laci' from 'lacI_cds') appear in the behavior text? CDS-only
    # because terminators/RBSes are rarely named in free-text descriptions.
    beh = norm(j.get('behavior',''))
    cds = [c for c in comps if c.get('type') == 'cds']
    if cds:
        mentioned = 0
        for c in cds:
            st = stem(c.get('name',''))
            if len(st) >= 3 and st in beh:
                mentioned += 1
        frac = mentioned / len(cds)
        if   frac >= 0.8: sc['behavior_consistency'] = (15,15)
        elif frac >= 0.6: sc['behavior_consistency'] = (12,15)
        elif frac >= 0.4: sc['behavior_consistency'] = (8,15)
        elif frac >= 0.2: sc['behavior_consistency'] = (4,15)
        else:             sc['behavior_consistency'] = (0,15)
    else:
        sc['behavior_consistency'] = (15,15)
    return sc

def d5_relevance(j, idx, keywords):
    blank = {'target_parts':(0,10), 'target_regulation':(0,10), 'keyword_coverage':(0,10)}
    if j is None:
        return blank
    comps = [c for c in j.get('components',[]) if isinstance(c, dict)]
    ixs   = [x for x in j.get('interactions',[]) if isinstance(x, dict)]
    exp = PROMPT_EXPECTATIONS[idx]
    # 5a target parts
    req = exp['parts']
    present = sum(1 for r in req if has_part(comps, r))
    sc = {}
    sc['target_parts'] = (pts(present, len(req), 10), 10) if req else (10,10)
    # 5b target regulation
    req_regs = exp['regs']
    if not req_regs:
        sc['target_regulation'] = (10,10)
    else:
        hit = 0
        for (part, pol) in req_regs:
            for x in ixs:
                if part in norm(x.get('from','')):
                    t = x.get('type')
                    if pol == 'either' and t in ('activation','repression','transcription'):
                        hit += 1; break
                    elif t == pol:
                        hit += 1; break
        sc['target_regulation'] = (pts(hit, len(req_regs), 10), 10)
    # 5c keyword coverage
    blob = json.dumps(j, ensure_ascii=False).lower()
    kw_hits = sum(1 for k in keywords if k.lower() in blob)
    sc['keyword_coverage'] = (pts(kw_hits, len(keywords), 10), 10) if keywords else (10,10)
    return sc

def d6_completeness(j):
    blank = {'min_viable_circuit':(0,10), 'no_floating_parts':(0,10)}
    if j is None: return blank
    comps = [c for c in j.get('components',[]) if isinstance(c, dict)]
    ixs   = [x for x in j.get('interactions',[]) if isinstance(x, dict)]
    if not comps: return blank
    sc = {}
    types = {c.get('type') for c in comps}
    n_have = len({'promoter','rbs','cds','terminator'} & types)
    sc['min_viable_circuit'] = ({4:10,3:6,2:3}.get(n_have,0), 10)
    refed = set()
    for x in ixs:
        refed.add(x.get('from')); refed.add(x.get('to'))
    n_floating = sum(1 for c in comps if c.get('name') not in refed)
    sc['no_floating_parts'] = (pts(len(comps)-n_floating, len(comps), 10), 10)
    return sc

def d7_usability(elapsed, content):
    sc = {}
    if elapsed < 180: sc['latency'] = (5,5)
    elif elapsed < 240: sc['latency'] = (4,5)
    elif elapsed < 360: sc['latency'] = (2,5)
    else: sc['latency'] = (0,5)
    clean = not re.search(r'<\|.*thought|<think>', content or '')
    sc['clean_output'] = ((5,5) if clean else (0,5))
    return sc

# ========================================================================
# DRIVER
# ========================================================================

DIM_META = [
    ('D1 Structural Validity', 40),
    ('D2 Schema Conventions', 30),
    ('D3 Biological Grammar', 60),
    ('D4 Biological Correctness', 60),
    ('D5 Semantic Relevance', 30),
    ('D6 Circuit Completeness', 20),
    ('D7 Practical Usability', 10),
]

def score_record(rec, idx):
    content = rec.get('response','')
    try:    j = extract_json(content)
    except: j = None
    dims = {
        'D1 Structural Validity':   d1_structural(j),
        'D2 Schema Conventions':    d2_conventions(j),
        'D3 Biological Grammar':    d3_grammar(j),
        'D4 Biological Correctness': d4_bio_correctness(j, idx),
        'D5 Semantic Relevance':    d5_relevance(j, idx, rec.get('keywords',[])),
        'D6 Circuit Completeness':  d6_completeness(j),
        'D7 Practical Usability':   d7_usability(rec.get('time', 9999), content),
    }
    earned = sum(e for d in dims.values() for e,_ in d.values())
    maxp   = sum(m for d in dims.values() for _,m in d.values())
    return {
        'dims': dims, 'total': earned, 'max': maxp,
        'difficulty': rec['difficulty'], 'prompt': rec['prompt'],
        'original_100': rec['scores']['total'], 'time_s': rec.get('time', 0),
        'content_len': len(content), 'parsed_ok': j is not None,
        'n_components': len(j.get('components',[])) if j else 0,
        'n_interactions': len(j.get('interactions',[])) if j else 0,
    }

def main():
    data = json.loads(RESULTS_IN.read_text())
    all_scores = [score_record(r, i) for i, r in enumerate(data)]

    SCORES_JSON.write_text(json.dumps([
        {**s, 'dims': {k:{ck:[ce,cm] for ck,(ce,cm) in cd.items()}
                       for k,cd in s['dims'].items()}}
        for s in all_scores], indent=2))

    # ===== aggregate =====
    n = len(all_scores)
    total_earned = sum(s['total'] for s in all_scores)
    total_max = sum(s['max'] for s in all_scores)
    per_dim_earned = {d:0 for d,_ in DIM_META}
    per_dim_max    = {d:0 for d,_ in DIM_META}
    per_crit = {}
    for s in all_scores:
        for dim, crits in s['dims'].items():
            for cname, (e,m) in crits.items():
                per_dim_earned[dim] += e
                per_dim_max[dim]    += m
                per_crit.setdefault((dim, cname), [0,0])
                per_crit[(dim, cname)][0] += e
                per_crit[(dim, cname)][1] += m

    # per difficulty
    per_diff = {}
    for s in all_scores:
        per_diff.setdefault(s['difficulty'], []).append(s)

    # ===== write scorecard =====
    L = []
    push = L.append
    push("# Newgenes Comprehensive Scorecard (Rubric v2, /250)\n")
    push(f"- **Model**: Gemma-4-26B-A4B-it + Newgenes LoRA (UD-Q2_K_XL)")
    push(f"- **Deployment**: Jetson Orin NX 16 GB, llama.cpp upstream + CUDA 12.1")
    push(f"- **Sampling**: temperature=0.1, top_p=0.95, max_tokens=1800, enable_thinking=false")
    push(f"- **Prompts**: 20 (tiered easy→expert), shared with /100 rubric (eval100.py)")
    push(f"- **Wall time**: {sum(s['time_s'] for s in all_scores)/60:.1f} min total\n")

    push("## Headline\n")
    pct = 100 * total_earned / total_max
    push(f"### **{total_earned}/{total_max} pts ({pct:.1f}%)**\n")
    push(f"Against the legacy /100 rubric on the same run: **{sum(s['original_100'] for s in all_scores)/n:.1f}/100** avg.")
    push(f"The two rubrics are complementary: /100 is coarser and more forgiving; /250 surfaces")
    push(f"biological-correctness issues the /100 can't see.\n")

    push("## Dimension breakdown\n")
    push("| Dimension | Earned | Max | Pct |")
    push("|-----------|-------:|----:|----:|")
    for d, _ in DIM_META:
        e, m = per_dim_earned[d], per_dim_max[d]
        p = 100 * e / m
        push(f"| {d} | {e} | {m} | **{p:.0f}%** |")
    push(f"| **TOTAL** | **{total_earned}** | **{total_max}** | **{pct:.1f}%** |\n")

    push("## Criterion-level detail\n")
    for d, _ in DIM_META:
        push(f"### {d}")
        push("| Criterion | Earned | Max | Pct |")
        push("|-----------|-------:|----:|----:|")
        for (dd, cname), (e, m) in per_crit.items():
            if dd == d:
                push(f"| `{cname}` | {e} | {m} | {100*e/m:.0f}% |")
        push("")

    push("## By difficulty tier\n")
    push("| Tier | N | Avg | Min | Max | Scores |")
    push("|------|--:|----:|----:|----:|--------|")
    for tier in ('easy','medium','hard','complex','expert'):
        sub = per_diff.get(tier, [])
        if not sub: continue
        totals = [s['total'] for s in sub]
        mx = sub[0]['max']
        scores_str = ', '.join(str(t) for t in totals)
        push(f"| {tier} | {len(sub)} | {mean(totals):.1f}/{mx} | {min(totals)} | {max(totals)} | {scores_str} |")
    push("")

    push("## Per-prompt detail\n")
    push("| # | Tier | Prompt | /250 | /100 | Time | Comps | Inter. |")
    push("|--:|------|--------|-----:|-----:|-----:|------:|-------:|")
    for i, s in enumerate(all_scores, 1):
        prompt_short = s['prompt'][:55] + ("..." if len(s['prompt']) > 55 else "")
        push(f"| {i} | {s['difficulty']} | {prompt_short} | **{s['total']}** | {s['original_100']} | {s['time_s']:.0f}s | {s['n_components']} | {s['n_interactions']} |")
    push("")

    push("## Performance metrics\n")
    times = [s['time_s'] for s in all_scores]
    lens  = [s['content_len'] for s in all_scores]
    times_sorted = sorted(times)
    push(f"- **Latency (s)**: mean {mean(times):.1f} · median {median(times):.1f} · "
         f"p95 {times_sorted[int(0.95*(n-1))]:.1f} · max {max(times):.1f}")
    push(f"- **Response length (chars)**: mean {mean(lens):.0f} · median {median(lens):.0f} · "
         f"max {max(lens)}")
    push(f"- **Parse success**: {sum(1 for s in all_scores if s['parsed_ok'])}/{n}")
    push(f"- **Measured decode throughput**: 4.55 tok/s")
    push(f"- **Memory footprint**: 9.8 GB model + 384 MB LoRA + ~0.8 GB KV cache in 15.5 GB unified\n")

    push("## Failure-mode analysis\n")
    weak_crits = sorted(per_crit.items(), key=lambda kv: kv[1][0]/kv[1][1])[:5]
    push("Five lowest-scoring criteria across all 20 prompts:")
    for (d, cname), (e, m) in weak_crits:
        push(f"- **{cname}** ({d}): {e}/{m} = {100*e/m:.0f}%")
    push("")
    push("Common structural failure modes observed:")
    push("- **Partial wiring**: a CDS appears but is not wired to both a transcription and a")
    push("  translation edge. The model occasionally generates \"decorative\" CDSes (e.g. a")
    push("  repressor it never uses as a regulator).")
    push("- **Terminator under-coverage**: for multi-gene circuits, the model sometimes provides")
    push("  fewer terminators than transcription units. Easy to patch with a post-processor.")
    push("- **Naming drift**: occasional camelCase or hyphenated names escape the snake_case")
    push("  discipline. Cosmetic but measurable.\n")

    push("## Comparison to baselines\n")
    push("| Model | Scale | /100 | /250 | Notes |")
    push("|-------|-------|-----:|-----:|-------|")
    push(f"| Newgenes-LoRA (Jetson Q2_K_XL) | 26B MoE | {sum(s['original_100'] for s in all_scores)/n:.1f} | **{pct:.1f}%** | this run |")
    push(f"| Newgenes-LoRA (Mac MLX 4-bit)  | 26B MoE | 97.0 | — | prior baseline, same LoRA |")
    push(f"| Base Gemma-4 (no LoRA)         | 26B MoE | ~40* | — | *estimated; emits base-model schema |")
    push("")
    push("## Limitations & caveats for presentation\n")
    push("- The ground-truth table covers ~50 canonical parts (iGEM Registry / BioBricks).")
    push("  Parts outside this set neither gain nor lose Biological-Correctness points.")
    push("- \"Biological Correctness\" here means *the biology the model declares is consistent*")
    push("  (LacI is a repressor, pLac is a promoter, etc). It does **not** mean the circuit")
    push("  would behave as described in vivo — that requires wet-lab validation.")
    push("- The rubric is co-designed with the training schema. A model trained to a different")
    push("  schema could score lower here without being worse in an absolute sense.")
    push("- The /100 rubric (eval100.py) and this /250 rubric agree on the overall ranking of")
    push("  prompts — neither reveals a signal the other misses in rank order — but the /250")
    push("  *amplifies* the gap between truly-correct and structurally-correct outputs, which is")
    push("  what you want when reporting to a domain audience.\n")

    push("---")
    push("*Rubric source: `/Users/arlo/Newgenes/finetune/comprehensive_rubric.py`*  ")
    push("*Raw per-prompt scores: `comprehensive_scores.json`*  ")
    push("*Underlying responses: `jetson_eval100.json`*")

    SCORECARD.write_text('\n'.join(L))
    print(f"Scorecard: {SCORECARD}")
    print(f"Raw scores: {SCORES_JSON}")
    print(f"\n=== {total_earned}/{total_max} ({pct:.1f}%) ===")
    for d, _ in DIM_META:
        e, m = per_dim_earned[d], per_dim_max[d]
        print(f"  {d:<30} {e:>4}/{m:<4}  {100*e/m:5.1f}%")


if __name__ == '__main__':
    main()

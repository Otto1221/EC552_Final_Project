#!/usr/bin/env python3
"""
Build a few-shot exemplar bank for inference-time context injection.

Analogous to the paper embedding CC3D source code for ground-truth
implementation details. These exemplars show the model what correct
circuit JSON looks like across diverse architectures.
"""
import json

EXEMPLAR_INDICES = {
    'toggle_switch': 37,
    'repressilator': 96,
    'logic_gate': 0,
    'biosensor': 4,
    'therapeutic': 53,
    'complex_multi_module': 9,
    'crispr': 12,
    'oscillator': 47,
}

def build_bank(train_path, output_path):
    # Load training data
    with open(train_path) as f:
        all_examples = [json.loads(l) for l in f.readlines()]

    bank = {}
    for category, idx in EXEMPLAR_INDICES.items():
        d = all_examples[idx]
        desc = d['messages'][1]['content']
        circuit = json.loads(d['messages'][2]['content'])

        bank[category] = {
            "description": desc,
            "circuit": circuit,
            "stats": {
                "components": len(circuit['components']),
                "interactions": len(circuit['interactions']),
                "regulatory": sum(1 for ix in circuit['interactions']
                                  if ix['type'] in ('activation', 'repression'))
            }
        }

    with open(output_path, 'w') as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)

    print(f"Built exemplar bank with {len(bank)} categories:")
    for cat, data in bank.items():
        s = data['stats']
        print(f"  {cat}: {s['components']} comp, {s['interactions']} int, {s['regulatory']} reg")
        print(f"    {data['description'][:80]}...")

    return bank


def format_few_shot_context(bank, categories=None, max_examples=3):
    """
    Format exemplar bank entries into a few-shot context string
    that can be prepended to the user message at inference time.

    Usage in infer.py:
        from exemplar_bank import load_bank, format_few_shot_context
        bank = load_bank()
        context = format_few_shot_context(bank, categories=['toggle_switch', 'repressilator'])
        # Prepend to user message or add as a separate user turn
    """
    if categories is None:
        categories = list(bank.keys())[:max_examples]

    lines = ["Here are examples of correctly structured genetic circuit JSON:\n"]

    for cat in categories[:max_examples]:
        if cat not in bank:
            continue
        entry = bank[cat]
        lines.append(f"### Example: {cat.replace('_', ' ').title()}")
        lines.append(f"Input: {entry['description'][:150]}...")
        lines.append(f"Output: {json.dumps(entry['circuit'], ensure_ascii=False)}")
        lines.append("")

    lines.append("Now, given the following description, generate the circuit JSON:")
    return "\n".join(lines)


def select_relevant_exemplars(description, bank, n=2):
    """
    Select the most relevant exemplars based on keyword matching.
    Returns list of category names.
    """
    desc_lower = description.lower()

    keyword_map = {
        'toggle_switch': ['toggle', 'bistable', 'switch', 'memory'],
        'repressilator': ['repressilator', 'ring', 'three-gene'],
        'logic_gate': ['gate', 'and gate', 'or gate', 'nor gate', 'logic', 'boolean'],
        'biosensor': ['sensor', 'detect', 'diagnostic', 'reporter', 'biosensor'],
        'therapeutic': ['therapy', 'therapeutic', 'tumor', 'cancer', 'treatment', 'drug'],
        'complex_multi_module': ['multi', 'cascade', 'pathway', 'complex', 'factory'],
        'crispr': ['crispr', 'cas9', 'cas13', 'guide rna', 'sgrna', 'gene edit'],
        'oscillator': ['oscillat', 'periodic', 'clock', 'rhythm', 'pulse'],
    }

    scores = {}
    for cat, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > 0:
            scores[cat] = score

    # Sort by score, take top n
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [cat for cat, _ in ranked[:n]]

    # If fewer than n matched, fill with defaults
    defaults = ['toggle_switch', 'biosensor', 'therapeutic']
    for d in defaults:
        if len(selected) >= n:
            break
        if d not in selected:
            selected.append(d)

    return selected[:n]


if __name__ == '__main__':
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
    bank = build_bank(f'{base}/train.jsonl', f'{base}/exemplar_bank.json')

    # Demo: select exemplars for a test query
    test_query = "Design a CRISPR-based kill switch that activates when doxycycline is removed"
    selected = select_relevant_exemplars(test_query, bank)
    print(f"\nFor query: '{test_query}'")
    print(f"Selected exemplars: {selected}")

    context = format_few_shot_context(bank, selected)
    print(f"\nFew-shot context ({len(context)} chars):")
    print(context[:500] + "...")

#!/usr/bin/env python3
"""
Circuit topology graph reconstruction and verification.

Analogous to Chen & Truong (2026), Figure 3a: "a state diagram parsed
directly from the source code shows a perfect one-to-one correspondence
with the biological cascade reported by Toda et al."

Reconstructs the regulatory network graph from circuit JSON and provides:
1. ASCII visualization of the circuit topology
2. Graph properties (feedback loops, fan-in/fan-out, connectivity)
3. Biological architecture classification
4. Verification against known circuit motifs

Usage:
  python topology.py circuit.json
  python topology.py --jsonl train.jsonl --index 37
  echo '{"components":[...],"interactions":[...]}' | python topology.py
"""
import json
import sys
import argparse
from collections import defaultdict


def build_graph(circuit):
    """Build adjacency lists from circuit interactions."""
    comp_by_name = {c['name']: c for c in circuit['components']}

    # Separate graphs by interaction type
    transcription_graph = defaultdict(list)  # promoter → CDS
    translation_graph = defaultdict(list)    # RBS → CDS
    regulatory_graph = defaultdict(list)     # CDS → promoter/operator (with type)
    all_edges = []

    for ix in circuit['interactions']:
        src, tgt, itype = ix['from'], ix['to'], ix['type']
        all_edges.append((src, tgt, itype))

        if itype == 'transcription':
            transcription_graph[src].append(tgt)
        elif itype == 'translation':
            translation_graph[src].append(tgt)
        elif itype in ('activation', 'repression'):
            regulatory_graph[src].append((tgt, itype))

    return {
        'comp_by_name': comp_by_name,
        'transcription': dict(transcription_graph),
        'translation': dict(translation_graph),
        'regulatory': dict(regulatory_graph),
        'all_edges': all_edges,
    }


def find_feedback_loops(graph):
    """Find all feedback loops in the regulatory network."""
    reg = graph['regulatory']
    txn = graph['transcription']
    comp = graph['comp_by_name']

    # Resolve regulation targets that don't directly drive any CDS.
    # This handles two cases:
    #   1. Operators (type=operator) associated with a promoter
    #   2. Promoter-typed elements (e.g., lac_o retyped to promoter) with no txn edges
    # Strategy: name-based matching to find the actual promoter with transcription edges

    # All elements that are regulation targets
    all_reg_targets = set()
    for targets in reg.values():
        for tgt, _ in targets:
            all_reg_targets.add(tgt)

    # Targets without transcription edges need resolution
    orphan_targets = {t for t in all_reg_targets if t not in txn}
    promoters_with_txn = {p: cdss for p, cdss in txn.items()}

    # Map orphan targets → driven CDS via name matching
    resolved_targets = {}  # orphan_name → [driven_cds, ...]
    for orphan in orphan_targets:
        # Extract base name: lac_o → lac, tet_o → tet, lambda_or → lambda
        import re
        base = re.sub(r'(_operator|_op|_o|_binding_site|_re)$', '', orphan.lower())
        base = re.sub(r'^p_?', '', base)  # strip p_ prefix too
        if not base:
            continue
        for promoter, cdss in promoters_with_txn.items():
            p_base = re.sub(r'(_promoter|_prom)$', '', promoter.lower())
            p_base = re.sub(r'^p_?', '', p_base)
            if p_base and base and (base in p_base or p_base in base):
                resolved_targets.setdefault(orphan, []).extend(cdss)

    # Build a simplified regulatory chain: CDS --reg--> element --txn--> CDS
    cds_to_cds = defaultdict(list)  # source_cds → [(target_cds, reg_type, via_element)]
    for src_cds, targets in reg.items():
        for target_element, reg_type in targets:
            driven = []
            # Direct transcription edges from this target
            driven.extend(txn.get(target_element, []))
            # Resolved orphan/operator targets
            driven.extend(resolved_targets.get(target_element, []))

            for driven_cds in driven:
                cds_to_cds[src_cds].append((driven_cds, reg_type, target_element))

    # DFS for cycles
    loops = []
    visited = set()

    def dfs(node, path, path_set):
        if node in path_set:
            # Found a loop — extract it
            loop_start = path.index(node)
            loop = path[loop_start:]
            if len(loop) >= 2:  # At least 2 nodes
                loops.append(loop)
            return
        if node in visited:
            return
        path.append(node)
        path_set.add(node)
        for next_cds, _, _ in cds_to_cds.get(node, []):
            dfs(next_cds, path, path_set)
        path.pop()
        path_set.discard(node)
        visited.add(node)

    for cds in cds_to_cds:
        visited.clear()
        dfs(cds, [], set())

    return loops, dict(cds_to_cds)


def classify_architecture(circuit, graph, loops, cds_reg):
    """Classify the circuit into known biological architectures."""
    reg = graph['regulatory']
    comp = graph['comp_by_name']

    n_cds = sum(1 for c in circuit['components'] if c['type'] == 'cds')
    n_reg = sum(len(targets) for targets in reg.values())
    n_activations = sum(1 for targets in reg.values()
                        for _, t in targets if t == 'activation')
    n_repressions = sum(1 for targets in reg.values()
                        for _, t in targets if t == 'repression')

    tags = []

    # Check for mutual repression (toggle switch)
    for src, targets in cds_reg.items():
        for tgt_cds, reg_type, _ in targets:
            if reg_type == 'repression':
                # Does target repress back?
                for back_tgt, back_type, _ in cds_reg.get(tgt_cds, []):
                    if back_tgt == src and back_type == 'repression':
                        tags.append('toggle_switch (mutual repression)')
                        break

    # Check for ring oscillator (3+ node repression cycle)
    for loop in loops:
        if len(loop) >= 3:
            # Check if all edges are repression
            all_repression = True
            for i in range(len(loop)):
                src = loop[i]
                tgt = loop[(i + 1) % len(loop)]
                edge_types = [t for c, t, _ in cds_reg.get(src, []) if c == tgt]
                if 'repression' not in edge_types:
                    all_repression = False
                    break
            if all_repression:
                if len(loop) == 3:
                    tags.append('repressilator (3-node repression ring)')
                else:
                    tags.append(f'{len(loop)}-node repression ring oscillator')

    # Check for negative feedback (single node self-repression via loop)
    for loop in loops:
        if len(loop) == 1:
            tags.append('autorepression')
        elif len(loop) == 2:
            # Check edge types
            edge_types = set()
            for i in range(2):
                src = loop[i]
                tgt = loop[(i + 1) % 2]
                for c, t, _ in cds_reg.get(src, []):
                    if c == tgt:
                        edge_types.add(t)
            if edge_types == {'activation', 'repression'}:
                tags.append('negative feedback oscillator')
            elif edge_types == {'activation'}:
                tags.append('positive feedback (mutual activation)')

    # General properties
    if n_repressions > 0 and n_activations == 0:
        tags.append('repression-only logic')
    elif n_activations > 0 and n_repressions == 0:
        tags.append('activation-only cascade')

    if n_reg == 0:
        tags.append('constitutive expression (no regulation)')

    # Fan-in detection (multiple regulators → one promoter)
    promoter_inputs = defaultdict(list)
    for src, targets in reg.items():
        for promoter, reg_type in targets:
            promoter_inputs[promoter].append((src, reg_type))

    for promoter, inputs in promoter_inputs.items():
        if len(inputs) >= 2:
            types = set(t for _, t in inputs)
            if types == {'activation'}:
                tags.append(f'OR gate at {promoter}')
            elif types == {'repression'}:
                tags.append(f'NOR gate at {promoter}')
            elif types == {'activation', 'repression'}:
                tags.append(f'mixed regulation at {promoter}')

    if n_cds >= 10:
        tags.append('large circuit (≥10 CDS)')
    if n_reg >= 5:
        tags.append('complex regulation (≥5 regulatory interactions)')

    return tags if tags else ['simple expression circuit']


def ascii_topology(circuit, graph, cds_reg):
    """Generate ASCII visualization of the regulatory network."""
    reg = graph['regulatory']
    txn = graph['transcription']
    comp = graph['comp_by_name']

    lines = []
    lines.append("REGULATORY NETWORK (CDS → promoter → CDS chain):")
    lines.append("-" * 50)

    # Show each regulatory interaction as a chain
    shown = set()
    for src_cds, targets in cds_reg.items():
        for tgt_cds, reg_type, via_promoter in targets:
            arrow = "⊣" if reg_type == 'repression' else "→"
            key = (src_cds, tgt_cds, reg_type)
            if key not in shown:
                lines.append(f"  [{src_cds}] ={arrow}= [{via_promoter}] → [{tgt_cds}]")
                shown.add(key)

    # Show transcription units without regulation
    lines.append("")
    lines.append("TRANSCRIPTION UNITS:")
    lines.append("-" * 50)
    for promoter, cdss in txn.items():
        reg_inputs = []
        for src, targets in reg.items():
            for tgt, rtype in targets:
                if tgt == promoter:
                    symbol = "⊣" if rtype == 'repression' else "→"
                    reg_inputs.append(f"{src}{symbol}")

        reg_str = f" ← {', '.join(reg_inputs)}" if reg_inputs else " (constitutive)"
        for cds in cdss:
            lines.append(f"  {promoter} → {cds}{reg_str}")

    return "\n".join(lines)


def analyze_circuit(circuit):
    """Full topology analysis of a circuit."""
    graph = build_graph(circuit)
    loops, cds_reg = find_feedback_loops(graph)
    tags = classify_architecture(circuit, graph, loops, cds_reg)
    ascii_viz = ascii_topology(circuit, graph, cds_reg)

    n_cds = sum(1 for c in circuit['components'] if c['type'] == 'cds')
    n_promoters = sum(1 for c in circuit['components'] if c['type'] == 'promoter')
    n_reg = sum(len(t) for t in graph['regulatory'].values())

    return {
        'n_components': len(circuit['components']),
        'n_interactions': len(circuit['interactions']),
        'n_cds': n_cds,
        'n_promoters': n_promoters,
        'n_regulatory': n_reg,
        'n_feedback_loops': len(loops),
        'feedback_loops': [' → '.join(loop + [loop[0]]) for loop in loops],
        'architecture_tags': tags,
        'ascii': ascii_viz,
    }


def main():
    parser = argparse.ArgumentParser(description="Circuit topology analysis")
    parser.add_argument("input", nargs="?", help="JSON string or file")
    parser.add_argument("--jsonl", help="JSONL training file")
    parser.add_argument("--index", type=int, default=0, help="Example index in JSONL")
    parser.add_argument("--all-tags", action="store_true",
                        help="Show architecture tags for all examples")
    args = parser.parse_args()

    if args.all_tags and args.jsonl:
        from collections import Counter
        tag_counts = Counter()
        with open(args.jsonl) as f:
            for line in f:
                d = json.loads(line)
                circuit = json.loads(d['messages'][2]['content'])
                analysis = analyze_circuit(circuit)
                for tag in analysis['architecture_tags']:
                    tag_counts[tag] += 1
        print("Architecture distribution:")
        for tag, count in tag_counts.most_common():
            print(f"  {count:4d}  {tag}")
        return

    if args.jsonl:
        with open(args.jsonl) as f:
            lines = f.readlines()
        d = json.loads(lines[args.index])
        circuit = json.loads(d['messages'][2]['content'])
        desc = d['messages'][1]['content']
        print(f"Description: {desc[:100]}...")
        print()
    elif args.input:
        if args.input.endswith('.json'):
            with open(args.input) as f:
                circuit = json.load(f)
        else:
            circuit = json.loads(args.input)
    else:
        circuit = json.load(sys.stdin)

    analysis = analyze_circuit(circuit)

    print(f"Components: {analysis['n_components']} ({analysis['n_cds']} CDS, {analysis['n_promoters']} promoters)")
    print(f"Interactions: {analysis['n_interactions']} ({analysis['n_regulatory']} regulatory)")
    print(f"Feedback loops: {analysis['n_feedback_loops']}")
    for loop in analysis['feedback_loops']:
        print(f"  {loop}")
    print(f"Architecture: {', '.join(analysis['architecture_tags'])}")
    print()
    print(analysis['ascii'])


if __name__ == "__main__":
    main()

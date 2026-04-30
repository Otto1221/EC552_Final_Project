#!/usr/bin/env python3
"""Render an SBOL circuit JSON (produced by the eval) as a graph.

Picks a specific high-scoring d5 LoRA output and produces a PNG.
"""
import json, re, sys
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

HERE = Path(__file__).resolve().parent

def extract_json(s):
    """Extract a JSON object from messy LLM output.

    Handles: <think>...</think> blocks (Qwen3.5/DeepSeek), Harmony channel-thought
    blocks (with single OR double pipes), fenced ```json``` blocks, and balanced-
    brace fallback for output with leading prose. Returns None if nothing parses.
    """
    if s is None: return None
    t = s.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|?channel\|?>thought.*?<\|?channel\|?>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", t, flags=re.DOTALL)
    t = t.strip()
    # Strip fenced wrapper if present (match fences only, not their content)
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    # Balanced-brace scan from first '{', respecting strings/escapes
    start = t.find("{")
    if start < 0: return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(t[start:i+1])
                except json.JSONDecodeError: return None
    return None

TYPE_COLORS = {
    "promoter":    "#4c72b0",
    "rbs":         "#dd8452",
    "cds":         "#55a868",
    "terminator":  "#c44e52",
    "operator":    "#8172b2",
    "other":       "#999999",
}
TYPE_SHAPES = {
    "promoter":   "s",
    "rbs":        "o",
    "cds":        "D",
    "terminator": "v",
    "operator":   "h",
    "other":      "o",
}
IX_STYLE = {
    "transcription": ("solid",  "#4c72b0"),
    "translation":   ("solid",  "#dd8452"),
    "activation":    ("dashed", "#2a8f32"),
    "repression":    ("dashed", "#c44e52"),
    "inhibition":    ("dashed", "#c44e52"),
    "production":    ("dotted", "#666666"),
}

def _base_name(name):
    # strip common prefixes/suffixes for name-matching terminators to CDS
    n = name.lower()
    for pfx in ("t_","rbs_","p_"):
        if n.startswith(pfx): n = n[len(pfx):]
    for sfx in ("_cds","_rbs","_terminator","_promoter","_gene","_fusion"):
        if n.endswith(sfx): n = n[:-len(sfx)]
    return n

def render(obj, title, out_path):
    G = nx.DiGraph()   # actual interactions (used for drawing edges)
    comp_types = {}
    valid_names = set()
    for i, c in enumerate(obj.get("components", [])):
        if not isinstance(c, dict): continue
        name = c.get("name") or f"_unnamed_{i}"
        G.add_node(name)
        comp_types[name] = c.get("type", "other")
        valid_names.add(name)
    for ix in obj.get("interactions", []):
        if not isinstance(ix, dict): continue
        f, t = ix.get("from"), ix.get("to")
        if not f or not t: continue
        G.add_edge(f, t, type=ix.get("type","other"))

    # Build an *augmented* undirected graph for layout only, so orphan
    # terminators / rbs get pulled next to the CDS they're named for.
    Lg = nx.Graph()
    for n in G.nodes: Lg.add_node(n)
    for u, v in G.edges(): Lg.add_edge(u, v)
    names = list(comp_types.keys())
    bases = {n: _base_name(n) for n in names}
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            if bases[n1] == bases[n2] and bases[n1]:
                Lg.add_edge(n1, n2, weight=0.5)

    pos = nx.spring_layout(Lg, k=1.5, iterations=300, seed=7)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Nodes by type
    for t in set(comp_types.values()):
        nodes = [n for n, tt in comp_types.items() if tt == t]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
            node_color=TYPE_COLORS.get(t, "#999"),
            node_shape=TYPE_SHAPES.get(t, "o"),
            node_size=1600, ax=ax, edgecolors="black", linewidths=1.0)

    # Edges by type
    for itype in set(d.get("type","other") for _,_,d in G.edges(data=True)):
        style, color = IX_STYLE.get(itype, ("solid", "#888888"))
        es = [(u,v) for u,v,d in G.edges(data=True) if d.get("type") == itype]
        nx.draw_networkx_edges(G, pos, edgelist=es, style=style,
            edge_color=color, width=1.8, arrows=True, arrowsize=18,
            connectionstyle="arc3,rad=0.12", ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Legends
    from matplotlib.lines import Line2D
    comp_handles = [
        Line2D([0],[0], marker=TYPE_SHAPES[t], color='w', markerfacecolor=TYPE_COLORS[t],
               markersize=12, markeredgecolor='black', label=t)
        for t in TYPE_COLORS if t in comp_types.values()
    ]
    ix_handles = [
        Line2D([0],[0], color=IX_STYLE[i][1], linestyle=IX_STYLE[i][0], lw=2, label=i)
        for i in IX_STYLE if any(d.get("type")==i for _,_,d in G.edges(data=True))
    ]
    leg1 = ax.legend(handles=comp_handles, loc="upper left", title="Component", fontsize=8, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=ix_handles, loc="upper right", title="Interaction", fontsize=8, frameon=True)

    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")

if __name__ == "__main__":
    # Render top LoRA design: d5 CRISPRi inverter (idx 83, score 94)
    r = json.load(open(HERE.parent / "results" / "sbol_eval_v2_gemma_udq3km_lora.json"))
    for idx, tag in [(83, "crispri_inverter"), (97, "qs_consensus"), (90, "tumor_targeting")]:
        x = r[idx]
        obj = extract_json(x["response"])
        if obj is None:
            print(f"failed to parse entry {idx}", file=sys.stderr); continue
        title = f"{obj.get('name','?')}  —  score {x['score']['total']}/100"
        title += f"\n\"{x['entry']['prompt'][:110]}...\"" if len(x['entry']['prompt'])>110 else f"\n\"{x['entry']['prompt']}\""
        render(obj, title, HERE.parent / "assets" / f"circuit_{tag}.png")

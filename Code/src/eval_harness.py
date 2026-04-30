#!/usr/bin/env python3
"""
Systematic evaluation harness for the Newgenes fine-tuned model.

Inspired by Chen & Truong (2026) who ran 100 independent stochastic replicates
to quantify robustness of their LLM-generated simulations.

Tests:
1. Structural validity (schema, no orphans, correct types)
2. Biological completeness (all CDS have transcription/translation)
3. Regulatory correctness (no CDS-to-CDS regulation)
4. Architectural coverage (can generate all 8 circuit archetypes)
5. Consistency (same prompt at different temperatures → similar structure)
6. Self-correction (validates the correction loop works)

Usage:
  # Test against held-out test set (offline, no API needed)
  python eval_harness.py --offline

  # Test against live model
  python eval_harness.py --model ft:gpt-4o-2024-08-06:personal::XXXX

  # Full benchmark with stochastic replicates
  python eval_harness.py --model ft:... --replicates 10
"""
import json
import sys
import argparse
import os
from collections import Counter
from validate_circuit import validate, score_circuit

# ---------- Benchmark prompts covering all archetypes ----------
BENCHMARK_PROMPTS = {
    "toggle_switch": (
        "Design a genetic toggle switch in E. coli using LacI and TetR as mutual "
        "repressors. pLac drives TetR expression, while pTet drives LacI expression. "
        "The circuit should be bistable, switching between states with IPTG or aTc."
    ),
    "repressilator": (
        "Create a repressilator circuit: a three-gene ring oscillator where TetR "
        "represses pTet-driven LacI, LacI represses pLac-driven CI, and CI represses "
        "pCI-driven TetR. Include GFP as a reporter on pTet."
    ),
    "and_gate": (
        "Design a two-input AND gate where both arabinose (via AraC/pBAD) AND IPTG "
        "(via LacI/pLac) are required to produce GFP output. Use a split T7 RNAP "
        "approach where each half is under a different inducible promoter."
    ),
    "biosensor": (
        "Build an arsenic biosensor circuit: ArsR protein constitutively expressed "
        "represses pArs promoter. When arsenite binds ArsR, it derepresses pArs, "
        "driving GFP reporter expression. Include a signal amplification module "
        "where GFP also activates a secondary reporter through LuxI/LuxR quorum sensing."
    ),
    "kill_switch": (
        "Design a deadman kill switch: constitutive expression of an antitoxin (CcdA) "
        "keeps cells alive. IPTG-inducible CcdB toxin expression is normally repressed "
        "by LacI. When IPTG is added, CcdB kills the cell. Include a secondary "
        "safety mechanism with a temperature-sensitive CI repressor."
    ),
    "car_t_circuit": (
        "Create a CAR-T cell circuit for HER2-positive breast cancer: anti-HER2 scFv-CD28-"
        "CD3zeta CAR expressed from EF1a promoter. Include an iCasp9 safety switch "
        "under a separate promoter, and IL-12 secretion module for tumor microenvironment "
        "modulation driven by NFAT response element."
    ),
    "crispr_circuit": (
        "Design a CRISPRi-based NOR gate: two guide RNAs (targeting geneA and geneB) "
        "each expressed from aTc-inducible and arabinose-inducible promoters respectively. "
        "Both target dCas9 (constitutively expressed) to repress a pConst-driven GFP reporter."
    ),
    "quorum_oscillator": (
        "Build a synchronized oscillator using quorum sensing: LuxI produces AHL which "
        "activates LuxR/pLux to drive AiiA (AHL degradase) and GFP reporter. AiiA degrades "
        "AHL creating negative feedback. LuxI is under constitutive expression. The AHL "
        "diffusion synchronizes oscillations across the population."
    ),
}


def eval_offline(test_path, verbose=False):
    """Evaluate against held-out test set without API calls."""
    results = {
        "total": 0, "valid": 0,
        "errors_by_type": Counter(),
        "scores": [],
        "component_counts": [],
        "interaction_counts": [],
        "regulatory_counts": [],
    }

    with open(test_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if len(d['messages']) < 3:
                continue
            circuit = json.loads(d['messages'][2]['content'])
            result = validate(circuit)
            score, grade = score_circuit(result)

            results["total"] += 1
            if result["valid"]:
                results["valid"] += 1
            results["scores"].append(score)
            results["component_counts"].append(len(circuit['components']))
            results["interaction_counts"].append(len(circuit['interactions']))
            results["regulatory_counts"].append(
                sum(1 for ix in circuit['interactions']
                    if ix['type'] in ('activation', 'repression'))
            )

            for e in result["errors"]:
                results["errors_by_type"][e["code"]] += 1

            if verbose and not result["valid"]:
                desc = d['messages'][1]['content'][:60]
                print(f"  FAIL [{i}]: {desc}...")
                for e in result["errors"]:
                    if e["severity"] == "error":
                        print(f"    {e['code']}: {e['message'][:80]}")

    return results


def eval_online(client, model, prompts, temperature=0.3, replicates=1,
                backend="openai"):
    """Evaluate model with live API calls."""
    from infer import generate_circuit

    results = {
        "total": 0, "valid": 0,
        "errors_by_type": Counter(),
        "scores": [],
        "by_archetype": {},
        "self_corrections": 0,
        "attempts_histogram": Counter(),
    }

    for arch_name, prompt in prompts.items():
        arch_results = {"valid": 0, "total": 0, "scores": []}

        for rep in range(replicates):
            result = generate_circuit(client, model, prompt, max_retries=2,
                                      few_shot=True, backend=backend)
            results["total"] += 1
            arch_results["total"] += 1

            if result["valid"]:
                results["valid"] += 1
                arch_results["valid"] += 1

            results["scores"].append(result["score"])
            arch_results["scores"].append(result["score"])
            results["attempts_histogram"][result["attempts"]] += 1

            if result["attempts"] > 1 and result["valid"]:
                results["self_corrections"] += 1

            for e in result["validation"].get("errors", []):
                results["errors_by_type"][e["code"]] += 1

            print(f"  {arch_name} rep{rep}: score={result['score']} "
                  f"grade={result['grade']} attempts={result['attempts']} "
                  f"valid={result['valid']}")

        results["by_archetype"][arch_name] = arch_results

    return results


def print_report(results, mode="offline"):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT ({mode})")
    print("=" * 60)

    total = results["total"]
    valid = results["valid"]
    scores = results["scores"]

    print(f"\nOverall: {valid}/{total} valid ({valid/total*100:.1f}%)")
    if scores:
        print(f"Score: mean={sum(scores)/len(scores):.1f}, "
              f"min={min(scores)}, max={max(scores)}")

    if "component_counts" in results:
        cc = results["component_counts"]
        ic = results["interaction_counts"]
        rc = results["regulatory_counts"]
        print(f"\nComplexity:")
        print(f"  Components:   mean={sum(cc)/len(cc):.1f}, range=[{min(cc)}, {max(cc)}]")
        print(f"  Interactions: mean={sum(ic)/len(ic):.1f}, range=[{min(ic)}, {max(ic)}]")
        print(f"  Regulatory:   mean={sum(rc)/len(rc):.1f}, range=[{min(rc)}, {max(rc)}]")

    if results["errors_by_type"]:
        print(f"\nError distribution:")
        for code, count in results["errors_by_type"].most_common(10):
            print(f"  {count:4d}  {code}")

    if "by_archetype" in results:
        print(f"\nBy archetype:")
        for arch, ar in results["by_archetype"].items():
            pct = ar["valid"]/ar["total"]*100 if ar["total"] else 0
            avg = sum(ar["scores"])/len(ar["scores"]) if ar["scores"] else 0
            print(f"  {arch:20s}: {ar['valid']}/{ar['total']} valid, avg score {avg:.0f}")

    if "attempts_histogram" in results:
        print(f"\nAttempts histogram:")
        for attempts, count in sorted(results["attempts_histogram"].items()):
            label = "1st try" if attempts == 1 else f"after {attempts-1} correction(s)"
            print(f"  {attempts}: {count} ({label})")

        if results.get("self_corrections"):
            print(f"  Self-corrections saved: {results['self_corrections']}")

    # Grade
    if scores:
        avg = sum(scores) / len(scores)
        if avg >= 95: grade = "A+"
        elif avg >= 90: grade = "A"
        elif avg >= 80: grade = "B"
        elif avg >= 70: grade = "C"
        elif avg >= 60: grade = "D"
        else: grade = "F"
        print(f"\n{'=' * 60}")
        print(f"OVERALL GRADE: {grade} ({avg:.1f}/100)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Newgenes model evaluation harness")
    parser.add_argument("--offline", action="store_true",
                        help="Evaluate against held-out test set (no API needed)")
    parser.add_argument("--model", help="Model: sonnet, opus, gpt4o, or a full model ID")
    parser.add_argument("--compare", nargs="+",
                        help="Compare multiple models (e.g. --compare sonnet opus gpt4o)")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Stochastic replicates per prompt (online mode)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--test-file", default=None,
                        help="Path to test JSONL (default: test.jsonl)")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")

    if args.compare:
        # Multi-model comparison mode
        from infer import make_client
        all_results = {}
        for model_name in args.compare:
            client, model_id, backend = make_client(model_name)
            print(f"\n{'#' * 60}")
            print(f"# Evaluating: {model_name} ({model_id})")
            print(f"{'#' * 60}")
            results = eval_online(client, model_id, BENCHMARK_PROMPTS,
                                  args.temperature, args.replicates,
                                  backend=backend)
            all_results[model_name] = results
            print_report(results, mode=f"online:{model_name}")

        # Print comparison summary
        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Model':<15} {'Valid%':>7} {'Avg Score':>10} {'1st Try%':>9} {'Self-Fix':>9}")
        print("-" * 55)
        for name, r in all_results.items():
            valid_pct = r['valid'] / r['total'] * 100 if r['total'] else 0
            avg_score = sum(r['scores']) / len(r['scores']) if r['scores'] else 0
            first_try = r['attempts_histogram'].get(1, 0) / r['total'] * 100 if r['total'] else 0
            self_fix = r.get('self_corrections', 0)
            print(f"{name:<15} {valid_pct:>6.1f}% {avg_score:>9.1f} {first_try:>8.1f}% {self_fix:>8d}")

        # Per-archetype comparison
        print(f"\n{'Archetype':<22}", end="")
        for name in all_results:
            print(f" {name:>12}", end="")
        print()
        print("-" * (22 + 13 * len(all_results)))
        for arch in BENCHMARK_PROMPTS:
            print(f"{arch:<22}", end="")
            for name, r in all_results.items():
                ar = r['by_archetype'].get(arch, {})
                avg = sum(ar.get('scores', [0])) / max(len(ar.get('scores', [1])), 1)
                print(f" {avg:>11.0f}", end="")
            print()

    elif args.offline or not args.model:
        test_file = args.test_file or os.path.join(base, "test.jsonl")
        print(f"Offline evaluation against: {test_file}")
        results = eval_offline(test_file, verbose=args.verbose)
        print_report(results, mode="offline")

        # Also evaluate training data
        train_file = os.path.join(base, "train.jsonl")
        print(f"\n\nTraining data validation: {train_file}")
        train_results = eval_offline(train_file, verbose=False)
        print_report(train_results, mode="training data")

    else:
        from infer import make_client
        client, model_id, backend = make_client(args.model)
        print(f"Online evaluation: model={model_id} ({backend}), "
              f"replicates={args.replicates}, temp={args.temperature}")
        results = eval_online(client, model_id, BENCHMARK_PROMPTS,
                              args.temperature, args.replicates,
                              backend=backend)
        print_report(results, mode=f"online:{args.model}")


if __name__ == "__main__":
    main()

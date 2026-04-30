#!/usr/bin/env python3
"""
Pre-submission acceptance checklist for Newgenes fine-tuning data.

Inspired by Chen & Truong (2026): "instituting an internal acceptance
checklist (unit tests, runtime validation, and user confirmation loops)
transformed the interaction from a one-shot code dump into an iterative
design-build-test cycle"

This script MUST pass before submitting to OpenAI fine-tuning API.
Run: python acceptance_check.py

Checks:
1. Schema compliance (all examples have correct message structure)
2. JSON validity (all assistant responses parse as valid JSON)
3. Validation pass rate (≥95% of circuits pass structural validation)
4. No cross-split leakage (no duplicate descriptions across train/valid/test)
5. System prompt uniformity (all examples use the same system prompt)
6. Component/interaction sanity (reasonable ranges)
7. Architecture diversity (≥5 architecture types represented)
8. Multi-turn format correctness (5-message structure)
9. SBOL3 convertibility (all circuits convert to valid XML)
10. Token budget (estimated fine-tuning cost within budget)
"""
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_circuit import validate
from json_to_sbol3 import circuit_to_sbol3
from topology import analyze_circuit

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check_schema(files):
    """Check 1: All examples have correct message structure."""
    issues = []
    for fname, examples in files.items():
        for i, d in enumerate(examples):
            msgs = d.get('messages', [])
            if len(msgs) not in (3, 5):
                issues.append(f"{fname}[{i}]: {len(msgs)} messages (expected 3 or 5)")
            if msgs[0].get('role') != 'system':
                issues.append(f"{fname}[{i}]: first message not system")
            if msgs[1].get('role') != 'user':
                issues.append(f"{fname}[{i}]: second message not user")
            if msgs[2].get('role') != 'assistant':
                issues.append(f"{fname}[{i}]: third message not assistant")
            if len(msgs) == 5:
                if msgs[3].get('role') != 'user':
                    issues.append(f"{fname}[{i}]: fourth message not user")
                if msgs[4].get('role') != 'assistant':
                    issues.append(f"{fname}[{i}]: fifth message not assistant")

    return (PASS if not issues else FAIL, issues)


def check_json_validity(files):
    """Check 2: All assistant responses parse as valid JSON."""
    issues = []
    for fname, examples in files.items():
        for i, d in enumerate(examples):
            for msg_idx in [2, 4] if len(d['messages']) == 5 else [2]:
                try:
                    json.loads(d['messages'][msg_idx]['content'])
                except (json.JSONDecodeError, IndexError) as e:
                    issues.append(f"{fname}[{i}] msg[{msg_idx}]: {e}")
    return (PASS if not issues else FAIL, issues)


def check_validation_rate(files):
    """Check 3: ≥95% of final circuits pass validation."""
    total = 0
    valid = 0
    issues = []
    for fname, examples in files.items():
        for i, d in enumerate(examples):
            # Use final assistant response
            msg_idx = 4 if len(d['messages']) == 5 else 2
            circuit = json.loads(d['messages'][msg_idx]['content'])
            result = validate(circuit)
            total += 1
            if result['valid']:
                valid += 1
            else:
                errs = [e for e in result['errors'] if e['severity'] == 'error']
                if errs:
                    issues.append(f"{fname}[{i}]: {errs[0]['message'][:60]}")

    rate = valid / total * 100 if total else 0
    status = PASS if rate >= 95 else (WARN if rate >= 80 else FAIL)
    return (status, [f"Validation rate: {valid}/{total} ({rate:.1f}%)"] + issues[:5])


def check_no_leakage(files):
    """Check 4: No duplicate descriptions across splits."""
    descs_by_split = {}
    for fname, examples in files.items():
        descs = set()
        for d in examples:
            descs.add(d['messages'][1]['content'][:200])
        descs_by_split[fname] = descs

    issues = []
    splits = list(descs_by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = descs_by_split[splits[i]] & descs_by_split[splits[j]]
            if overlap:
                issues.append(f"{splits[i]} ∩ {splits[j]}: {len(overlap)} shared descriptions")

    return (PASS if not issues else FAIL, issues)


def check_system_prompt(files):
    """Check 5: All examples use the same system prompt."""
    prompts = set()
    for fname, examples in files.items():
        for d in examples:
            prompts.add(d['messages'][0]['content'][:100])

    if len(prompts) == 1:
        return (PASS, [f"1 unified system prompt"])
    return (FAIL, [f"{len(prompts)} different system prompts found"])


def check_sanity(files):
    """Check 6: Component/interaction counts are reasonable."""
    issues = []
    for fname, examples in files.items():
        for i, d in enumerate(examples):
            msg_idx = 4 if len(d['messages']) == 5 else 2
            circuit = json.loads(d['messages'][msg_idx]['content'])
            nc = len(circuit.get('components', []))
            ni = len(circuit.get('interactions', []))
            if nc == 0:
                issues.append(f"{fname}[{i}]: 0 components")
            if nc > 100:
                issues.append(f"{fname}[{i}]: {nc} components (unusually large)")
            if ni == 0:
                issues.append(f"{fname}[{i}]: 0 interactions")

    return (PASS if not issues else WARN, issues)


def check_diversity(files):
    """Check 7: ≥5 architecture types represented."""
    tags = Counter()
    for examples in files.values():
        for d in examples:
            circuit = json.loads(d['messages'][2]['content'])
            analysis = analyze_circuit(circuit)
            for tag in analysis['architecture_tags']:
                if 'gate at' not in tag and 'regulation at' not in tag:
                    tags[tag] += 1

    n_types = len(tags)
    top = tags.most_common(8)
    info = [f"{n_types} architecture types found"] + [f"  {c:3d}x {t}" for t, c in top]
    return (PASS if n_types >= 5 else FAIL, info)


def check_multiturn(files):
    """Check 8: Multi-turn examples have correct structure."""
    issues = []
    mt_count = 0
    for fname, examples in files.items():
        for i, d in enumerate(examples):
            if len(d['messages']) == 5:
                mt_count += 1
                # Flawed response should be valid JSON
                try:
                    flawed = json.loads(d['messages'][2]['content'])
                except json.JSONDecodeError:
                    issues.append(f"{fname}[{i}]: flawed response not valid JSON")
                    continue
                # Feedback should mention fixing
                feedback = d['messages'][3]['content'].lower()
                if 'fix' not in feedback and 'correct' not in feedback and 'error' not in feedback:
                    issues.append(f"{fname}[{i}]: feedback doesn't mention fixing")

    info = [f"{mt_count} multi-turn examples found"]
    return (PASS if not issues else WARN, info + issues)


def check_sbol3(files):
    """Check 9: All circuits convert to valid SBOL3 XML."""
    issues = []
    total = 0
    for fname, examples in files.items():
        for i, d in enumerate(examples):
            circuit = json.loads(d['messages'][2]['content'])
            total += 1
            try:
                xml = circuit_to_sbol3(circuit, circuit_name=f"test_{i}")
                from xml.etree import ElementTree as ET
                ET.fromstring(xml)
            except Exception as e:
                issues.append(f"{fname}[{i}]: {str(e)[:60]}")

    info = [f"{total - len(issues)}/{total} convert to valid SBOL3 XML"]
    return (PASS if not issues else FAIL, info + issues[:5])


def check_token_budget(files):
    """Check 10: Estimated fine-tuning cost."""
    total_tokens = 0
    for fname, examples in files.items():
        for d in examples:
            # Rough estimate: 1 token ≈ 4 chars
            chars = sum(len(m['content']) for m in d['messages'])
            total_tokens += chars // 4

    n_train = len(files.get('train.jsonl', []))
    epochs = 3
    cost_per_1k = 0.008  # GPT-4o fine-tuning cost per 1K tokens
    estimated_cost = total_tokens / 1000 * cost_per_1k * epochs

    info = [
        f"Training examples: {n_train}",
        f"Estimated tokens: ~{total_tokens:,}",
        f"Estimated cost (3 epochs): ~${estimated_cost:.2f}",
    ]
    return (PASS if estimated_cost < 50 else WARN, info)


def main():
    # Load all files
    files = {}
    for fname in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
        filepath = os.path.join(BASE, fname)
        with open(filepath) as f:
            files[fname] = [json.loads(l) for l in f]

    print("=" * 60)
    print("NEWGENES PRE-SUBMISSION ACCEPTANCE CHECKLIST")
    print("=" * 60)

    checks = [
        ("1. Schema compliance", check_schema),
        ("2. JSON validity", check_json_validity),
        ("3. Validation pass rate", check_validation_rate),
        ("4. No cross-split leakage", check_no_leakage),
        ("5. System prompt uniformity", check_system_prompt),
        ("6. Component/interaction sanity", check_sanity),
        ("7. Architecture diversity", check_diversity),
        ("8. Multi-turn format", check_multiturn),
        ("9. SBOL3 convertibility", check_sbol3),
        ("10. Token budget", check_token_budget),
    ]

    results = []
    for name, check_fn in checks:
        status, details = check_fn(files)
        results.append((name, status, details))

        icon = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
        print(f"\n[{icon}] {name}: {status}")
        for detail in details[:3]:
            print(f"    {detail}")
        if len(details) > 3:
            print(f"    ... and {len(details) - 3} more")

    # Summary
    passes = sum(1 for _, s, _ in results if s == PASS)
    fails = sum(1 for _, s, _ in results if s == FAIL)
    warns = sum(1 for _, s, _ in results if s == WARN)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passes} PASS, {warns} WARN, {fails} FAIL")

    if fails == 0:
        print("STATUS: READY FOR SUBMISSION")
    elif fails <= 2 and warns <= 3:
        print("STATUS: REVIEW WARNINGS BEFORE SUBMISSION")
    else:
        print("STATUS: NOT READY — FIX FAILURES BEFORE SUBMISSION")
    print("=" * 60)

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

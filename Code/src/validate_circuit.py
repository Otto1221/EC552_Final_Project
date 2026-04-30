#!/usr/bin/env python3
"""
Post-inference circuit validator for Newgenes fine-tuned model.

Inspired by Chen & Truong (2026) three-phase workflow:
  1. Generate circuit JSON from natural language
  2. Validate against structural/biological rules
  3. If errors found, feed back for self-correction

Usage:
  # Validate a single JSON string
  python validate_circuit.py '{"components":[...],"interactions":[...]}'

  # Validate from file
  python validate_circuit.py --file output.json

  # Use as a module in inference pipeline
  from validate_circuit import validate, format_feedback

  result = validate(circuit_dict)
  if not result["valid"]:
      feedback = format_feedback(result)
      # Feed 'feedback' back to the model as a follow-up user message
"""
import json
import sys
import argparse

VALID_COMPONENT_TYPES = {"promoter", "rbs", "cds", "terminator", "operator", "other"}
VALID_INTERACTION_TYPES = {"transcription", "translation", "activation", "repression"}

# Valid source→target type constraints for interactions
INTERACTION_CONSTRAINTS = {
    "transcription": {"from": {"promoter"}, "to": {"cds"}},
    "translation":   {"from": {"rbs"},      "to": {"cds"}},
    "activation":    {"from": {"cds"},      "to": {"promoter", "operator"}},
    "repression":    {"from": {"cds"},      "to": {"promoter", "operator"}},
}


def validate(circuit: dict) -> dict:
    """
    Validate a circuit JSON object. Returns:
    {
      "valid": bool,
      "errors": [{"code": str, "severity": "error"|"warning", "message": str}],
      "stats": {"components": int, "interactions": int, ...}
    }
    """
    errors = []
    stats = {}

    # --- Schema checks ---
    if not isinstance(circuit, dict):
        return {"valid": False, "errors": [{"code": "SCHEMA_ROOT", "severity": "error",
                "message": "Root must be a JSON object"}], "stats": {}}

    if "components" not in circuit:
        errors.append({"code": "SCHEMA_COMPONENTS", "severity": "error",
                       "message": "Missing 'components' array"})
    if "interactions" not in circuit:
        errors.append({"code": "SCHEMA_INTERACTIONS", "severity": "error",
                       "message": "Missing 'interactions' array"})

    if errors:
        return {"valid": False, "errors": errors, "stats": {}}

    components = circuit["components"]
    interactions = circuit["interactions"]

    if not isinstance(components, list):
        errors.append({"code": "SCHEMA_COMP_TYPE", "severity": "error",
                       "message": "'components' must be an array"})
        return {"valid": False, "errors": errors, "stats": {}}

    if not isinstance(interactions, list):
        errors.append({"code": "SCHEMA_INT_TYPE", "severity": "error",
                       "message": "'interactions' must be an array"})
        return {"valid": False, "errors": errors, "stats": {}}

    # Build lookup — components use 'name' as the unique identifier
    comp_names = set()
    comp_by_name = {}
    type_counts = {}

    for i, c in enumerate(components):
        # Required fields: name and type
        for field in ["name", "type"]:
            if field not in c:
                errors.append({"code": "COMP_FIELD", "severity": "error",
                               "message": f"Component [{i}] missing '{field}' field"})

        cname = c.get("name", f"__missing_{i}")
        ctype = c.get("type", "unknown")

        # Duplicate name
        if cname in comp_names:
            errors.append({"code": "COMP_DUP_NAME", "severity": "error",
                           "message": f"Duplicate component name: '{cname}'"})
        comp_names.add(cname)
        comp_by_name[cname] = c

        # Valid type
        if ctype not in VALID_COMPONENT_TYPES:
            errors.append({"code": "COMP_TYPE", "severity": "error",
                           "message": f"Component '{cname}' has invalid type '{ctype}'. Valid: {VALID_COMPONENT_TYPES}"})

        type_counts[ctype] = type_counts.get(ctype, 0) + 1

    stats["components"] = len(components)
    stats["type_counts"] = type_counts

    # --- Interaction checks ---
    int_type_counts = {}
    for i, ix in enumerate(interactions):
        for field in ["type", "from", "to"]:
            if field not in ix:
                errors.append({"code": "INT_FIELD", "severity": "error",
                               "message": f"Interaction [{i}] missing '{field}' field"})

        itype = ix.get("type", "unknown")
        ifrom = ix.get("from", "")
        ito = ix.get("to", "")

        # Valid type
        if itype not in VALID_INTERACTION_TYPES:
            errors.append({"code": "INT_TYPE", "severity": "error",
                           "message": f"Interaction [{i}] has invalid type '{itype}'. Valid: {VALID_INTERACTION_TYPES}"})

        # Orphan reference check
        if ifrom and ifrom not in comp_names:
            errors.append({"code": "INT_ORPHAN", "severity": "error",
                           "message": f"Interaction [{i}] 'from' references non-existent component '{ifrom}'"})

        if ito and ito not in comp_names:
            errors.append({"code": "INT_ORPHAN", "severity": "error",
                           "message": f"Interaction [{i}] 'to' references non-existent component '{ito}'"})

        # Type constraint check (from/to component types must match)
        if itype in INTERACTION_CONSTRAINTS and ifrom in comp_by_name and ito in comp_by_name:
            constraint = INTERACTION_CONSTRAINTS[itype]
            from_type = comp_by_name[ifrom].get("type", "")
            to_type = comp_by_name[ito].get("type", "")

            if from_type not in constraint["from"]:
                errors.append({"code": "INT_FROM_TYPE", "severity": "warning",
                               "message": f"Interaction [{i}] ({itype}): 'from' component '{ifrom}' is type '{from_type}', expected {constraint['from']}"})

            if to_type not in constraint["to"]:
                errors.append({"code": "INT_TO_TYPE", "severity": "warning",
                               "message": f"Interaction [{i}] ({itype}): 'to' component '{ito}' is type '{to_type}', expected {constraint['to']}"})

        # Self-reference
        if ifrom and ito and ifrom == ito:
            errors.append({"code": "INT_SELF_REF", "severity": "error",
                           "message": f"Interaction [{i}] is a self-reference: '{ifrom}' → '{ito}'"})

        int_type_counts[itype] = int_type_counts.get(itype, 0) + 1

    stats["interactions"] = len(interactions)
    stats["interaction_types"] = int_type_counts

    # --- Biological completeness checks ---
    # Every CDS should have at least one transcription and one translation interaction
    cds_ids = {cname for cname, c in comp_by_name.items() if c.get("type") == "cds"}
    transcribed = set()
    translated = set()
    for ix in interactions:
        if ix.get("type") == "transcription" and ix.get("to") in cds_ids:
            transcribed.add(ix["to"])
        if ix.get("type") == "translation" and ix.get("to") in cds_ids:
            translated.add(ix["to"])

    for cid in cds_ids:
        if cid not in transcribed:
            errors.append({"code": "BIO_NO_TRANSCRIPTION", "severity": "warning",
                           "message": f"CDS '{cid}' has no transcription interaction (no promoter driving it)"})
        if cid not in translated:
            errors.append({"code": "BIO_NO_TRANSLATION", "severity": "warning",
                           "message": f"CDS '{cid}' has no translation interaction (no RBS upstream)"})

    # Every promoter should drive at least one CDS
    promoter_ids = {cname for cname, c in comp_by_name.items() if c.get("type") == "promoter"}
    driving_promoters = {ix.get("from") for ix in interactions if ix.get("type") == "transcription"}
    for pid in promoter_ids:
        if pid not in driving_promoters:
            errors.append({"code": "BIO_ORPHAN_PROMOTER", "severity": "warning",
                           "message": f"Promoter '{pid}' does not drive any CDS via transcription"})

    # Every RBS should translate at least one CDS
    rbs_ids = {cname for cname, c in comp_by_name.items() if c.get("type") == "rbs"}
    translating_rbs = {ix.get("from") for ix in interactions if ix.get("type") == "translation"}
    for rid in rbs_ids:
        if rid not in translating_rbs:
            errors.append({"code": "BIO_ORPHAN_RBS", "severity": "warning",
                           "message": f"RBS '{rid}' does not translate any CDS"})

    # Minimum complexity check
    if len(components) < 3:
        errors.append({"code": "BIO_TOO_SIMPLE", "severity": "warning",
                       "message": f"Circuit has only {len(components)} components — likely incomplete"})
    if len(interactions) < 2:
        errors.append({"code": "BIO_FEW_INTERACTIONS", "severity": "warning",
                       "message": f"Circuit has only {len(interactions)} interactions — likely missing wiring"})

    # CDS-to-CDS direct regulation (should go through promoter/operator)
    for i, ix in enumerate(interactions):
        if ix.get("type") in ("activation", "repression"):
            to_type = comp_by_name.get(ix.get("to", ""), {}).get("type", "")
            if to_type == "cds":
                errors.append({"code": "BIO_CDS_TO_CDS", "severity": "error",
                               "message": f"Interaction [{i}]: {ix['type']} targets CDS '{ix['to']}' directly — should target a promoter or operator"})

    # Compute severity
    error_count = sum(1 for e in errors if e["severity"] == "error")
    warning_count = sum(1 for e in errors if e["severity"] == "warning")
    stats["error_count"] = error_count
    stats["warning_count"] = warning_count

    return {
        "valid": error_count == 0,
        "errors": errors,
        "stats": stats
    }


def diagnose_failure_mode(result: dict) -> str:
    """
    Provide mechanistic biological diagnosis of what's wrong — not just
    error codes, but WHY the circuit is biologically invalid and HOW to fix it.

    Inspired by Chen & Truong (2026) who diagnosed that premature reciprocal
    signaling caused spheroid fragmentation, then prescribed a specific fix.
    """
    errors = result.get("errors", [])
    if not errors:
        return ""

    error_codes = [e["code"] for e in errors]
    diagnoses = []

    # Pattern: CDS-to-CDS regulation
    cds_to_cds = [e for e in errors if e["code"] == "BIO_CDS_TO_CDS"]
    if cds_to_cds:
        diagnoses.append(
            "DIAGNOSIS — Direct protein-to-protein regulation:\n"
            "  The circuit models regulation as one protein directly controlling another protein. "
            "In real biology, transcription factors regulate gene expression by binding to DNA "
            "(promoters or operator sites), NOT by directly modifying other proteins. "
            "A repressor like LacI binds the lacO operator in the pLac promoter region to "
            "block RNA polymerase — it does not 'repress' another CDS.\n"
            "  FIX: Change each regulatory interaction's target from the CDS to its upstream "
            "promoter or operator. For example, if LacI represses GFP, the interaction should be "
            "laci_cds → plac_promoter (repression), NOT laci_cds → gfp_cds."
        )

    # Pattern: Missing transcription/translation
    no_txn = [e for e in errors if e["code"] == "BIO_NO_TRANSCRIPTION"]
    no_tln = [e for e in errors if e["code"] == "BIO_NO_TRANSLATION"]
    if no_txn or no_tln:
        diagnoses.append(
            "DIAGNOSIS — Incomplete transcription units:\n"
            "  Some coding sequences lack the molecular machinery needed for expression. "
            "In real biology, a gene requires: (1) a promoter to recruit RNA polymerase and "
            "initiate transcription, (2) a ribosome binding site (RBS/Shine-Dalgarno) for "
            "translation initiation, and (3) a terminator to release the transcript. "
            "Without these, the protein product cannot be produced.\n"
            "  FIX: For each CDS missing transcription, add a promoter component and a "
            "'transcription' interaction (promoter → cds). For each CDS missing translation, "
            "add an RBS component and a 'translation' interaction (rbs → cds)."
        )

    # Pattern: Orphan references
    orphans = [e for e in errors if e["code"] == "INT_ORPHAN"]
    if orphans:
        orphan_names = set()
        for e in orphans:
            # Extract the component name from the message
            import re
            m = re.search(r"'([^']+)'$", e["message"])
            if m:
                orphan_names.add(m.group(1))
        diagnoses.append(
            f"DIAGNOSIS — Broken wiring (orphan references):\n"
            f"  Interactions reference components that don't exist: {orphan_names}. "
            "This is like drawing a wire to a part that isn't on the breadboard — "
            "the circuit cannot function.\n"
            "  FIX: Either add the missing components to the components array, or "
            "correct the interaction references to match existing component names exactly."
        )

    # Pattern: Duplicate names
    dup_names = [e for e in errors if e["code"] == "BIO_DUP_NAME"]
    if dup_names:
        diagnoses.append(
            "DIAGNOSIS — Ambiguous component identity:\n"
            "  Multiple components share the same name, making it impossible to determine "
            "which one an interaction refers to. Each biological part must be uniquely "
            "identifiable.\n"
            "  FIX: Give each component a unique snake_case name. If you have two copies "
            "of the same part, suffix them: e.g., gfp_cds_1 and gfp_cds_2."
        )

    # Pattern: Orphan promoters/RBS
    orphan_p = [e for e in errors if e["code"] == "BIO_ORPHAN_PROMOTER"]
    if orphan_p:
        diagnoses.append(
            "DIAGNOSIS — Promoters not driving any gene:\n"
            "  Some promoters exist in the circuit but have no transcription interaction — "
            "they're like open reading frames with no start signal. A promoter without a "
            "downstream gene serves no purpose.\n"
            "  FIX: Add a 'transcription' interaction from each orphan promoter to the CDS "
            "it is intended to drive."
        )

    # Pattern: Too simple
    if "BIO_TOO_SIMPLE" in error_codes or "BIO_FEW_INTERACTIONS" in error_codes:
        diagnoses.append(
            "DIAGNOSIS — Circuit too simple:\n"
            "  The circuit has very few components or interactions, suggesting it's incomplete. "
            "Even simple genetic circuits like a constitutive reporter need at least a promoter, "
            "RBS, CDS, and terminator with transcription and translation interactions.\n"
            "  FIX: Add all implied components even if not explicitly named in the description."
        )

    return "\n\n".join(diagnoses)


def format_feedback(result: dict) -> str:
    """
    Format validation errors into a feedback message suitable for
    feeding back to the model as a follow-up user message for self-correction.

    Combines error listing with mechanistic biological diagnosis.
    """
    if result["valid"] and not result["errors"]:
        return ""

    lines = ["The circuit JSON has the following issues that need fixing:\n"]

    # Errors first
    errs = [e for e in result["errors"] if e["severity"] == "error"]
    warns = [e for e in result["errors"] if e["severity"] == "warning"]

    if errs:
        lines.append("ERRORS (must fix):")
        for e in errs:
            lines.append(f"  - [{e['code']}] {e['message']}")

    if warns:
        lines.append("\nWARNINGS (should fix):")
        for e in warns[:5]:  # Cap warnings to avoid token bloat
            lines.append(f"  - [{e['code']}] {e['message']}")
        if len(warns) > 5:
            lines.append(f"  ... and {len(warns) - 5} more warnings")

    # Add biological diagnosis
    diagnosis = diagnose_failure_mode(result)
    if diagnosis:
        lines.append(f"\n{diagnosis}")

    lines.append("\nPlease regenerate the circuit JSON with these issues corrected.")
    return "\n".join(lines)


def score_circuit(result: dict) -> tuple:
    """Return (score 0-100, grade letter) for a validation result."""
    if not result["valid"]:
        base = max(0, 50 - result["stats"].get("error_count", 0) * 10)
    else:
        base = 100 - result["stats"].get("warning_count", 0) * 5
    score = max(0, min(100, base))
    if score >= 95: grade = "A+"
    elif score >= 90: grade = "A"
    elif score >= 80: grade = "B"
    elif score >= 70: grade = "C"
    elif score >= 60: grade = "D"
    else: grade = "F"
    return score, grade


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Newgenes circuit JSON")
    parser.add_argument("json_str", nargs="?", help="Circuit JSON string")
    parser.add_argument("--file", "-f", help="Read circuit from JSON file")
    parser.add_argument("--jsonl", help="Validate all circuits in a JSONL training file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.jsonl:
        # Batch validation mode for training data
        total = 0; valid = 0; all_errors = 0; all_warnings = 0
        with open(args.jsonl) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                circuit = json.loads(d["messages"][2]["content"])
                result = validate(circuit)
                total += 1
                if result["valid"]:
                    valid += 1
                all_errors += result["stats"].get("error_count", 0)
                all_warnings += result["stats"].get("warning_count", 0)
                if args.verbose and not result["valid"]:
                    print(f"\n--- Example {i} ---")
                    print(f"User: {d['messages'][1]['content'][:80]}...")
                    for e in result["errors"]:
                        if e["severity"] == "error":
                            print(f"  ERROR: {e['message']}")

        pct = valid/total*100 if total else 0
        print(f"\n{'='*50}")
        print(f"Validated {total} circuits")
        print(f"  Valid:    {valid}/{total} ({pct:.1f}%)")
        print(f"  Errors:   {all_errors}")
        print(f"  Warnings: {all_warnings}")
    else:
        # Single circuit validation
        if args.file:
            with open(args.file) as f:
                circuit = json.load(f)
        elif args.json_str:
            circuit = json.loads(args.json_str)
        else:
            print("Reading from stdin...")
            circuit = json.load(sys.stdin)

        result = validate(circuit)
        score, grade = score_circuit(result)

        print(f"Valid: {result['valid']}")
        print(f"Score: {score}/100 ({grade})")
        print(f"Components: {result['stats'].get('components', 0)}")
        print(f"Interactions: {result['stats'].get('interactions', 0)}")

        if result["errors"]:
            print(f"\nIssues ({result['stats']['error_count']} errors, {result['stats']['warning_count']} warnings):")
            for e in result["errors"]:
                marker = "ERROR" if e["severity"] == "error" else "WARN "
                print(f"  [{marker}] {e['message']}")

            print(f"\n--- Self-correction feedback ---")
            print(format_feedback(result))

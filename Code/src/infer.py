#!/usr/bin/env python3
"""
Newgenes inference pipeline with self-correction loop.

Inspired by Chen & Truong (2026) three-phase workflow:
  Phase 1: Natural language → circuit JSON (model generation)
  Phase 2: Validate against structural/biological rules
  Phase 3: If errors, feed back for self-correction (up to max_retries)

Usage:
  # Interactive mode
  python infer.py

  # Single query
  python infer.py "Design a toggle switch with LacI and TetR"

  # With custom model
  python infer.py --model ft:gpt-4o-2024-08-06:personal::XXXX "Design a repressilator"

Requires: pip install openai
"""
import json
import sys
import argparse
import os
from validate_circuit import validate, format_feedback, score_circuit

# Supported backends
BACKENDS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "gpt4o": "gpt-4o-2024-08-06",
}


def make_client(model: str):
    """
    Create the appropriate API client based on model string.
    Returns (client, model_id, backend_type).
    """
    # Check for aliases first
    if model in BACKENDS:
        model = BACKENDS[model]

    if model.startswith("claude-"):
        import anthropic
        return anthropic.Anthropic(), model, "anthropic"
    else:
        from openai import OpenAI
        return OpenAI(), model, "openai"


def call_model(client, model: str, messages: list, backend: str,
               temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Unified call interface for OpenAI and Anthropic backends."""
    if backend == "anthropic":
        # Anthropic uses system param separately, not in messages
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_msgs.append(m)

        response = client.messages.create(
            model=model,
            system=system_msg or "",
            messages=chat_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text.strip()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

# The same enhanced system prompt used in fine-tuning
SYSTEM_PROMPT = """You are an expert synthetic biology assistant that converts natural language descriptions of genetic circuits into structured JSON representations following SBOL3 (Synthetic Biology Open Language) conventions.

## Output Format
Return a JSON object with exactly two arrays:
{
  "components": [ { "name": "string (unique snake_case identifier)", "type": "string" } ],
  "interactions": [ { "type": "string", "from": "component_name", "to": "component_name" } ]
}

## Component Type Reference (SBOL3 / Sequence Ontology)
| Type       | SO Role      | Description                                      | Examples                                    |
|------------|-------------|--------------------------------------------------|---------------------------------------------|
| promoter   | SO:0000167  | DNA region initiating transcription              | pLac, pTet, pBAD, CMV, EF1a, T7, pCAG      |
| rbs        | SO:0000139  | Ribosome binding site enabling translation       | B0034, Kozak, strong/weak RBS               |
| cds        | SO:0000316  | Coding sequence producing a protein              | GFP, LacI, TetR, Cas9, CAR, scFv, cytokine |
| terminator | SO:0000141  | Transcription termination signal                 | B0015, rrnB, SV40 pA, BGH pA               |
| operator   | SO:0000057  | Protein binding site mediating regulation        | lacO, tetO, araO, lambda OL/OR              |
| other      | —           | All other functional elements                    | degradation tags (ssrA), signal peptides, linkers, 2A peptides (T2A/P2A), ITRs, insulators, loxP/FRT sites, riboswitches, aptamers, IRES, scaffold proteins, split protein domains |

## Interaction Type Reference
| Type          | Valid from → to                    | Biological meaning                                    |
|---------------|-------------------------------------|-------------------------------------------------------|
| transcription | promoter → cds                     | Promoter drives transcription of a coding sequence    |
| translation   | rbs → cds                          | RBS enables ribosomal translation of a coding sequence|
| activation    | cds → promoter or cds → operator   | Protein product activates transcription at target      |
| repression    | cds → promoter or cds → operator   | Protein product represses transcription at target      |

## Structural Rules (Guardrails)
1. Every transcription unit MUST contain: promoter → RBS → CDS → terminator
2. Every CDS MUST have exactly one transcription interaction (from its promoter) and one translation interaction (from its RBS)
3. Regulatory interactions (activation/repression) go from a CDS (the protein source) TO a promoter or operator (the DNA target) — NEVER from CDS to CDS directly
4. Every interaction must reference component names that exist in the components array — no orphan references
5. Use descriptive snake_case names: e.g., plac_promoter, laci_cds, gfp_reporter, b0034_rbs
6. Include ALL components mentioned or implied by the description — do not omit RBS/terminator even if not explicitly named
7. For multi-module circuits, represent each transcription unit completely with its own promoter-RBS-CDS-terminator stack
8. For feedback loops and regulatory cascades, trace the full chain: CDS protein → target promoter/operator → downstream CDS
9. Operators should be included when the description specifies binding sites; otherwise, regulatory interactions can target promoters directly
10. Use "other" type for any functional element that does not fit the five standard types

Respond with valid JSON only. No explanation, no markdown fences, no commentary."""


def generate_circuit(client, model: str, description: str, max_retries: int = 2,
                     few_shot: bool = True, backend: str = "openai") -> dict:
    """
    Generate a circuit from description with validation and self-correction.

    When few_shot=True, injects relevant exemplar circuits as context
    (analogous to Chen & Truong embedding CC3D source code for ground-truth).

    Returns:
      {
        "circuit": dict,         # The final circuit JSON
        "valid": bool,           # Whether it passed validation
        "score": int,            # 0-100 quality score
        "grade": str,            # Letter grade
        "attempts": int,         # How many generation attempts
        "validation": dict       # Full validation result
      }
    """
    # Inject SBOL3 reference context (analogous to paper embedding CC3D source code)
    user_content = description
    try:
        from sbol3_reference import get_compact_sbol3_context
        sbol3_ctx = get_compact_sbol3_context()
        user_content = sbol3_ctx + "\n\n---\n\n" + description
    except ImportError:
        pass

    # Optionally inject few-shot exemplar context
    if few_shot:
        try:
            from exemplar_bank import select_relevant_exemplars, format_few_shot_context
            bank_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'exemplar_bank.json')
            with open(bank_path) as f:
                bank = json.load(f)
            selected = select_relevant_exemplars(description, bank, n=2)
            context = format_few_shot_context(bank, selected)
            user_content = user_content + "\n\n" + context
        except (ImportError, FileNotFoundError):
            pass  # Fall back to no few-shot if bank not available

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    for attempt in range(1, max_retries + 2):  # +2 because range is exclusive and attempt 1 is the initial
        raw = call_model(client, model, messages, backend)

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            circuit = json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt <= max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"Your response was not valid JSON. Error: {e}. "
                    "Please output ONLY valid JSON with no explanation."})
                continue
            return {
                "circuit": None,
                "valid": False, "score": 0, "grade": "F",
                "attempts": attempt,
                "validation": {"valid": False, "errors": [
                    {"code": "JSON_PARSE", "severity": "error",
                     "message": f"Failed to parse JSON: {e}"}
                ], "stats": {}}
            }

        # Validate
        result = validate(circuit)
        score, grade = score_circuit(result)

        # If valid or out of retries, return
        if result["valid"] or attempt > max_retries:
            return {
                "circuit": circuit,
                "valid": result["valid"],
                "score": score, "grade": grade,
                "attempts": attempt,
                "validation": result
            }

        # Self-correction: feed validation errors back
        feedback = format_feedback(result)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})

    # Should not reach here
    return {"circuit": circuit, "valid": False, "score": score,
            "grade": grade, "attempts": max_retries + 1, "validation": result}


def main():
    parser = argparse.ArgumentParser(description="Newgenes circuit inference with self-correction")
    parser.add_argument("description", nargs="?", help="Circuit description")
    parser.add_argument("--model", default="sonnet",
                        help="Model: sonnet, opus, gpt4o, or a full model ID")
    parser.add_argument("--retries", type=int, default=2, help="Max self-correction retries")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot exemplar injection")
    args = parser.parse_args()

    client, model_id, backend = make_client(args.model)
    print(f"Backend: {backend} | Model: {model_id}")

    if args.interactive or not args.description:
        print("Newgenes Circuit Designer (type 'quit' to exit)")
        print("=" * 50)
        while True:
            try:
                desc = input("\nDescribe your circuit: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if desc.lower() in ("quit", "exit", "q"):
                break
            if not desc:
                continue

            result = generate_circuit(client, model_id, desc, args.retries,
                                      few_shot=not args.no_few_shot,
                                      backend=backend)
            print(f"\n{'='*50}")
            print(f"Score: {result['score']}/100 ({result['grade']}) | "
                  f"Attempts: {result['attempts']} | Valid: {result['valid']}")

            if result["circuit"]:
                print(json.dumps(result["circuit"], indent=2))

            if result["validation"].get("errors"):
                errs = [e for e in result["validation"]["errors"] if e["severity"] == "error"]
                warns = [e for e in result["validation"]["errors"] if e["severity"] == "warning"]
                if errs:
                    print(f"\nRemaining errors ({len(errs)}):")
                    for e in errs[:5]:
                        print(f"  - {e['message']}")
                if warns:
                    print(f"\nWarnings ({len(warns)}):")
                    for w in warns[:5]:
                        print(f"  - {w['message']}")
    else:
        result = generate_circuit(client, model_id, args.description, args.retries,
                                  few_shot=not args.no_few_shot, backend=backend)
        output = {
            "model": model_id,
            "backend": backend,
            "valid": result["valid"],
            "score": result["score"],
            "grade": result["grade"],
            "attempts": result["attempts"],
            "circuit": result["circuit"],
            "errors": [e for e in result["validation"].get("errors", []) if e["severity"] == "error"],
            "warnings": [e for e in result["validation"].get("errors", []) if e["severity"] == "warning"]
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

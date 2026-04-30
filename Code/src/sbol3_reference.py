#!/usr/bin/env python3
"""
SBOL3 reference context for inference-time injection.

Analogous to Chen & Truong (2026) embedding CC3D source code alongside
the reference manual. This provides "ground-truth implementation details"
that resolve ambiguities and ensure correct output format.

Usage in infer.py:
    from sbol3_reference import get_sbol3_context
    context = get_sbol3_context(description)
    # Prepend to user message at inference time
"""
import json
import os

# Full SBOL3 namespace reference for the model
SBOL3_NAMESPACES = """## SBOL3 XML Namespaces (for reference)
- sbol3: http://sbols.org/v3#
- rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#
- dcterms: http://purl.org/dc/terms/
- prov: http://www.w3.org/ns/prov#
- SO: http://identifiers.org/SO:  (Sequence Ontology)
- SBO: http://identifiers.org/biomodels.sbo/SBO:  (Systems Biology Ontology)"""

# Complete SO role mapping with full URIs
SO_ROLE_REFERENCE = """## Sequence Ontology Roles (complete URI mapping)
promoter   → http://identifiers.org/SO:0000167  (a regulatory_region that is a transcriptional_cis_regulatory_region)
rbs        → http://identifiers.org/SO:0000139  (ribosome_entry_site — region on mRNA for ribosome binding)
cds        → http://identifiers.org/SO:0000316  (CDS — contiguous sequence translated into protein)
terminator → http://identifiers.org/SO:0000141  (terminator — sequence marking end of transcript)
operator   → http://identifiers.org/SO:0000057  (operator — DNA region where repressor/activator binds)
other      → http://identifiers.org/SO:0000110  (sequence_feature — generic functional element)"""

# Complete SBO interaction mapping
SBO_INTERACTION_REFERENCE = """## SBO Interaction Types (complete URI mapping)
transcription → SBO:0000589 (genetic_production — promoter template produces mRNA/protein)
  Participants: template (SBO:0000645) = promoter, product (SBO:0000011) = CDS

translation   → SBO:0000184 (genetic_production — RBS template directs protein synthesis)
  Participants: template (SBO:0000645) = RBS, product (SBO:0000011) = CDS

activation    → SBO:0000170 (stimulation — protein enhances transcription rate)
  Participants: stimulator (SBO:0000459) = CDS, stimulated (SBO:0000643) = promoter/operator

repression    → SBO:0000169 (inhibition — protein reduces transcription rate)
  Participants: inhibitor (SBO:0000020) = CDS, inhibited (SBO:0000642) = promoter/operator"""

# SBOL3 XML structure template
SBOL3_XML_STRUCTURE = """## SBOL3 XML Structure (your JSON maps to this)

<rdf:RDF xmlns:sbol3="http://sbols.org/v3#" xmlns:rdf="...">
  <!-- Top-level Component = your entire circuit -->
  <sbol3:Component rdf:about="https://newgenes.org/{circuit_name}">
    <sbol3:type rdf:resource="SBO:0000251"/>     <!-- DNA molecule -->
    <sbol3:role rdf:resource="SO:0000804"/>       <!-- engineered_region -->

    <!-- Each component in your JSON becomes a SubComponent -->
    <sbol3:hasFeature>
      <sbol3:SubComponent>
        <sbol3:role rdf:resource="{SO_TERM}"/>    <!-- from type mapping -->
        <sbol3:instanceOf rdf:resource="{part_definition_uri}"/>
      </sbol3:SubComponent>
    </sbol3:hasFeature>

    <!-- Each interaction becomes an Interaction with Participations -->
    <sbol3:hasInteraction>
      <sbol3:Interaction>
        <sbol3:type rdf:resource="{SBO_TERM}"/>   <!-- from type mapping -->
        <sbol3:hasParticipation>
          <sbol3:Participation>
            <sbol3:role rdf:resource="{role}"/>    <!-- stimulator/inhibitor/template/product -->
            <sbol3:participant rdf:resource="{component_uri}"/>
          </sbol3:Participation>
        </sbol3:hasParticipation>
      </sbol3:Interaction>
    </sbol3:hasInteraction>
  </sbol3:Component>
</rdf:RDF>"""

# Common biological knowledge grounding
BIOLOGY_REFERENCE = """## Key Biological Principles for Circuit Modeling

### Transcription Unit Architecture
DNA: 5'—[promoter]—[operator]—[RBS]—[CDS]—[terminator]—3'
     ↓ RNA polymerase binds         ↓ ribosome binds    ↓ transcription stops
     ↓ transcription starts         ↓ translation starts

### Regulatory Mechanisms (how proteins control gene expression)
1. Repression: repressor protein (e.g., LacI) binds operator DNA (e.g., lacO) → blocks RNAP → no mRNA
   - IPTG binds LacI → releases from lacO → transcription ON (induction)
   - TetR binds tetO → blocks pTet → aTc releases TetR (induction)

2. Activation: activator protein binds upstream DNA → recruits RNAP → enhanced transcription
   - AraC + arabinose → binds pBAD → activation
   - LuxR + AHL → binds pLux → activation

3. KEY RULE: Regulation acts at the DNA level.
   A repressor BINDS to a PROMOTER/OPERATOR (DNA). It does NOT directly "repress" another protein.
   Therefore: repression/activation interactions go from CDS → promoter/operator, NEVER CDS → CDS.

### Common Circuit Motifs
- Inverter (NOT gate): constitutive repressor → represses target promoter → output is inverse of input
- Toggle switch: two repressors mutually repress each other's promoters → bistable memory
- Repressilator: 3+ repressors in a ring → each represses the next → oscillation
- AND gate: output requires two inputs (e.g., split T7 RNAP halves from two inducible promoters)
- Positive feedback: activator enhances its own promoter → switch-like behavior
- Negative feedback: protein represses its own promoter → homeostasis"""


def get_sbol3_context(description=None):
    """
    Build full SBOL3 reference context for injection at inference time.

    This is analogous to the paper providing CC3D source code + manual
    to the LLM at every generation step.
    """
    sections = [
        "# SBOL3 Reference Context",
        "",
        SBOL3_NAMESPACES,
        "",
        SO_ROLE_REFERENCE,
        "",
        SBO_INTERACTION_REFERENCE,
        "",
        SBOL3_XML_STRUCTURE,
        "",
        BIOLOGY_REFERENCE,
    ]

    return "\n".join(sections)


def get_compact_sbol3_context():
    """Shorter version for token-constrained scenarios."""
    return f"""{SO_ROLE_REFERENCE}

{SBO_INTERACTION_REFERENCE}

KEY RULE: Regulation (activation/repression) ALWAYS targets promoter or operator DNA.
NEVER target a CDS directly — transcription factors bind DNA, not proteins."""


if __name__ == '__main__':
    full = get_sbol3_context()
    compact = get_compact_sbol3_context()
    print(f"Full SBOL3 context: {len(full)} chars, ~{len(full)//4} tokens")
    print(f"Compact context: {len(compact)} chars, ~{len(compact)//4} tokens")
    print()
    print("--- Full context ---")
    print(full)

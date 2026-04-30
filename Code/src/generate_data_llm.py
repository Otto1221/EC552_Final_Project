"""
Two-stage training data generator for Newgenes fine-tuning.

Stage 1: Qwen2.5-72B (local, free) generates simple/medium circuit examples
Stage 2: GPT-5.4 (API, paid) generates complex circuit examples

Outputs train.jsonl, valid.jsonl, test.jsonl for mlx-lm LoRA.
"""

import json
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Circuit description prompts — the LLM generates the JSON output for each
# ---------------------------------------------------------------------------

SIMPLE_DESCRIPTIONS = [
    # Constitutive expression
    "A constitutive mCherry expression circuit with a J23100 promoter, B0034 RBS, mCherry coding sequence, and B0015 terminator.",
    "A constitutive luciferase expression cassette driven by the Anderson promoter J23119.",
    "A constitutive BFP expression unit with a strong RBS and rrnB terminator.",
    "A simple beta-galactosidase reporter circuit under a constitutive promoter.",
    "A constitutive expression circuit for an antibiotic resistance gene (KanR) with a medium-strength promoter.",

    # Single inducible
    "An IPTG-inducible GFP expression circuit using pLac promoter and LacI repressor.",
    "An arabinose-inducible mCherry circuit with pBAD promoter and AraC activator.",
    "A tetracycline-inducible luciferase circuit using pTet promoter and TetR repressor.",
    "An aTc-inducible RFP expression circuit with constitutively expressed TetR.",
    "A rhamnose-inducible GFP circuit using the rhaBAD promoter system.",
    "An IPTG-inducible beta-galactosidase circuit with a strong RBS.",
    "A salicylate-inducible GFP circuit using the NahR/Psal system.",
    "A vanillic acid-inducible YFP circuit using VanR and the pVan promoter.",
    "A cumate-inducible mCherry circuit using CymR and the pCmt promoter.",
    "An anhydrotetracycline-inducible CFP circuit.",

    # Simple repression
    "A circuit where LacI constitutively represses GFP. Adding IPTG derepresses GFP.",
    "A NOT gate where CI repressor blocks GFP expression from the pR promoter.",
    "A TetR-based inverter: TetR constitutively expressed represses pTet-driven RFP.",
    "A LexA repression circuit where LexA blocks GFP from the recA promoter.",

    # Simple activation
    "A circuit where AraC activates pBAD to express GFP in the presence of arabinose.",
    "An OmpR-based activation circuit where OmpR activates pOmpC to drive GFP.",
    "A LuxR activation circuit where LuxR-AHL complex activates pLux driving mCherry.",

    # Simple biosensors
    "A lead biosensor using PbrR to activate GFP expression from pPbr when lead is present.",
    "A zinc biosensor where ZntR activates pZnt to produce GFP in response to zinc ions.",
    "A copper biosensor with CueR activating pCopA to drive mCherry expression.",
    "A cadmium biosensor using CadC to activate GFP expression upon cadmium detection.",
    "A nickel biosensor with NikR activating pNik to express luciferase.",
    "A formaldehyde biosensor using FrmR derepression to activate GFP.",
    "A phosphate starvation biosensor using PhoB to activate pPho driving GFP.",

    # Two-gene operons
    "An operon with two genes: GFP and mCherry expressed from a single pTac promoter with individual RBS sites.",
    "A bicistronic operon expressing AmilCP (blue chromoprotein) and KanR from a constitutive promoter.",
    "A two-gene operon under pBAD: LacZ and GFP with separate RBS elements.",

    # Operator-explicit circuits
    "A LacI repression circuit with explicit lacO operator site between pLac promoter and GFP. LacI binds lacO to block transcription. IPTG relieves repression.",
    "A TetR repression circuit with two tetO operator sites flanking the promoter region upstream of mCherry. TetR binds both tetO sites cooperatively.",
    "A lambda phage switch with operators OL1, OL2, OR1, OR2 controlling CI and Cro expression from divergent promoters pRM and pR.",
    "An AraC-regulated circuit with araO1 and araO2 operator sites. AraC loops DNA between operators in absence of arabinose, repressing pBAD. Arabinose converts AraC to an activator.",
    "A LexA-regulated SOS response circuit with an SOS box operator between the recA promoter and recA coding sequence. DNA damage triggers LexA autocleavage.",

    # Circuits with 'other' type components
    "A GFP expression circuit with an ssrA degradation tag fused to GFP for rapid protein turnover. Includes a constitutive promoter, RBS, GFP-ssrA fusion, and terminator.",
    "A secretion circuit where GFP is fused to an OmpA signal peptide for periplasmic export. The signal peptide is cleaved after translocation.",
    "A circuit with a theophylline riboswitch aptamer controlling translation of GFP. The riboswitch is positioned in the 5' UTR between the promoter and RBS.",
    "A Cre-lox recombination circuit: Cre recombinase is expressed from an IPTG-inducible promoter and flips a loxP-flanked terminator cassette to activate downstream GFP expression.",
    "A split-intein protein circuit: the N-terminal half of GFP fused to IntN is expressed from pBAD, and the C-terminal half fused to IntC is expressed from pTet. Trans-splicing reconstitutes functional GFP.",
    "A genetic insulator circuit using a ribozyme insulator (RiboJ) between the promoter and RBS to normalize expression levels across different promoter contexts.",
    "A circuit using FRT recombinase sites flanking a transcriptional stop cassette. Flp recombinase expressed from an arabinose-inducible promoter excises the stop to activate GFP.",
    "A protein scaffold circuit where three metabolic enzymes (AtoB, HMGS, HMGR) are co-localized on a synthetic scaffold protein with SH3, PDZ, and GBD domains for enhanced pathway flux.",
]

MEDIUM_DESCRIPTIONS = [
    # Logic gates
    "A genetic AND gate: GFP is only expressed when both IPTG and arabinose are present. pLac drives T7 RNAP, and pBAD drives a sigma factor. Both are needed to activate the T7 promoter driving GFP.",
    "A genetic NAND gate: a constitutive GFP is repressed only when both LacI and TetR are active. Each repressor is under a separate inducible promoter.",
    "A genetic NOR gate using LacI and TetR. Both repress a hybrid promoter driving GFP. GFP is only ON when neither inducer is present.",
    "A genetic OR gate where two independent promoters (pBAD and pLac) each drive their own copy of GFP.",
    "A genetic XNOR gate using a double-inverter topology with LacI and TetR cross-wired to produce GFP only when both or neither input is present.",
    "A genetic buffer gate that amplifies a weak arabinose signal using T7 RNAP cascade to drive high GFP output.",
    "A genetic IMPLIES gate: if arabinose is present, output is always ON. Without arabinose, output depends on IPTG.",
    "A majority gate with three inputs (IPTG, arabinose, aTc) where GFP is expressed only when at least two of three inputs are present.",

    # Toggle switches
    "A bistable toggle switch with LacI and TetR mutually repressing each other. GFP reports the LacI-dominant state, mCherry reports the TetR-dominant state.",
    "A three-state toggle switch using three repressors (LacI, TetR, CI) in a circular repression topology with three corresponding reporters.",
    "A toggle switch with asymmetric promoter strengths: strong pTrc driving LacI and weak pTet driving TetR, biased toward the LacI state.",

    # Oscillators
    "A repressilator with three repressors (LacI, TetR, CI) in a cycle, with a GFP reporter fused to the TetR output.",
    "A relaxation oscillator using a positive feedback loop (LuxR-AHL) coupled with a delayed negative feedback (AiiA degrading AHL).",
    "A two-node oscillator with an activator and repressor: the activator promotes both itself and the repressor, while the repressor inhibits the activator.",

    # Cascades
    "A three-stage signal cascade: IPTG induces T7 RNAP, which drives SP6 RNAP from a T7 promoter, which then drives GFP from an SP6 promoter.",
    "A phosphorelay cascade: EnvZ senses osmolarity, phosphorylates OmpR, which activates pOmpC to drive a second kinase that activates GFP.",
    "An amplification cascade where a weak input signal activates a transcription factor that drives T7 RNAP, which massively amplifies GFP expression.",

    # Feedback circuits
    "A negative autoregulation circuit where TetR represses its own promoter, creating fast response times and reduced noise.",
    "A positive autoregulation circuit where LuxR-AHL activates its own expression from pLux, creating bistability.",
    "A combined positive-negative feedback circuit: an activator promotes its own expression and a repressor, while the repressor inhibits the activator.",

    # Multi-gene biosensors with processing
    "An arsenic biosensor with signal amplification: ArsR detects arsenic, derepresses T7 RNAP, which amplifies GFP output 100-fold via T7 promoter.",
    "A mercury biosensor with a genetic filter: MerR activates GFP and also activates a delayed repressor to create a pulse response.",
    "A dual-input biosensor that detects both arsenic (via ArsR) and mercury (via MerR), outputting GFP only when both are present (AND logic).",
    "A ratiometric biosensor using two fluorescent proteins: GFP under an inducible promoter and mCherry under a constitutive promoter for normalization.",

    # Metabolic pathways (medium)
    "A three-enzyme pathway for naringenin production: 4CL, CHS, and CHI expressed from a single IPTG-inducible operon.",
    "A two-module isobutanol pathway: upstream module (AlsS, IlvC, IlvD) under pTrc and downstream module (KivD, AdhA) under pBAD.",
    "A carotenoid pathway: CrtE, CrtB, CrtI under constitutive promoter with RBS library variants for expression balancing.",

    # Cell-cell communication
    "A sender-receiver quorum sensing pair: sender constitutively produces LuxI (makes AHL), receiver has LuxR that activates GFP upon AHL detection.",
    "A bidirectional quorum sensing system: cells produce both AHL (via LuxI) and AI-2 (via LuxS), with GFP reporter for AHL and mCherry for AI-2.",

    # Medium complexity with operators
    "A toggle switch with explicit operator sites: LacI binds lacO in the pTet-lacO promoter to repress TetR, and TetR binds tetO in the pLac-tetO promoter to repress LacI. GFP reporter under the LacI-controlled arm.",
    "A genetic AND gate with explicit operators: pBAD with araO operator drives T7 RNAP, pLac with lacO operator drives a sigma factor. Both required to activate T7 promoter driving GFP. Include all operators explicitly.",
    "A repressilator with explicit operator sites: lacO in pLac, tetO in pTet, and lambda OR in pR. Three repressors (LacI, TetR, CI) bind their respective operators cyclically.",
    "A dual-operator repression circuit: CI repressor binds both OR1 and OR2 operators cooperatively to repress the pR promoter driving Cro. Cro in turn binds OR3 to repress CI from pRM.",

    # Medium complexity with 'other' components
    "A synthetic degradation timer: GFP is tagged with an ssrA-LAA degradation tag for ClpXP-mediated degradation. A separate circuit expresses an anti-adaptor protein (SspB inhibitor) from an IPTG-inducible promoter to stabilize GFP when induced.",
    "A recombinase-based memory circuit: Bxb1 integrase expressed from an arabinose-inducible promoter flips a DNA segment flanked by attB and attP sites, permanently switching from RFP to GFP expression. The flip is irreversible.",
    "A riboswitch-controlled metabolic valve: a TPP riboswitch in the 5' UTR of a bottleneck enzyme (HMGR) tunes translation in response to intracellular thiamine levels. The pathway includes three upstream enzymes under constitutive expression.",
    "A signal peptide secretion system: a therapeutic protein (human insulin) is fused to a PelB signal peptide for periplasmic export, with an enterokinase cleavage site linker between the signal peptide and insulin. Expression is controlled by IPTG-inducible pTac.",
    "A scaffold-mediated channeling circuit: three enzymes in a violacein pathway are recruited to a synthetic protein scaffold via cohesin-dockerin interactions. Each enzyme has a dockerin domain tag, and the scaffold has three matching cohesin domains.",
]

COMPLEX_DESCRIPTIONS = [
    # Multi-layer feedback with multiple reporters
    "A synthetic ecosystem with predator-prey dynamics: prey cells produce AHL via LuxI which activates a lysis gene in predator cells. Predator cells produce a toxin that kills prey. Both populations have fluorescent reporters (GFP for prey, mCherry for predator) and antibiotic resistance genes.",

    # Complex CRISPR circuits
    "A CRISPRi-based 4-bit decoder: dCas9 with four different gRNAs, each targeting a different promoter driving a different fluorescent protein (GFP, mCherry, BFP, YFP). An IPTG-inducible system selects which gRNAs are expressed, enabling combinatorial reporter control.",
    "A CRISPR-based gene drive circuit with safeguards: Cas9 cuts a target gene and inserts a copy of itself plus a gRNA and a payload gene. A separate kill switch using an inducible promoter drives an anti-CRISPR protein to halt the drive.",
    "A CRISPRa cascade with three stages: dCas9-VPR activates gRNA2 expression, gRNA2 with a second dCas9-p65 activates gRNA3, and gRNA3 activates the final GFP output. Each stage includes a different activation domain.",

    # Multicellular computing
    "A distributed 2-bit full adder using four engineered cell strains. Strain A computes XOR via a toggle switch. Strain B computes AND via dual-input promoter. Strains communicate via orthogonal quorum sensing molecules (AHL and AI-2). Sum output is GFP, carry output is mCherry.",
    "A three-strain consortium for sequential processing: Strain 1 detects arabinose and produces AHL. Strain 2 detects AHL and produces a diffusible intermediate (AI-2). Strain 3 detects AI-2 and produces GFP. Each strain has a different antibiotic resistance for selection.",

    # Complex metabolic engineering
    "A dynamic metabolic pathway with feedback regulation: a four-enzyme terpenoid pathway (AtoB, HMGS, HMGR, MK) where pathway flux is sensed by a FPP-responsive biosensor that feeds back to tune expression of the bottleneck enzyme HMGR via an RNA thermometer.",
    "A co-culture metabolic system: Strain A produces vanillin from ferulic acid via Fcs and Ech enzymes. Strain B converts vanillin to vanillic acid via Vdh. Cross-feeding is mediated by vanillin diffusion. Both strains have biosensors reporting their intermediate concentrations.",
    "A five-enzyme artemisinin precursor pathway with dynamic regulation: ADS, CYP71AV1, CPR, DBR2, and ALDH1 in two operons. A malonyl-CoA sensor tunes upstream flux, and a FPP sensor tunes downstream flux.",

    # Complex oscillators and pattern formation
    "A reaction-diffusion pattern formation circuit: a Turing-type system with a short-range activator (LuxR-AHL positive feedback) and a long-range inhibitor (a small diffusible repressor molecule). Includes both the activator and inhibitor gene circuits with their respective reporters.",
    "A synchronized genetic clock: a repressilator coupled to quorum sensing. LuxI is fused to one repressilator node, producing AHL that synchronizes oscillations across a population. All three nodes have different fluorescent reporters.",
    "A pulse generator with tunable width: an incoherent feedforward loop where IPTG simultaneously activates GFP directly and activates a slow-folding repressor (ssrA-tagged LacI variant) that eventually shuts GFP off. Pulse width is controlled by the repressor degradation rate.",

    # Genetic state machines
    "A two-state finite automaton that transitions between states based on sequential inputs. State 1 (GFP): IPTG input transitions to State 2. State 2 (mCherry): arabinose input transitions back to State 1. States are maintained by a toggle switch, and input detection uses recombinase-based irreversible switches.",
    "A genetic counter that counts up to 3 using nested recombinases: first Cre inverts a promoter (count 1), then Flp inverts a second promoter (count 2), then PhiC31 integrase activates a third element (count 3). Each count state has a unique reporter.",

    # Therapeutic circuits
    "A tumor-killing engineered bacterium: a hypoxia sensor (FNR) activates invasion genes (inv/hlyA) for tumor penetration. Inside the tumor, a quorum sensing circuit (LuxI/LuxR) triggers synchronized lysis releasing an anti-cancer toxin (ClyA) and a checkpoint inhibitor nanobody. A safety kill switch (inducible MazF) provides external control.",
    "A diabetes treatment circuit: a glucose sensor (chimeric receptor) activates calcium signaling which drives an NFAT-responsive promoter to express insulin. A negative feedback loop using insulin receptor signaling dampens production when insulin levels are sufficient. Includes a safety OFF switch using a small molecule-inducible repressor.",

    # Optogenetic circuits
    "A two-color optogenetic logic gate: blue light activates EL222 driving an intermediate repressor, red light activates PhyB-PIF3 driving GFP. The repressor inhibits GFP, creating a NOT-blue-AND-red logic. Includes all photoreceptor expression cassettes and cofactor biosynthesis genes.",
    "A spatiotemporal gene expression system using three orthogonal optogenetic channels: blue (EL222), green (CcaS/CcaR), and red (PhyB/PIF3). Each controls a different fluorescent protein. A master clock circuit pulses each light channel sequentially to create dynamic striped patterns.",

    # Biosafety and containment
    "A multi-layered biocontainment system: (1) an essential gene is replaced with a synthetic version requiring an unnatural amino acid, (2) a toxin-antitoxin pair (MazEF) where the antitoxin requires an external inducer, (3) a CRISPR-based self-destruct that targets the organism's own genome when a specific environmental signal is absent.",

    # RNA circuits
    "A ribocomputing circuit: three toehold switch sensors detect three different mRNAs (A, B, C) and compute (A AND B) OR C to produce GFP. Each toehold switch has a trigger RNA binding site, a hairpin structure, and a downstream reporter. The OR operation is achieved by having two separate GFP expression paths.",
    "An RNA-based genetic switchboard: four small transcription activating RNAs (STARs) each activate a different antisense RNA target controlling four different genes. A master RNA polymerase III-driven circuit selects which STARs are produced based on two chemical inputs.",

    # Complex circuits with operators
    "A multi-operator phage lambda decision circuit: the CI-Cro bistable switch with operators OL1, OL2, OL3, OR1, OR2, OR3. CI cooperatively binds OR1 and OR2 to repress Cro and activate its own transcription from pRM. At high concentrations CI also binds OR3 to repress itself. Cro binds OR3 first to repress CI. Include the N protein antitermination system and Q protein late gene activation.",
    "A synthetic chromosome partitioning system: ParA and ParB proteins with a parS operator site ensure plasmid segregation. ParB binds parS and nucleates into a complex. ParA-ATP binds DNA nonspecifically and is stimulated by ParB to hydrolyze ATP, creating a Brownian ratchet. Include a GFP-ParB fusion reporter and an IPTG-inducible ParA expression cassette.",

    # Complex circuits with 'other' components
    "A multi-layered protein quality control circuit: a target protein is tagged with both an ssrA degradation tag and a SsrA-protector peptide. Under normal conditions the protector shields the tag. When stress is detected by a sigma-32 responsive promoter, a TEV protease is expressed that cleaves the protector, exposing the ssrA tag for ClpXP degradation. Includes GFP reporter fused to the target.",
    "A synthetic organelle circuit: encapsulin shell proteins self-assemble into a protein compartment. Three cargo enzymes (IndA, IndB, IndC) are tagged with encapsulin targeting peptides for co-localization inside the compartment. The shell protein and each cargo are expressed from separate IPTG-inducible operons. An mCherry-targeting peptide fusion serves as a fluorescent loading reporter.",
    "A recombinase cascade with permanent memory: three sequential inputs (IPTG → arabinose → aTc) activate Cre, Flp, and PhiC31 integrase respectively. Each recombinase flips a DNA segment flanked by its cognate sites (loxP, FRT, attB/attP), creating an irreversible 3-bit memory register. Eight possible states each produce a unique combination of three fluorescent reporters (GFP, mCherry, BFP). Include insulators between each module.",
    "A directed evolution accelerator circuit: an error-prone DNA polymerase (PolI mutant with increased mutagenesis) is targeted to a specific genomic locus by a ColE1 origin. A theophylline riboswitch controls translation of the mutant PolI. A growth-coupled selection module links the target gene's function to antibiotic resistance via a synthetic regulatory cascade with explicit operator sites.",
    "A synthetic auxin signaling circuit in yeast: a plant TIR1 F-box protein is expressed and recruits an AUX/IAA degron-tagged transcriptional repressor for proteasomal degradation when auxin (IAA) is added. The repressor normally blocks a synthetic promoter with GAL4 UAS operators driving GFP. Auxin triggers repressor degradation, activating GFP. Includes SCF complex adapter proteins and a nuclear localization signal on the repressor.",
]

SYSTEM_PROMPT = """You are an expert synthetic biologist. Given a natural language description of a genetic circuit, output a JSON object describing the circuit components and interactions.

The JSON must follow this schema:
{
  "name": "string — circuit name",
  "description": "string — brief description",
  "components": [
    {
      "id": "string — unique identifier (snake_case, descriptive)",
      "type": "promoter | rbs | cds | terminator | operator | other",
      "name": "string — component name",
      "sequence": "string or null — DNA sequence if known"
    }
  ],
  "interactions": [
    {
      "type": "activation | repression | transcription | translation",
      "from": "component id",
      "to": "component id"
    }
  ]
}

Component type guide:
- "promoter": DNA region where transcription begins (e.g., pLac, pTet, pBAD, T7 promoter)
- "rbs": ribosome binding site for translation initiation (e.g., B0034, strong RBS)
- "cds": coding sequence for a protein (e.g., GFP, LacI, Cas9)
- "terminator": transcription termination signal (e.g., B0015, rrnB)
- "operator": DNA binding site for a repressor/activator within or near a promoter (e.g., lacO, tetO, araO, lambda operators OL/OR)
- "other": any component that does not fit above — degradation tags (ssrA/LAA), protein linkers, signal peptides, insulators, riboswitch aptamers, recombinase recognition sites (loxP, FRT, attB/attP), anti-CRISPR proteins, scaffold proteins, split protein domains, cofactors, small RNA elements

Interaction type guide:
- "transcription": promoter → cds (a promoter driving transcription of a gene)
- "translation": rbs → cds (a ribosome binding site enabling translation of a gene)
- "activation": protein/component → promoter/operator (a transcription factor activating expression)
- "repression": protein/component → promoter/operator (a transcription factor repressing expression)

Rules:
- Use ONLY these interaction types: activation, repression, transcription, translation
- Every component must have a unique id in snake_case
- Every promoter should have a transcription interaction to its downstream gene
- Every transcription unit MUST include explicit translation interactions (rbs → cds)
- Include RBS and terminator for each transcription unit
- Use "operator" type for DNA binding sites where repressors/activators bind
- Use "other" type for degradation tags, linkers, signal peptides, recombinase sites, riboswitches, scaffolds, insulators, and any non-standard parts
- Respond with valid JSON only, no explanation, no markdown fences."""


def generate_with_qwen(descriptions: list[str]) -> list[dict]:
    """Generate circuit JSONs using local Qwen2.5-72B via MLX."""
    from mlx_lm import load, generate

    print(f"Loading Qwen2.5-72B for {len(descriptions)} examples...")
    model, tokenizer = load("mlx-community/Qwen2.5-72B-Instruct-4bit")

    results = []
    for i, desc in enumerate(descriptions):
        print(f"  [{i+1}/{len(descriptions)}] Generating: {desc[:60]}...")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        t0 = time.time()
        response = generate(
            model, tokenizer, prompt=formatted, max_tokens=2048, verbose=False
        )
        elapsed = time.time() - t0

        # Try to parse and validate
        try:
            cleaned = response.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            # Validate interaction types
            valid_types = {"activation", "repression", "transcription", "translation"}
            bad = [
                ix for ix in parsed.get("interactions", [])
                if ix.get("type") not in valid_types
            ]
            if bad:
                print(f"    WARNING: Invalid interaction types found, skipping: {[b['type'] for b in bad]}")
                continue
            results.append({"description": desc, "circuit": parsed})
            print(f"    OK ({elapsed:.1f}s) — {len(parsed['components'])} components, {len(parsed['interactions'])} interactions")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    FAILED to parse JSON ({e}), skipping")
            continue

    # Free memory
    del model, tokenizer
    return results


def generate_with_gpt(descriptions: list[str], api_key: str) -> list[dict]:
    """Generate circuit JSONs using GPT-5.4 API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    results = []

    for i, desc in enumerate(descriptions):
        print(f"  [{i+1}/{len(descriptions)}] GPT-5.4: {desc[:60]}...")
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": desc},
                ],
                max_completion_tokens=2048,
                temperature=0.7,
            )
            output = response.choices[0].message.content
            elapsed = time.time() - t0

            cleaned = output.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            # Validate interaction types
            valid_types = {"activation", "repression", "transcription", "translation"}
            bad = [
                ix for ix in parsed.get("interactions", [])
                if ix.get("type") not in valid_types
            ]
            if bad:
                print(f"    WARNING: Invalid interaction types found, skipping: {[b['type'] for b in bad]}")
                continue
            results.append({"description": desc, "circuit": parsed})
            print(f"    OK ({elapsed:.1f}s) — {len(parsed['components'])} components, {len(parsed['interactions'])} interactions")
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    return results


def build_chat_entry(description: str, circuit_json: dict) -> dict:
    """Build a single chat-format training entry for mlx-lm LoRA."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
            {"role": "assistant", "content": json.dumps(circuit_json, indent=2)},
        ]
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data for Newgenes fine-tuning")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key for GPT-5.4 complex examples")
    parser.add_argument("--skip-local", action="store_true", help="Skip local Qwen generation")
    parser.add_argument("--skip-api", action="store_true", help="Skip GPT-5.4 API generation")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    all_results = []

    # Load existing hand-crafted examples from the original generate_data.py
    from generate_data import TRAINING_PAIRS, SYSTEM_PROMPT as ORIG_PROMPT
    print(f"Loaded {len(TRAINING_PAIRS)} hand-crafted examples")
    for desc, circuit in TRAINING_PAIRS:
        all_results.append({"description": desc, "circuit": circuit})

    # Stage 1: Local Qwen2.5-72B for simple + medium
    if not args.skip_local:
        local_descs = SIMPLE_DESCRIPTIONS + MEDIUM_DESCRIPTIONS
        print(f"\n=== Stage 1: Qwen2.5-72B ({len(local_descs)} examples) ===")
        local_results = generate_with_qwen(local_descs)
        all_results.extend(local_results)
        print(f"Stage 1 complete: {len(local_results)}/{len(local_descs)} successful")

    # Stage 2: GPT-5.4 for complex
    if not args.skip_api and args.openai_key:
        print(f"\n=== Stage 2: GPT-5.4 ({len(COMPLEX_DESCRIPTIONS)} examples) ===")
        api_results = generate_with_gpt(COMPLEX_DESCRIPTIONS, args.openai_key)
        all_results.extend(api_results)
        print(f"Stage 2 complete: {len(api_results)}/{len(COMPLEX_DESCRIPTIONS)} successful")
    elif not args.skip_api and not args.openai_key:
        print("\nSkipping Stage 2 (no --openai-key provided)")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_results)

    n = len(all_results)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train.jsonl": all_results[:train_end],
        "valid.jsonl": all_results[train_end:valid_end],
        "test.jsonl": all_results[valid_end:],
    }

    for filename, items in splits.items():
        path = out_dir / filename
        with open(path, "w") as f:
            for item in items:
                entry = build_chat_entry(item["description"], item["circuit"])
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(items)} examples to {path}")

    print(f"\nTotal: {n} examples ({train_end} train / {valid_end - train_end} valid / {n - valid_end} test)")


if __name__ == "__main__":
    main()

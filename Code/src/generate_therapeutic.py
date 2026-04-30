"""
Generate complex therapeutic circuit examples via GPT-5.4.
These target the gap in our training data: complex multi-module circuits
with activation/repression interactions for therapeutic applications.
"""

import json
import time
import argparse
from pathlib import Path

THERAPEUTIC_DESCRIPTIONS = [
    # === CAR-T Cell Circuits ===
    "A CAR-T cell circuit targeting CD19: a constitutive EF1a promoter drives expression of the anti-CD19 scFv-CD28-CD3zeta chimeric antigen receptor. Upon antigen binding, NFAT signaling activates an NFAT-responsive promoter that drives IL-2 secretion for autocrine stimulation. A separate safety module uses a small molecule (AP1903)-inducible iCasp9 kill switch.",

    "A dual-antigen AND-gate CAR-T circuit: two synNotch receptors detect HER2 and EpCAM separately. HER2-synNotch activates expression of the EpCAM-CAR via a Gal4-UAS promoter. Only cells seeing both antigens activate the full CAR and produce cytokines (IL-12 and IFN-gamma) from an NFAT-responsive promoter. Includes a TetR-controlled safety OFF switch.",

    "A universal CAR-T circuit with switchable specificity: a leucine zipper-based split CAR system where the intracellular signaling domain (CD3zeta-4-1BB) is fused to a leucine zipper adapter expressed constitutively. Bispecific adapter molecules link the zipper to different tumor antigens. An NFAT-responsive promoter drives a GFP reporter for activation monitoring.",

    # === Tumor-Targeting Bacteria ===
    "A tumor-homing engineered Salmonella circuit: a hypoxia-responsive FNR promoter activates expression of invasion genes (inv and hlyA) for tumor penetration. Once inside, a population density quorum sensing module (LuxI/LuxR with pLux promoter) triggers synchronized lysis via a phage lysis cassette (E-gene). Lysis releases a pre-accumulated anti-PD-L1 nanobody and the cytotoxin ClyA. An arabinose-inducible MazF kill switch provides external control.",

    "A tumor-sensing bacterial therapy circuit: a lactate sensor (LldR repression relieved by lactate) activates expression of a tumor-killing payload. The payload operon contains TNF-alpha, an anti-CTLA4 nanobody, and a flagellin adjuvant, each with their own RBS. A second module uses a constitutive promoter driving a TetR-repressed self-destruct (doc toxin) — removing tetracycline outside the tumor triggers bacterial death.",

    "A synthetic probiotic for colon cancer detection and treatment: a thiosulfate sensor (ThsS/ThsR two-component system) detects inflammation markers. ThsR activates a pThsR promoter driving CRISPRi repression of an oncogene mimic reporter (mCherry). Simultaneously, a nitric oxide sensor (NorR with pNorV promoter) activates expression of a therapeutic enzyme (myrosinase) that converts dietary glucosinolates into anti-cancer sulforaphane. Both sensors share a common quorum-sensing synchronization module.",

    # === Gene Therapy Circuits ===
    "A self-regulating gene therapy for hemophilia: Factor IX is expressed from a liver-specific albumin promoter. A negative feedback loop uses a Factor IX activity-responsive element that drives expression of a microRNA targeting the Factor IX mRNA 3'UTR, preventing overexpression. An insulator element flanks the therapeutic cassette, and AAV ITRs bracket the full construct.",

    "A gene therapy circuit for Parkinson's disease: three enzymes for dopamine synthesis (TH, AADC, GCH1) are expressed from a single neuron-specific synapsin promoter as a polycistronic mRNA with 2A self-cleaving peptides between each CDS. A GDNF neurotrophic factor is co-expressed from a separate activity-dependent promoter (c-fos) to provide neuroprotection only during neuronal activity.",

    "A conditional gene therapy for retinal degeneration: a rhodopsin promoter drives expression of channelrhodopsin-2 (ChR2) for optogenetic vision restoration. A light-responsive CRY2-CIB1 dimerization system acts as a safety valve — excessive light exposure activates CIB1-fused TetR to repress ChR2 expression via tetO operators, preventing phototoxicity. Includes woodchuck hepatitis virus posttranscriptional regulatory element (WPRE) for enhanced expression.",

    # === Metabolic Disease ===
    "A closed-loop insulin delivery circuit: a glucose-responsive promoter (using a chimeric glucose receptor coupled to a synthetic signaling cascade) drives insulin expression. Secreted insulin activates an insulin receptor on the same cell, which through a PI3K-Akt pathway activates a FoxO-responsive promoter driving expression of a transcriptional repressor that shuts down the glucose-responsive promoter. This creates a negative feedback homeostatic loop. Includes an ssrA degradation tag on the repressor for tunable dynamics.",

    "A phenylketonuria (PKU) treatment circuit: phenylalanine hydroxylase (PAH) and its cofactor recycling enzyme (DHPR) are expressed from a gut-specific promoter in engineered Lactobacillus. A phenylalanine-responsive riboswitch in the 5'UTR of PAH tunes expression based on substrate availability. A constitutive promoter drives expression of a surface-display anchor protein fused to a mucin-binding domain for gut colonization.",

    "A synthetic bile acid sensor for liver disease monitoring: a farnesoid X receptor (FXR) response element drives expression of a secreted nanoluciferase reporter detectable in blood. High bile acids activate FXR, which in turn activates the reporter. A second arm uses a constitutive promoter driving LXR which activates an LXR response element controlling CYP7A1, the rate-limiting enzyme in bile acid synthesis, creating a homeostatic feedback loop.",

    # === Immunotherapy ===
    "A cytokine circuit breaker for autoimmune therapy: a TNF-alpha responsive NF-kB promoter activates expression of IL-10 (anti-inflammatory cytokine) and a soluble TNF receptor decoy (sTNFR1). IL-10 feeds back to dampen NF-kB activation. A rapamycin-inducible FRB-FKBP dimerization system can force-activate the anti-inflammatory arm independent of TNF levels for emergency intervention.",

    "A synthetic regulatory T-cell programming circuit: a constitutive promoter drives a CAR targeting MOG (myelin antigen) for multiple sclerosis. Antigen recognition activates an NFAT-responsive promoter driving FOXP3 (master Treg transcription factor) and IL-35 (immunosuppressive cytokine). A positive feedback loop where FOXP3 activates its own promoter ensures stable Treg identity. An iCasp9 safety switch under AP1903 control.",

    "A bispecific T-cell engager (BiTE) production circuit: an engineered bacterium senses tumor hypoxia via FNR and produces a BiTE antibody linking CD3 (T-cell) to EpCAM (tumor). A quorum sensing timer (LuxI/LuxR with a delayed lysis cassette) controls release timing. A second BiTE targeting CD3-HER2 is co-expressed. Both BiTEs have secretion signal peptides (PelB). Population density is monitored via a GFP reporter under pLux control.",

    # === Diagnostic Circuits ===
    "A multiplexed disease biomarker detection circuit: three toehold switch RNA sensors detect miR-21 (cancer), miR-155 (inflammation), and miR-122 (liver damage). Each sensor controls a different reporter enzyme: miR-21 activates LacZ (blue), miR-155 activates GFP (green), miR-122 activates mCherry (red). An AND gate combining miR-21 and miR-155 produces a fourth output (luciferase) for cancer-with-inflammation detection.",

    "A CRISPR-based pathogen detection circuit: Cas13a programmed with a guide RNA targeting SARS-CoV-2 RNA. Upon target recognition, Cas13a collateral cleavage activates a quenched fluorescent reporter. A second Cas12a module targets influenza RNA with a different reporter (HEX fluorophore). Both systems share a T7 RNA polymerase amplification loop: initial detection activates T7 expression which amplifies the signal through T7 promoter-driven guide RNA production.",

    # === Biosafety Kill Switches ===
    "A triple-redundant kill switch for engineered organisms: (1) a toxin-antitoxin module (CcdB/CcdA) where the antitoxin CcdA requires constitutive IPTG induction — removal kills the cell; (2) an essential gene (thyA) is knocked out and supplied on a plasmid requiring synthetic amino acid supplementation; (3) a CRISPR self-destruct system where Cas9 targets the organism's origin of replication, held inactive by an anti-CRISPR protein (AcrIIA4) that requires arabinose induction. Removing any of the three inputs triggers death.",

    "A genetic firewall circuit: all essential aminoacyl-tRNA synthetases are replaced with versions requiring a synthetic amino acid (pAzF). A quorum sensing module (LuxI/LuxR) monitors population density and activates a lysis gene (holins) above a threshold to prevent environmental escape. A CRISPR-based horizontal gene transfer blocker (Cas9 targeting common conjugation genes) prevents genetic material from spreading. Each module has a constitutive GFP, mCherry, or BFP reporter respectively.",

    # === Tissue Engineering ===
    "A synthetic morphogen gradient circuit for tissue engineering: a source cell constitutively produces Sonic Hedgehog (Shh) fused to a GFP tag. Receiving cells express a Shh-responsive Gli promoter driving different transcription factors at different Shh concentrations: high Shh activates NKX2.2 (via a high-threshold promoter), medium activates OLIG2 (via a medium-threshold promoter using an operator-based band-pass filter), and low activates PAX6 (via a low-threshold promoter with a repressor-based low-pass filter).",

    "A stem cell differentiation controller: a four-stage sequential differentiation circuit. Stage 1 (pluripotency): constitutive Oct4 expression. Stage 2 (mesoderm): doxycycline-inducible Brachyury represses Oct4 via an operator and activates itself. Stage 3 (cardiac progenitor): Brachyury activates Mesp1 via a Brachyury-responsive promoter; Mesp1 represses Brachyury and activates itself. Stage 4 (cardiomyocyte): Mesp1 activates Nkx2.5 and GATA4 from a Mesp1-responsive promoter; Nkx2.5 and GATA4 cross-activate each other and repress Mesp1. Each stage has a unique fluorescent reporter.",

    # === Anti-Microbial ===
    "A phage-based antibiotic potentiator circuit: an engineered M13 phage delivers a CRISPR-Cas9 payload targeting the bacterial SOS response gene lexA. Disrupting SOS response re-sensitizes resistant bacteria to antibiotics. A second payload cassette expresses a beta-lactamase inhibitor (avibactam biosynthesis operon: three enzymes with individual RBS). A phage T7 promoter drives both cassettes after injection. A lysogeny decision module using CI repressor and Cro provides a toggle between lytic (therapeutic) and lysogenic (maintenance) modes.",

    "A living antibiotic circuit: an engineered Lactobacillus detects pathogenic E. coli via a CAI-1 quorum sensing receptor (CqsS/CqsR). Upon detection, the circuit activates production of: (1) a narrow-spectrum bacteriocin (microcin J25) targeting E. coli, (2) a biofilm-disrupting enzyme (DspB), and (3) a competitive exclusion factor (a superior iron siderophore biosynthesis operon). A self-immunity gene for microcin J25 is constitutively expressed. The entire therapeutic arm is under the control of the CAI-1-responsive promoter with a transcriptional amplifier cascade.",

    # === Drug Delivery ===
    "A smart drug delivery circuit in engineered yeast: a hypoxia sensor (DAN1 promoter) and an acidic pH sensor (ASR1 promoter) feed into an AND gate (both promoters drive halves of a split T7 RNA polymerase that reconstitutes only when both halves are expressed). The reconstituted T7 RNAP activates a T7 promoter driving: a prodrug-converting enzyme (cytosine deaminase converting 5-FC to 5-FU), a cell-penetrating peptide fused to p53 tumor suppressor, and a secreted VEGF trap. A galactose-inducible safety kill switch (URA3 with 5-FOA) provides external control.",

    "A theranostic circuit combining diagnosis and treatment: a two-input system senses both elevated lactate (via LldR) and low oxygen (via FNR). When both tumor markers are present, an AND gate activates: (1) a luciferase reporter for non-invasive imaging, (2) a therapeutic payload (tumor necrosis factor-related apoptosis-inducing ligand, TRAIL), and (3) a checkpoint inhibitor (anti-PD1 nanobody). The AND gate uses a split intein system where FNR drives the N-terminal half and LldR derepression drives the C-terminal half. A constitutive TetR-based timer delays therapeutic payload release by 6 hours after initial tumor detection, ensuring proper localization before drug release.",

    # === Neurological ===
    "A seizure-responsive therapeutic circuit: a glutamate-responsive promoter (using mGluR-based chimeric receptor) detects excess glutamate during seizures. This activates expression of: (1) an inhibitory DREADD receptor (hM4Di) for neuronal silencing upon administration of CNO, (2) neuropeptide Y (NPY) as an anti-epileptic peptide, and (3) adenosine kinase shRNA to boost local adenosine levels. A GABA-responsive element provides negative feedback — when inhibition is restored, the therapeutic circuit dampens. A destabilization domain on all therapeutic proteins ensures they are cleared rapidly when the trigger ceases.",

    "A Huntington's disease gene therapy circuit: a neuron-specific enolase promoter drives expression of an artificial miRNA targeting mutant HTT mRNA (with CAG repeats) while sparing wild-type HTT via allele-specific targeting. A BDNF neurotrophin is co-expressed from an activity-dependent promoter (Arc) for neuroprotection. A self-regulating module uses a decoy HTT mRNA fragment that titrates the miRNA when HTT levels drop too low, preventing over-silencing. The construct includes two insulator elements (cHS4) flanking the therapeutic cassette.",

    # === Aging / Senescence ===
    "A senescent cell elimination circuit (senolytic): a p16INK4a promoter (active in senescent cells) drives expression of a pro-apoptotic BH3-mimetic peptide and a TRAIL ligand to trigger apoptosis. A p21-responsive promoter provides a second input via an AND gate with p16 (both must be active = true senescence). The AND gate uses a two-hybrid split transcription factor. Non-senescent cells are protected by constitutive expression of anti-apoptotic Bcl-2 from a housekeeping promoter that is silenced in senescent cells. An external doxycycline-inducible override ensures the circuit can be paused.",

    "A telomere maintenance therapy circuit: a telomere damage sensor (using a TRF2-based reporter system) activates expression of a modified TERT (telomerase reverse transcriptase) with enhanced processivity. A cell-cycle-gated module ensures TERT is only active during S-phase by placing it under a cyclin E promoter with an additional CDK2-activity-responsive element. An RB-responsive safety brake represses TERT if the cell shows signs of hyperproliferation. Includes a GFP reporter fused to TRF1 for telomere length monitoring.",

    # === Microbiome Engineering ===
    "A synthetic probiotic for inflammatory bowel disease: a tetrathionate sensor (TtrS/TtrR two-component system) detects gut inflammation. The sensor activates production of: (1) anti-TNF-alpha nanobody for localized immunosuppression, (2) trefoil factor 3 (TFF3) for epithelial repair, and (3) butyrate biosynthesis operon (four enzymes: Thl, Hbd, Crt, Bcd) for short-chain fatty acid production. A constitutive promoter drives the essential colonization factor (EcN fimbriae). A CRISPR-based kill switch targeting the oriV is activated by a temperature-sensitive repressor — fever (>39C) triggers self-destruction.",

    "A gut-brain axis modulator circuit: engineered E. coli Nissle produces serotonin precursor 5-HTP via tryptophan hydroxylase (TPH1) and its cofactor BH4 recycling enzyme (DHPR). Expression is controlled by a cortisol-responsive promoter (using a chimeric glucocorticoid receptor fused to an E. coli DNA-binding domain). High cortisol (stress) increases 5-HTP production. A negative feedback via a serotonin-responsive repressor dampens production when serotonin levels normalize. GABA production from glutamate decarboxylase (GadB) provides a second anxiolytic output from a constitutive promoter.",

    # === Additional Complex Regulation ===
    "A genetic toggle switch with asymmetric stability for cell fate decisions: LacI represses pTet-cI while CI represses pLac-lacI, forming a bistable switch. However, LacI has an ssrA degradation tag making State 2 (CI high) more stable than State 1 (LacI high). A noise filter using a coherent feedforward loop (LacI activates an intermediate AraC which also must be present to repress CI) prevents spurious switching. Each state drives a different differentiation factor. GFP and mCherry reporters indicate the current state.",

    "A band-pass filter circuit for concentration-dependent gene activation: at low inducer (IPTG) concentrations, LacI represses the output GFP. At medium concentrations, LacI is titrated away and GFP is expressed. At high concentrations, a second repressor (TetR) expressed from a high-threshold promoter (requiring high IPTG via a cooperative LacI-release mechanism on a separate pLac variant with two lacO operators) represses GFP. The net effect is GFP expression only at intermediate IPTG levels. Includes all promoters, operators, RBS, and terminators explicitly.",

    "A winner-take-all neural network motif: three transcription factors (LacI, TetR, AraC) each activate their own expression (positive autoregulation) and repress the other two (mutual inhibition). The circuit settles into one of three stable states depending on initial conditions and input signals. Each state produces a different reporter (GFP, mCherry, BFP). Cross-repression uses dedicated operator sites for each pair interaction. A shared degradation tag ensures competition is based on production rate not accumulation.",

    "A pulse-width modulation circuit for precise gene dosing: a clock oscillator (repressilator: LacI→TetR→CI→LacI) generates periodic pulses. The duty cycle is controlled by an IPTG-tunable degradation rate of LacI (via ClpXP and an IPTG-responsive ssrA tag variant). Higher IPTG = faster LacI degradation = shorter LacI-high phase = longer TetR-high phase. The TetR-high phase drives a therapeutic output (insulin) from pTet. The pulse frequency is set by the repressilator period. Includes a GFP reporter on each node for monitoring.",

    # === Additional Therapeutic ===
    "A combination cancer immunotherapy circuit: an engineered bacterium produces three synergistic therapeutics controlled by different environmental sensors. (1) A hypoxia sensor (FNR) drives anti-PD-L1 nanobody for checkpoint blockade. (2) A lactate sensor (LldR) drives GM-CSF for dendritic cell recruitment. (3) A combined hypoxia-AND-quorum-sensing gate (FNR + LuxR/pLux) drives a tumor antigen-adjuvant fusion protein for in situ vaccination. A synchronized lysis circuit (LuxI/LuxR quorum sensing driving phage lysis genes) ensures pulsatile release of all therapeutics. An oral-administered aTc controls a master TetR safety switch.",

    "A multi-armed wound healing circuit: an engineered probiotic applied topically senses the wound environment. A low-oxygen sensor (FNR) drives VEGF for angiogenesis. A high-pH sensor (PhoP/PhoQ responding to alkaline wound pH) drives antimicrobial peptide LL-37. A quorum-sensing density monitor (LasI/LasR) limits population growth by activating a growth arrest gene (SulA) above threshold density. A constitutive promoter drives EGF (epidermal growth factor) for tissue repair. When the wound heals (normal oxygen + neutral pH), the therapeutic outputs shut off and a constitutive low-level toxin-antitoxin module (HipA/HipB with an unstable antitoxin) ensures the bacteria die without their engineered niche.",

    "A rheumatoid arthritis theranostic implant: engineered cells in a hydrogel capsule sense synovial fluid TNF-alpha via an NF-kB-responsive promoter. This drives: (1) secreted IL-1 receptor antagonist (IL-1Ra) for anti-inflammatory therapy, (2) soluble TNF receptor (sTNFR1) as a TNF decoy, and (3) secreted alkaline phosphatase (SEAP) as a blood-detectable biomarker. A negative feedback loop where IL-1Ra signaling dampens NF-kB activation creates homeostatic control. A doxycycline-inducible master OFF switch (TetR repressing the NF-kB sensor output) allows external circuit shutdown. The capsule also contains a constitutive erythromycin-dependent survival circuit — withdrawal of erythromycin kills the implanted cells.",
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

Component types:
- promoter: Drives transcription of downstream genes (e.g., pLac, pTet, pBAD, CMV, EF1a)
- rbs: Ribosome binding site — enables translation of the next CDS
- cds: Coding sequence — gene that produces a protein (e.g., GFP, LacI, TetR, Cas9)
- terminator: Stops transcription (e.g., B0015, rrnB, SV40)
- operator: DNA binding site for regulatory proteins (e.g., lacO, tetO, araO)
- other: Degradation tags, linkers, signal peptides, recombinase sites, ITRs, insulators, 2A peptides, etc.

Interaction types:
- transcription: promoter → cds (which gene a promoter drives)
- translation: rbs → cds (every RBS must translate its downstream CDS)
- activation: cds → promoter/operator (protein activates a promoter or via an operator)
- repression: cds → promoter/operator (protein represses a promoter or via an operator)

Rules:
1. Every CDS should have a transcription interaction (from a promoter) and a translation interaction (from an RBS)
2. Regulation targets should be promoters or operators, NOT other CDS
3. Include ALL components explicitly mentioned in the description
4. For complex circuits with many modules, include all modules with their complete promoter-RBS-CDS-terminator structure
5. For feedback loops, show the full regulatory chain
6. Use descriptive snake_case IDs (e.g., laci_cds, plac_promoter, gfp_reporter)

Respond with valid JSON only, no explanation."""


def generate_with_gpt(descriptions: list[str], api_key: str) -> list[dict]:
    """Generate circuit JSONs using GPT-5.4 API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    results = []

    for i, desc in enumerate(descriptions):
        print(f"  [{i+1}/{len(descriptions)}] {desc[:80]}...")
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": desc},
                ],
                max_completion_tokens=4096,
                temperature=0.7,
            )
            output = response.choices[0].message.content
            elapsed = time.time() - t0

            cleaned = output.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)

            # Validate
            valid_types = {"activation", "repression", "transcription", "translation"}
            comp_ids = {c['id'] for c in parsed.get('components', [])}

            bad_inter = [ix for ix in parsed.get("interactions", [])
                        if ix.get("type") not in valid_types]
            if bad_inter:
                print(f"    WARNING: Invalid interaction types: {[b['type'] for b in bad_inter]}")
                # Remove bad interactions instead of skipping entirely
                parsed['interactions'] = [ix for ix in parsed['interactions'] if ix['type'] in valid_types]

            # Remove interactions referencing non-existent components
            parsed['interactions'] = [
                ix for ix in parsed['interactions']
                if ix['from'] in comp_ids and ix['to'] in comp_ids
            ]

            n_comp = len(parsed['components'])
            n_inter = len(parsed['interactions'])
            inter_types = set(ix['type'] for ix in parsed['interactions'])

            if n_comp < 5:
                print(f"    SKIP: Only {n_comp} components (too simple for therapeutic)")
                continue

            results.append({"description": desc, "circuit": parsed})
            print(f"    OK ({elapsed:.1f}s) — {n_comp} comps, {n_inter} interactions ({', '.join(inter_types)})")
        except json.JSONDecodeError as e:
            print(f"    FAILED (JSON parse): {e}")
        except Exception as e:
            print(f"    FAILED: {e}")

        time.sleep(0.5)  # Rate limit

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate therapeutic circuit training data")
    parser.add_argument("--openai-key", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / 'scraped'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"=== Generating {len(THERAPEUTIC_DESCRIPTIONS)} therapeutic circuit examples via GPT-5.4 ===\n")
    results = generate_with_gpt(THERAPEUTIC_DESCRIPTIONS, args.openai_key)

    # Save raw results
    raw_path = output_dir / 'therapeutic_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} raw circuits to {raw_path}")

    # Convert to training format
    training = []
    for r in results:
        entry = {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': r['description']},
                {'role': 'assistant', 'content': json.dumps(r['circuit'], indent=2)},
            ]
        }
        training.append(entry)

    # Save JSONL
    training_path = output_dir / 'therapeutic_training.jsonl'
    with open(training_path, 'w') as f:
        for entry in training:
            f.write(json.dumps(entry) + '\n')

    print(f"\n=== SUMMARY ===")
    print(f"Generated: {len(results)}/{len(THERAPEUTIC_DESCRIPTIONS)} therapeutic circuits")
    print(f"Saved to: {training_path}")

    # Stats
    total_comps = sum(len(r['circuit']['components']) for r in results)
    total_inters = sum(len(r['circuit']['interactions']) for r in results)
    from collections import Counter
    inter_types = Counter()
    for r in results:
        for ix in r['circuit']['interactions']:
            inter_types[ix['type']] += 1

    print(f"Avg components: {total_comps/len(results):.1f}")
    print(f"Avg interactions: {total_inters/len(results):.1f}")
    print(f"Interaction types: {dict(inter_types)}")


if __name__ == '__main__':
    main()

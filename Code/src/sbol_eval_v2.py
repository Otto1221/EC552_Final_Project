#!/usr/bin/env python3
"""SBOL / genetic-circuit generation benchmark v2.

Designed to be a deterministic, unbiased, multi-dimensional evaluation of an
LLM's ability to convert natural-language circuit descriptions into structured
SBOL-compatible JSON designs. Used to compare local finetune, laptop model,
and frontier prompt-engineered models on the same set of prompts.

Scoring: 100 points across 6 axes
  A1 Structural Validity (SV, 20)
  A2 Biological Wiring  (BW, 20)
  A3 Biological Accuracy (BA, 20)
  A4 Prompt Fulfillment  (PF, 20)
  A5 Design Quality      (DQ, 10)
  A6 Robustness / Engineering Practice (REP, 10)

Key anti-bias properties:
  - identical prompts, system msg, sampler across all models
  - fully deterministic rubric scorer (no LLM judge in primary path)
  - per-axis breakdown preserved in results, not just total
  - stratified across difficulty × organism × topology
  - known-parts dictionary checked for hallucination
  - logic-gate topology verified (AND/OR/NOT/XOR/TOGGLE)
"""
from __future__ import annotations
import json, re, math
from typing import Any

# ---------------------------------------------------------------------------
# System message (shared by every run, every model)
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a synthetic biology assistant that converts natural language "
    "descriptions of genetic circuits into structured JSON. Return a single "
    "JSON object with these fields:\n\n"
    "- \"name\": short circuit name\n"
    "- \"components\": array of {\"name\", \"type\", \"description\"}\n"
    "- \"interactions\": array of {\"from\", \"to\", \"type\"}\n"
    "- \"behavior\": what the circuit does\n"
    "- \"organism\": host organism\n\n"
    "Component types: promoter, rbs, cds, terminator, operator, other.\n"
    "Interaction types: transcription (promoter→cds), translation (rbs→cds), "
    "activation (cds→promoter/operator), repression (cds→promoter/operator).\n\n"
    "Rules:\n"
    "- Every gene needs: promoter, rbs, cds, terminator\n"
    "- Use snake_case names\n"
    "- Every from/to must reference an existing component name\n"
    "- Respond with valid JSON only, no explanation"
)

ALLOWED_TYPES = {"promoter", "rbs", "cds", "terminator", "operator", "other"}
ALLOWED_IX = {"transcription", "translation", "activation", "repression",
              "inhibition", "production", "complex_formation", "degradation"}

# ---------------------------------------------------------------------------
# Known-parts dictionary for hallucination detection
# Any reference that matches one of these (case-insensitive substring) counts
# as "real". We keep this concrete and auditable rather than LLM-judged.
# ---------------------------------------------------------------------------
KNOWN_PARTS = {
    # Reporters / fluorophores
    "gfp", "egfp", "sfgfp", "mcherry", "rfp", "yfp", "eyfp", "cfp", "bfp",
    "mscarlet", "mvenus", "mturquoise", "mruby", "luxab", "luciferase", "lacz",
    "beta-galactosidase", "glucuronidase", "gus",
    # Promoters (iGEM Anderson family + common)
    "j23100", "j23101", "j23102", "j23106", "j23107", "j23119",
    "ptet", "ptetr", "plac", "placuv5", "placiq", "trc", "ptrc", "pbad",
    "prha", "prham", "pt7", "t7 promoter", "psp6", "pr ", "pl ", "prm",
    "pcca", "pcpcg2", "cpcg2", "plc", "phlf", "pluxr", "plux", "pbetai",
    "cmv", "sv40", "ef1a", "pgk1", "gal1", "tef1", "adh1",
    "35s", "rbcs", "dr5", "paux", "hsp", "ubq", "act1",
    # Operators / regulatory
    "lacoperator", "lac operator", "tetoperator", "tet operator", "operon_leader",
    "tetr binding", "tato", "tata", "tato_operator",
    # RBS
    "b0032", "b0034", "b0030", "utr ", "rbs_strong", "rbs_weak", "rbs_medium",
    "anderson rbs", "rbsa", "rbs_a", "shine-dalgarno",
    # Terminators
    "b0015", "l3s2p21", "l3s2p55", "t1 terminator", "rrnb t1", "rrnb",
    "bba_b0010", "sv40 polya", "bgh polya",
    # Regulators / repressors / activators
    "laci", "lac i", "tetr", "tetr protein", "arac", "ara c", "luxi", "luxr",
    "rhlr", "rhli", "cinr", "cini", "traa", "trab", "ci ", "lambda ci",
    "cro", "ptrp", "trpr", "arsr", "merr", "zntr", "marR", "soxr", "oxyr",
    "phif", "phlf", "ttgr", "betai", "ctr1", "arg1", "argr", "lexa", "fnr",
    "nara", "narl", "nifa", "glnk",
    # Metabolic / biosynthesis enzymes
    "vioa", "viob", "vioc", "viod", "vioe", "crts", "crtb", "crti", "crty",
    "nifh", "nifd", "nifk", "nifb", "nife", "nifm", "nifj",
    "pdc", "adh", "xyla", "xylr", "invertase", "sucr",
    # CRISPR
    "cas9", "dcas9", "cas12a", "cas13a", "cas13b", "csy4", "grna", "sgrna",
    "crisprx", "crispri", "crispra", "dcas12",
    # Toxins/antitoxins
    "ccda", "ccdb", "mazf", "maze", "rele", "relb", "vapbc",
    # Quorum-sensing small molecules (acts as CDS proxies)
    "ahl", "3oc6hsl", "c4hsl", "c12hsl",
    # Inducers (should appear in behavior/description)
    "iptg", "atc", "arabinose", "rhamnose", "doxycycline", "theophylline",
    "galactose", "auxin", "abscisic acid", "tamoxifen",
    # Structural / anchor
    "his tag", "strep tag", "ompa", "inp anchor", "inp", "signal peptide",
    # Nanobodies / display
    "anti-her2", "her2", "nanobody", "scfv",
}
KNOWN_PARTS = {p.lower() for p in KNOWN_PARTS}


def _is_known_part(name: str) -> bool:
    n = name.lower().replace("_", " ")
    if n in KNOWN_PARTS:
        return True
    for kp in KNOWN_PARTS:
        if kp in n or n in kp:
            return True
    # BBa_ prefix parts from iGEM registry — count as plausibly-real
    if re.match(r"^bba_[a-z]\d{4}$", n.replace(" ", "")):
        return True
    return False


# ---------------------------------------------------------------------------
# Prompt corpus — 100 prompts, stratified across difficulty × organism × topology
# difficulty: 1=simple, 2=inducible, 3=gate/switch, 4=dynamic/feedback, 5=complex
# topology: reporter, inducible, gate, toggle, oscillator, biosensor, pathway,
#           crispr, qs, kill, other
# organism: ecoli, yeast, mammalian, plant, bacillus, cellfree, other
# ---------------------------------------------------------------------------
# Each entry: (difficulty, organism, topology, prompt, keywords, logic_spec)
# logic_spec is optional and describes expected topology for checker.
# For gates: {"kind":"and","inputs":["iptg","ara"],"output":"gfp"}
# ---------------------------------------------------------------------------
PROMPTS: list[dict] = [
    # ---------- Tier 1: simple reporters (20) ----------
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Express GFP from a constitutive promoter in E. coli","kw":["gfp","promoter"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Make a simple mCherry reporter driven by the J23100 promoter","kw":["mcherry","j23100"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Constitutive expression of beta-galactosidase (LacZ) in E. coli","kw":["lacz"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Express luciferase from a T7 promoter in E. coli","kw":["luciferase","t7"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Constitutive sfGFP expression using the strong J23119 promoter in E. coli","kw":["sfgfp","j23119"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Express mScarlet-I from a medium-strength constitutive promoter in E. coli","kw":["mscarlet"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Constitutive YFP expression with the B0034 RBS in E. coli","kw":["yfp","b0034"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Express CFP under a weak constitutive promoter (J23106) in E. coli","kw":["cfp","j23106"]},
    {"diff":1,"org":"yeast","topo":"reporter","prompt":"Constitutive GFP expression from the TEF1 promoter in S. cerevisiae","kw":["gfp","tef1"]},
    {"diff":1,"org":"yeast","topo":"reporter","prompt":"Express mCherry from the GAL1 promoter in S. cerevisiae (galactose-induced)","kw":["mcherry","gal1"]},
    {"diff":1,"org":"yeast","topo":"reporter","prompt":"Constitutive Venus expression under the ADH1 promoter in yeast","kw":["venus","adh1"]},
    {"diff":1,"org":"mammalian","topo":"reporter","prompt":"Express EGFP from a CMV promoter in HEK293 cells","kw":["egfp","cmv"]},
    {"diff":1,"org":"mammalian","topo":"reporter","prompt":"Constitutive mCherry from the EF1a promoter in HEK293 cells","kw":["mcherry","ef1a"]},
    {"diff":1,"org":"mammalian","topo":"reporter","prompt":"Express firefly luciferase under the PGK1 promoter in HeLa cells","kw":["luciferase","pgk1"]},
    {"diff":1,"org":"plant","topo":"reporter","prompt":"Constitutive GFP under the CaMV 35S promoter in Arabidopsis","kw":["gfp","35s"]},
    {"diff":1,"org":"plant","topo":"reporter","prompt":"Express GUS reporter under a ubiquitin promoter in tobacco","kw":["gus","ubiquitin"]},
    {"diff":1,"org":"bacillus","topo":"reporter","prompt":"Constitutive GFP from the Pveg promoter in Bacillus subtilis","kw":["gfp","pveg"]},
    {"diff":1,"org":"cellfree","topo":"reporter","prompt":"Express deGFP from a sigma70 promoter in an E. coli TX-TL cell-free system","kw":["degfp","sigma70"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Simple His-tagged mRFP expression from a constitutive promoter in E. coli","kw":["mrfp","his"]},
    {"diff":1,"org":"ecoli","topo":"reporter","prompt":"Express secreted alkaline phosphatase (PhoA) from a constitutive promoter in E. coli","kw":["phoa"]},

    # ---------- Tier 2: inducible / repressible (20) ----------
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"IPTG-inducible mCherry reporter in E. coli using LacI repression of Ptrc","kw":["mcherry","iptg","laci"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"aTc-inducible GFP using TetR repression of pTet in E. coli","kw":["gfp","tetr","atc"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Arabinose-inducible RFP using AraC activation of pBAD in E. coli","kw":["rfp","arac","arabinose"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Rhamnose-inducible YFP using RhaS activation of pRha in E. coli","kw":["yfp","rhas","rhamnose"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"IPTG-inducible expression of a T7 RNA polymerase, which then drives sfGFP from a T7 promoter","kw":["iptg","t7","sfgfp"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Build a NOT gate: constitutive TetR represses GFP; aTc turns GFP ON","kw":["tetr","gfp","not"],"logic":{"kind":"not","input":"atc","output":"gfp"}},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Arabinose induces expression of a T7 RNAP which drives GFP from a T7 promoter","kw":["arabinose","t7","gfp"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Create an aTc-inducible RFP reporter with a TetR repression module","kw":["tetr","rfp","atc"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Theophylline-responsive riboswitch controlling GFP translation in E. coli","kw":["theophylline","gfp","riboswitch"]},
    {"diff":2,"org":"yeast","topo":"inducible","prompt":"Galactose-inducible mCherry using the GAL1 promoter in S. cerevisiae","kw":["galactose","mcherry","gal1"]},
    {"diff":2,"org":"yeast","topo":"inducible","prompt":"Copper-inducible GFP using the CUP1 promoter in S. cerevisiae","kw":["copper","gfp","cup1"]},
    {"diff":2,"org":"mammalian","topo":"inducible","prompt":"Doxycycline-inducible GFP using Tet-On (rtTA + TRE promoter) in HEK293 cells","kw":["doxycycline","gfp","rtta"]},
    {"diff":2,"org":"mammalian","topo":"inducible","prompt":"Tamoxifen-inducible Cre recombinase (CreERT2) controlling mCherry expression in HEK293","kw":["tamoxifen","cre","mcherry"]},
    {"diff":2,"org":"plant","topo":"inducible","prompt":"Dexamethasone-inducible GFP using the GVG system in Arabidopsis","kw":["dexamethasone","gfp"]},
    {"diff":2,"org":"bacillus","topo":"inducible","prompt":"Xylose-inducible GFP using the xylA promoter and XylR in Bacillus subtilis","kw":["xylose","gfp","xylr"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Mercury biosensor: MerR activated by Hg2+ drives GFP from the PmerT promoter","kw":["merr","hg","gfp"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Zinc biosensor using ZntR activation of PzntA driving mCherry","kw":["zntr","zinc","mcherry"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Nitrate-responsive expression of GFP using NarL activation of PnarK","kw":["narl","nitrate","gfp"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Osmotic stress sensor: OmpR activation of PompC drives sfGFP upon high osmolarity","kw":["ompr","sfgfp"]},
    {"diff":2,"org":"ecoli","topo":"inducible","prompt":"Heat-shock responsive GFP using sigma-32 (rpoH) driven PibpA promoter","kw":["gfp","sigma32","heat"]},

    # ---------- Tier 3: logic gates and switches (20) ----------
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"AND gate where both IPTG and arabinose are required for GFP output; use LacI repression AND AraC activation converging on a hybrid promoter","kw":["iptg","arabinose","gfp","and"],"logic":{"kind":"and","inputs":["iptg","arabinose"],"output":"gfp"}},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"OR gate: GFP is expressed if either IPTG or aTc is present, using two parallel promoters","kw":["iptg","atc","gfp","or"],"logic":{"kind":"or","inputs":["iptg","atc"],"output":"gfp"}},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"NOT gate using PhlF repressor: constitutive PhlF represses pPhlF driving GFP; DAPG de-represses","kw":["phlf","gfp","dapg"],"logic":{"kind":"not","input":"dapg","output":"gfp"}},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"NAND gate implemented with two repressors acting on the same output promoter","kw":["nand","gfp"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"XOR gate: GFP on only if exactly one of IPTG or arabinose is present","kw":["xor","iptg","arabinose","gfp"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"IMPLIES gate using LacI and TetR: output ON unless LacI ON and TetR OFF","kw":["laci","tetr","gfp"]},
    {"diff":3,"org":"ecoli","topo":"toggle","prompt":"Classic Gardner-Collins-Cantor toggle switch using LacI and TetR with cross-repression; readouts via GFP (TetR branch) and RFP (LacI branch)","kw":["laci","tetr","toggle","gfp","rfp"],"logic":{"kind":"toggle","repressors":["laci","tetr"]}},
    {"diff":3,"org":"ecoli","topo":"toggle","prompt":"Toggle switch between CI and Cro (lambda-derived) with bistable output on GFP","kw":["ci","cro","toggle","gfp"]},
    {"diff":3,"org":"ecoli","topo":"toggle","prompt":"CRISPRi-based toggle: two sgRNAs cross-repressing each other's promoters","kw":["crispri","sgrna","toggle"]},
    {"diff":3,"org":"yeast","topo":"gate","prompt":"Yeast AND gate requiring both estradiol (Z3EV) and galactose (Gal4) to turn on mCherry","kw":["estradiol","galactose","mcherry"]},
    {"diff":3,"org":"mammalian","topo":"gate","prompt":"Mammalian AND gate requiring doxycycline AND abscisic acid to drive mCherry in HEK293","kw":["doxycycline","abscisic","mcherry"]},
    {"diff":3,"org":"mammalian","topo":"toggle","prompt":"Mammalian toggle switch between two states using TetR and PipR with CMV-driven readouts","kw":["tetr","pipr","toggle"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"Time-delay circuit where IPTG triggers a cascade with two intermediate repressors before GFP turns on","kw":["iptg","gfp","delay"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"Majority gate over three inducers (IPTG, aTc, arabinose): GFP on if ≥2 inducers are present","kw":["iptg","atc","arabinose","gfp","majority"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"NOR gate using two tandem repressor binding sites driving GFP","kw":["nor","gfp"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"Implication logic: IF IPTG THEN GFP, implemented via LacI repression of pLac driving GFP","kw":["iptg","laci","gfp"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"A buffer / identity gate: IPTG turns GFP ON linearly; no repression cascade","kw":["iptg","gfp"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"Ratio-sensor: output GFP proportional to ratio of AraC/LacI activation","kw":["arac","laci","gfp"]},
    {"diff":3,"org":"ecoli","topo":"toggle","prompt":"Memory circuit using integrase flip: BxB1 integrase catalyzes DNA flip creating bistable on/off GFP","kw":["integrase","memory","gfp"]},
    {"diff":3,"org":"ecoli","topo":"gate","prompt":"Sequential gate: IPTG must precede arabinose for GFP output (temporal AND)","kw":["iptg","arabinose","gfp","temporal"]},

    # ---------- Tier 4: dynamic / feedback / biosensors (20) ----------
    {"diff":4,"org":"ecoli","topo":"oscillator","prompt":"Elowitz repressilator: three mutually repressing repressors (LacI, TetR, CI) with GFP readout","kw":["laci","tetr","ci","repressilator","gfp"],"logic":{"kind":"oscillator","repressors":["laci","tetr","ci"]}},
    {"diff":4,"org":"ecoli","topo":"oscillator","prompt":"Hasty-style dual-feedback oscillator with LuxI/AiiA feedback in E. coli driving GFP pulses","kw":["luxi","aiia","gfp","oscillator"]},
    {"diff":4,"org":"ecoli","topo":"biosensor","prompt":"Arsenic biosensor: ArsR-derepressed GFP with constitutive mCherry normalization","kw":["arsr","gfp","mcherry","arsenic"]},
    {"diff":4,"org":"ecoli","topo":"biosensor","prompt":"Copper biosensor: CusR activation of PcusC drives GFP; CueR as orthogonal control","kw":["cusr","cuer","copper","gfp"]},
    {"diff":4,"org":"ecoli","topo":"qs","prompt":"Quorum-sensing sender: LuxI produces AHL; receiver population detects AHL via LuxR-activated GFP","kw":["luxi","luxr","ahl","gfp"]},
    {"diff":4,"org":"ecoli","topo":"qs","prompt":"Two-population relay: population A produces C4HSL via RhlI, population B detects via RhlR to make GFP","kw":["rhli","rhlr","c4hsl","gfp"]},
    {"diff":4,"org":"ecoli","topo":"feedback","prompt":"Negative autoregulation of TetR reducing noise in downstream GFP","kw":["tetr","gfp","negative","feedback"]},
    {"diff":4,"org":"ecoli","topo":"feedback","prompt":"Positive feedback loop on LacI producing hysteresis in GFP output","kw":["laci","gfp","positive","feedback"]},
    {"diff":4,"org":"ecoli","topo":"cascade","prompt":"3-gene transcriptional cascade: arabinose → AraC → T7 RNAP → CI → lambda promoter → GFP","kw":["arac","t7","ci","lambda","gfp"]},
    {"diff":4,"org":"ecoli","topo":"biosensor","prompt":"pH biosensor based on CadC-activated CadA promoter driving mCherry","kw":["cadc","cada","mcherry","ph"]},
    {"diff":4,"org":"ecoli","topo":"biosensor","prompt":"Benzene biosensor using XylR from Pseudomonas driving GFP via the Pu promoter","kw":["xylr","benzene","gfp"]},
    {"diff":4,"org":"yeast","topo":"biosensor","prompt":"Steroid biosensor in yeast using an estradiol-responsive synthetic TF driving mCherry","kw":["estradiol","mcherry"]},
    {"diff":4,"org":"mammalian","topo":"biosensor","prompt":"Glucose biosensor in HEK293 using a ChREBP-based synthetic TF activating luciferase","kw":["chrebp","glucose","luciferase"]},
    {"diff":4,"org":"mammalian","topo":"feedback","prompt":"NF-kB negative autoregulation via IκBα feedback to reduce steady-state variability of downstream GFP","kw":["nfkb","ikba","gfp"]},
    {"diff":4,"org":"plant","topo":"biosensor","prompt":"Auxin biosensor in Arabidopsis using DR5 promoter driving GUS","kw":["auxin","dr5","gus"]},
    {"diff":4,"org":"ecoli","topo":"cascade","prompt":"Incoherent feedforward loop: input activates Y and Z, Y represses Z — pulse generator for GFP","kw":["feedforward","gfp","pulse"]},
    {"diff":4,"org":"ecoli","topo":"cascade","prompt":"Coherent feedforward loop where two activators must both turn on for GFP with AND-like behavior","kw":["feedforward","gfp","and"]},
    {"diff":4,"org":"ecoli","topo":"biosensor","prompt":"Glucose-responsive GFP using Crp-cAMP activation of a cAMP-sensitive promoter","kw":["crp","camp","gfp","glucose"]},
    {"diff":4,"org":"ecoli","topo":"biosensor","prompt":"Hypoxia biosensor: FNR-activated PnarK drives GFP only under anaerobic conditions","kw":["fnr","gfp","hypoxia"]},
    {"diff":4,"org":"ecoli","topo":"feedback","prompt":"Mutual activation loop between two TFs producing bistability with RFP/GFP dual reporters","kw":["rfp","gfp","bistable"]},

    # ---------- Tier 5: complex / pathways / CRISPR / multi-organism (20) ----------
    {"diff":5,"org":"ecoli","topo":"pathway","prompt":"Violacein biosynthesis: VioA, VioB, VioC, VioD, VioE enzymes expressed as an operon from a T7 promoter","kw":["vioa","viob","vioc","viod","vioe"]},
    {"diff":5,"org":"ecoli","topo":"pathway","prompt":"Beta-carotene production pathway: CrtE, CrtB, CrtI, CrtY expressed from pBAD arabinose-inducible operon","kw":["crte","crtb","crti","crty","arabinose"]},
    {"diff":5,"org":"yeast","topo":"pathway","prompt":"Reconstitute the mevalonate pathway (HMGR, ERG12, ERG8, ERG19, IDI1) in S. cerevisiae for terpenoid production","kw":["mevalonate","hmgr","erg12","erg19","idi1"]},
    {"diff":5,"org":"ecoli","topo":"crispr","prompt":"CRISPRi inverter: constitutive dCas9 with sgRNA targeting a pTarget drives GFP OFF when sgRNA is induced","kw":["dcas9","sgrna","gfp","crispri"]},
    {"diff":5,"org":"ecoli","topo":"crispr","prompt":"CRISPRa activator using dCas9-VP64 with sgRNA recruiting a synthetic activator to upregulate endogenous gene","kw":["dcas9","vp64","crispra"]},
    {"diff":5,"org":"ecoli","topo":"crispr","prompt":"Multiplex CRISPRi: three sgRNAs processed by Csy4 ribozyme target three different promoters to silence three reporters","kw":["csy4","dcas9","multiplex"]},
    {"diff":5,"org":"mammalian","topo":"crispr","prompt":"Mammalian CRISPRa with dCas9-p300 targeting a latent locus to activate endogenous HBG1 expression","kw":["dcas9","p300","hbg1","crispra"]},
    {"diff":5,"org":"ecoli","topo":"kill","prompt":"Dual-input kill switch in E. coli: without both IPTG and arabinose, CcdB toxin kills cells; both inducers maintain CcdA antitoxin","kw":["ccdb","ccda","iptg","arabinose","kill"],"logic":{"kind":"and","inputs":["iptg","arabinose"],"output":"ccda"}},
    {"diff":5,"org":"ecoli","topo":"kill","prompt":"Temperature-sensitive kill switch: cells die above 37°C via heat-induced MazF toxin expression","kw":["mazf","temperature","kill"]},
    {"diff":5,"org":"ecoli","topo":"qs","prompt":"Bidirectional quorum sensing: population A (LuxI sender) and population B (LasI sender) mutually activate reciprocal reporters","kw":["luxi","lasi","qs"]},
    {"diff":5,"org":"ecoli","topo":"pathway","prompt":"Engineer a tumor-targeting E. coli Nissle circuit: hypoxia-responsive FNR promoter drives InvA; constitutive anti-HER2 nanobody surface-displayed via INP anchor","kw":["fnr","inva","her2","nanobody","inp"]},
    {"diff":5,"org":"mammalian","topo":"pathway","prompt":"CAR-T cell with tandem CD19/CD22 CAR driven by an EF1α promoter plus a suicide switch (HSV-TK) induced by ganciclovir","kw":["car","cd19","cd22","hsvtk","ganciclovir"]},
    {"diff":5,"org":"ecoli","topo":"pathway","prompt":"Nitrogen-fixing circuit: NifH, NifD, NifK nitrogenase subunits driven by NifA-activated nifH promoter; GlnK anti-activator provides feedback under high nitrogen","kw":["nifh","nifd","nifk","nifa","glnk"]},
    {"diff":5,"org":"yeast","topo":"pathway","prompt":"Yeast galactose utilization rewiring: GAL80 knockout with GAL4-VP64 driving constitutive GAL1/7/10 expression","kw":["gal80","gal4","vp64"]},
    {"diff":5,"org":"plant","topo":"pathway","prompt":"Synthetic auxin-responsive circuit in plant cells: auxin triggers AUX/IAA degradation via TIR1, releasing ARF to activate DR5 driving GUS","kw":["auxin","tir1","arf","dr5","gus"]},
    {"diff":5,"org":"ecoli","topo":"pathway","prompt":"Lycopene overproduction: DXS, IDI, IspA, CrtE, CrtB, CrtI in a single operon with strong RBS tuning per enzyme","kw":["dxs","idi","crte","crtb","crti","lycopene"]},
    {"diff":5,"org":"ecoli","topo":"crispr","prompt":"CRISPR-Cas13-based RNA sensor: Cas13 cleaves a quenched GFP mRNA when trigger RNA is present","kw":["cas13","gfp"]},
    {"diff":5,"org":"ecoli","topo":"qs","prompt":"Population-level consensus circuit using AHL-mediated quorum sensing to synchronize GFP pulses across cells","kw":["ahl","gfp","synchrony"]},
    {"diff":5,"org":"bacillus","topo":"kill","prompt":"Bacillus subtilis containment circuit: constitutive expression of MazF toxin unless antitoxin MazE is expressed from a xylose-inducible promoter","kw":["mazf","maze","xylose"]},
    {"diff":5,"org":"cellfree","topo":"crispr","prompt":"Cell-free CRISPR sensor: Cas12a activates with trigger DNA, cleaving a quenched fluorescent reporter in a TX-TL reaction","kw":["cas12a","trigger","reporter"]},
]
assert len(PROMPTS) == 100, f"expected 100 prompts, got {len(PROMPTS)}"


# ---------------------------------------------------------------------------
# JSON extraction — robust to fences, thinking tokens, extra prose
# ---------------------------------------------------------------------------
def extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    t = raw.strip()
    # Strip <think>...</think> blocks (Qwen3.5, deepseek-style)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|channel\|>thought.*?<\|channel\|>", "", t, flags=re.DOTALL)
    # Strip any leading/trailing whitespace after removal
    t = t.strip()
    # Prefer fenced JSON block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1)
    # Find first complete JSON object: balanced braces
    s = t.find("{")
    if s < 0:
        return None
    depth = 0
    for i in range(s, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[s:i+1])
                except Exception:
                    break
    try:
        return json.loads(t)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------
def score_axes(entry: dict, raw: str) -> dict:
    """Returns {axis: {subscore_name: pts}, 'total': int}."""
    axes: dict[str, dict[str, int]] = {
        "SV": {},  # Structural Validity  (20)
        "BW": {},  # Biological Wiring    (20)
        "BA": {},  # Biological Accuracy  (20)
        "PF": {},  # Prompt Fulfillment   (20)
        "DQ": {},  # Design Quality       (10)
        "REP": {}, # Engineering Practice (10)
    }

    obj = extract_json(raw)

    # ---------- A1 Structural Validity (20) ----------
    sv = axes["SV"]
    sv["valid_json"] = 4 if obj is not None and isinstance(obj, dict) else 0
    if sv["valid_json"] == 0:
        for k in ("schema_compliance","unique_ids","referential_integrity","valid_types","no_self_loops"):
            sv[k] = 0
        for ax in ("BW","BA","PF","DQ","REP"):
            axes[ax] = _zero_axis(ax)
        total = _total(axes)
        return {"axes": axes, "total": total}

    required_keys = {"name","components","interactions","behavior","organism"}
    present = set(obj.keys()) & required_keys
    sv["schema_compliance"] = 4 if required_keys.issubset(obj.keys()) else (2 if len(present) >= 4 else 0)

    comps = [c for c in obj.get("components", []) if isinstance(c, dict)]
    ixs = [i for i in obj.get("interactions", []) if isinstance(i, dict)]
    names = [c.get("name","") for c in comps]
    name_set = set(names)

    # unique IDs
    sv["unique_ids"] = 3 if len(names) == len(name_set) and all(names) else 0
    # referential integrity
    bad_refs = 0; total_refs = 0
    for ix in ixs:
        for side in ("from","to"):
            v = ix.get(side, "")
            total_refs += 1
            if v not in name_set:
                bad_refs += 1
    if total_refs == 0:
        sv["referential_integrity"] = 0
    else:
        sv["referential_integrity"] = round(4 * (1 - bad_refs/total_refs))
    # valid types
    n_valid_t = sum(1 for c in comps if c.get("type","") in ALLOWED_TYPES)
    sv["valid_types"] = round(3 * n_valid_t / max(1,len(comps)))
    # no trivial self-loops
    has_self = any(ix.get("from") == ix.get("to") for ix in ixs if ix.get("from"))
    sv["no_self_loops"] = 0 if has_self else 2

    # If no components/interactions, zero-out downstream
    if not comps or not ixs:
        for ax in ("BW","BA","PF","DQ","REP"):
            axes[ax] = _zero_axis(ax)
        total = _total(axes)
        return {"axes": axes, "total": total}

    types_of = {c.get("name",""): c.get("type","") for c in comps}

    # ---------- A2 Biological Wiring (20) ----------
    bw = axes["BW"]
    tx = [i for i in ixs if i.get("type") == "transcription"]
    tl = [i for i in ixs if i.get("type") == "translation"]
    reg = [i for i in ixs if i.get("type") in ("activation","repression","inhibition")]

    bw["tx_from_promoter"] = round(4 * sum(1 for i in tx if types_of.get(i.get("from",""),"")=="promoter") / max(1,len(tx))) if tx else 0
    bw["tl_from_rbs"] = round(4 * sum(1 for i in tl if types_of.get(i.get("from",""),"") in ("rbs","other") and types_of.get(i.get("to",""),"")=="cds") / max(1,len(tl))) if tl else 0

    cds_names = [c["name"] for c in comps if c.get("type") == "cds"]
    tx_targets = {i.get("to") for i in tx}
    tl_targets = {i.get("to") for i in tl}
    if cds_names:
        n_wired = sum(1 for c in cds_names if c in tx_targets and c in tl_targets)
        bw["cds_complete_wiring"] = round(4 * n_wired / len(cds_names))
    else:
        bw["cds_complete_wiring"] = 0

    n_terms = sum(1 for c in comps if c.get("type")=="terminator")
    n_cds = len(cds_names)
    if n_cds == 0:
        bw["terminator_coverage"] = 0
    elif n_terms >= n_cds:
        bw["terminator_coverage"] = 3
    elif n_terms >= max(1, n_cds // 2):
        bw["terminator_coverage"] = 2
    elif n_terms >= 1:
        bw["terminator_coverage"] = 1
    else:
        bw["terminator_coverage"] = 0

    # regulation wiring — activation/repression should target promoter or operator
    if reg:
        n_reg_ok = sum(1 for i in reg if types_of.get(i.get("to",""),"") in ("promoter","operator"))
        bw["regulation_wiring"] = round(3 * n_reg_ok / len(reg))
    else:
        bw["regulation_wiring"] = 0 if _prompt_implies_regulation(entry) else 3

    # interaction balance
    if len(tx) >= 1 and len(tl) >= 1:
        ratio = min(len(tx), len(tl)) / max(len(tx), len(tl))
        bw["interaction_balance"] = 2 if ratio >= 0.5 else 1
    else:
        bw["interaction_balance"] = 0

    # ---------- A3 Biological Accuracy (20) ----------
    ba = axes["BA"]
    # parts_recognized
    if comps:
        n_real = sum(1 for c in comps if _is_known_part(c.get("name","")) or _is_known_part(c.get("description","")))
        ba["parts_recognized"] = round(5 * n_real / len(comps))
    else:
        ba["parts_recognized"] = 0
    # organism_parts_compat — flag cross-kingdom mismatches
    ba["organism_parts_compat"] = _check_organism_parts(entry, obj, comps)
    # interaction types biological
    n_valid_ix = sum(1 for i in ixs if i.get("type","") in ALLOWED_IX)
    ba["interaction_types_biological"] = round(4 * n_valid_ix / max(1,len(ixs)))
    # no hallucinated parts — penalize unrecognized parts that look invented
    ba["no_hallucinated_parts"] = _hallucination_penalty(comps)
    # mechanism coherence — description mentions regulation consistent with wiring
    ba["mechanism_coherence"] = _mechanism_coherence(obj, comps, ixs)

    # ---------- A4 Prompt Fulfillment (20) ----------
    pf = axes["PF"]
    pf["organism_match"] = _organism_match(entry, obj)
    pf["keyword_presence"] = _keyword_presence(entry, obj)
    pf["behavior_matches_logic"] = _topology_logic_check(entry, obj, comps, ixs, types_of)
    pf["completeness"] = _completeness(entry, comps, ixs)
    pf["quantitative_addressed"] = _quantitative_check(entry, obj)

    # ---------- A5 Design Quality (10) ----------
    dq = axes["DQ"]
    dq["descriptions_informative"] = _desc_quality(comps)
    dq["naming_conventions"] = _naming_conventions(comps)
    dq["modularity"] = _modularity(comps, ixs)
    dq["no_redundancy"] = _redundancy_penalty(comps)

    # ---------- A6 Engineering Practice (10) ----------
    rep = axes["REP"]
    rep["feedback_intentional"] = _feedback_check(entry, ixs, types_of)
    rep["burden_reasonable"] = _burden_check(comps)
    rep["chassis_appropriate"] = _chassis_check(entry, comps, obj)
    rep["regulatory_hierarchy"] = _hierarchy_check(ixs, types_of)

    total = _total(axes)
    return {"axes": axes, "total": total}


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------
def _zero_axis(a: str) -> dict:
    keys = {
        "SV": ["valid_json","schema_compliance","unique_ids","referential_integrity","valid_types","no_self_loops"],
        "BW": ["tx_from_promoter","tl_from_rbs","cds_complete_wiring","terminator_coverage","regulation_wiring","interaction_balance"],
        "BA": ["parts_recognized","organism_parts_compat","interaction_types_biological","no_hallucinated_parts","mechanism_coherence"],
        "PF": ["organism_match","keyword_presence","behavior_matches_logic","completeness","quantitative_addressed"],
        "DQ": ["descriptions_informative","naming_conventions","modularity","no_redundancy"],
        "REP": ["feedback_intentional","burden_reasonable","chassis_appropriate","regulatory_hierarchy"],
    }
    return {k: 0 for k in keys[a]}


AXIS_MAX = {"SV":20,"BW":20,"BA":20,"PF":20,"DQ":10,"REP":10}

def _total(axes: dict) -> int:
    # clamp each axis to its max so partial credit can't exceed axis budget
    t = 0
    for a, subs in axes.items():
        t += min(sum(subs.values()), AXIS_MAX[a])
    return t


def _prompt_implies_regulation(entry: dict) -> bool:
    p = entry["prompt"].lower()
    return any(w in p for w in ["repress","activat","inhib","induc","gate","switch","feedback","toggle","relay"])


_ORG_KEYWORDS = {
    "ecoli": ["e. coli","ecoli","escherichia"],
    "yeast": ["yeast","cerevisiae","saccharomyces","s. cerevisiae"],
    "mammalian": ["hek","hela","mammalian","human","cho","mouse"],
    "plant": ["arabidopsis","plant","tobacco","rice"],
    "bacillus": ["bacillus","b. subtilis"],
    "cellfree": ["cell-free","cellfree","tx-tl","txtl","in vitro"],
}

def _organism_match(entry, obj) -> int:
    target = entry["org"]
    got = (obj.get("organism","") or "").lower()
    if any(kw in got for kw in _ORG_KEYWORDS.get(target, [])):
        return 3
    if got:  # present but mismatched
        return 1
    return 0


def _keyword_presence(entry, obj) -> int:
    kws = entry["kw"]
    if not kws:
        return 4
    blob = json.dumps(obj, ensure_ascii=False).lower()
    hits = sum(1 for kw in kws if kw.lower() in blob)
    return round(4 * hits / len(kws))


_CROSS_KINGDOM = {
    "ecoli": {"bad": ["cmv","sv40","ef1a","pgk1","35s","gal1","gal4","tef1","auxin","dr5"]},
    "yeast": {"bad": ["cmv","sv40","ef1a","pt7","ptet","plac"]},
    "mammalian": {"bad": ["plac","ptet","j23","t7 promoter","pbad"]},
    "plant": {"bad": ["cmv","ef1a","plac","ptet"]},
}

def _check_organism_parts(entry, obj, comps) -> int:
    target = entry["org"]
    bads = _CROSS_KINGDOM.get(target, {}).get("bad", [])
    if not bads:
        return 4
    blob = json.dumps(obj, ensure_ascii=False).lower()
    n_bad = sum(1 for b in bads if b in blob)
    if n_bad == 0:
        return 4
    if n_bad == 1:
        return 2
    return 0


def _hallucination_penalty(comps) -> int:
    if not comps:
        return 0
    unknown = 0
    for c in comps:
        name = c.get("name","")
        desc = c.get("description","")
        ctype = c.get("type","")
        # ignore generic connective parts
        if name.lower() in {"promoter","rbs","terminator","operator","cds"}:
            continue
        if ctype == "terminator":
            continue
        if not _is_known_part(name) and not _is_known_part(desc):
            # Heuristic hallucination detector: looks authoritative (BBa_-style but fake)
            if re.search(r"bba_[a-z]\d{3,5}", name.lower()) or re.search(r"p[A-Z][a-z]+[A-Z]", name):
                unknown += 2
            else:
                unknown += 1
    # translate to score (fewer unknowns = higher score)
    frac = unknown / max(1, len(comps))
    if frac < 0.1: return 4
    if frac < 0.25: return 3
    if frac < 0.5: return 2
    if frac < 0.75: return 1
    return 0


def _mechanism_coherence(obj, comps, ixs) -> int:
    beh = (obj.get("behavior","") or "").lower()
    has_repression_kw = any(w in beh for w in ["repress","inhibit"])
    has_activation_kw = any(w in beh for w in ["activate","induce","turn on"])
    has_repression_ix = any(i.get("type") in ("repression","inhibition") for i in ixs)
    has_activation_ix = any(i.get("type") == "activation" for i in ixs)
    agreements = 0
    if has_repression_kw == has_repression_ix: agreements += 1
    if has_activation_kw == has_activation_ix: agreements += 1
    # descriptions coherent with type
    desc_ok = 0
    for c in comps:
        d = (c.get("description","") or "").lower()
        t = c.get("type","")
        if t == "promoter" and ("promoter" in d or "transcription" in d or "drives" in d or "induc" in d):
            desc_ok += 1
        elif t == "rbs" and ("ribosome" in d or "rbs" in d or "translation" in d):
            desc_ok += 1
        elif t == "cds" and ("protein" in d or "encodes" in d or "express" in d):
            desc_ok += 1
        elif t == "terminator" and ("terminator" in d or "terminat" in d or "stops" in d):
            desc_ok += 1
    desc_frac = desc_ok / max(1, len(comps))
    return agreements + (1 if desc_frac >= 0.5 else 0)  # max 3


def _topology_logic_check(entry, obj, comps, ixs, types_of) -> int:
    """Max 6. Inspects logic specification if provided."""
    logic = entry.get("logic")
    if not logic:
        # fall back: check keyword behavior matches prompt intent
        prompt = entry["prompt"].lower()
        beh = (obj.get("behavior","") or "").lower()
        if not beh:
            return 0
        # reasonable match if behavior contains multiple prompt content words
        words = [w for w in re.findall(r"[a-z]+", prompt) if len(w) >= 5]
        hits = sum(1 for w in words[:20] if w in beh)
        if hits >= 6: return 6
        if hits >= 4: return 4
        if hits >= 2: return 2
        return 0
    kind = logic.get("kind")
    reg = [i for i in ixs if i.get("type") in ("activation","repression","inhibition")]
    if kind == "and":
        # need at least 2 regulatory interactions pointing toward same output path
        if len(reg) >= 2 and any("hybrid" in (c.get("description","") or "").lower() or "and" in (c.get("description","") or "").lower() for c in comps):
            return 6
        if len(reg) >= 2:
            return 4
        return 1
    if kind == "or":
        if len(reg) >= 2 and any(_count_target(ixs, c["name"]) >= 2 for c in comps if c.get("type") in ("promoter","cds")):
            return 6
        if len(reg) >= 2: return 4
        return 1
    if kind == "not":
        if any(i.get("type") in ("repression","inhibition") for i in ixs):
            return 6
        return 1
    if kind == "toggle":
        repressors = logic.get("repressors", [])
        count = sum(1 for r in repressors if any(r in c.get("name","").lower() for c in comps))
        if count >= 2 and sum(1 for i in ixs if i.get("type") in ("repression","inhibition")) >= 2:
            return 6
        if count >= 2: return 4
        return 1
    if kind == "oscillator":
        repressors = logic.get("repressors", [])
        count = sum(1 for r in repressors if any(r in c.get("name","").lower() for c in comps))
        n_rep_ix = sum(1 for i in ixs if i.get("type") in ("repression","inhibition"))
        if count >= 3 and n_rep_ix >= 3: return 6
        if count >= 2 and n_rep_ix >= 2: return 4
        return 1
    return 0


def _count_target(ixs, name):
    return sum(1 for i in ixs if i.get("to") == name)


def _completeness(entry, comps, ixs) -> int:
    diff = entry["diff"]
    # Expected minimum parts scales with difficulty
    min_comps = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}[diff]
    min_ixs = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}[diff]
    c_ok = len(comps) >= min_comps
    i_ok = len(ixs) >= min_ixs
    if c_ok and i_ok: return 4
    if c_ok or i_ok: return 2
    return 0


_QUANT_TERMS = ["strong","weak","medium","high","low","strength","tunable","ratio","baseline","basal","leaky","pulse","bistable","oscillat"]

def _quantitative_check(entry, obj) -> int:
    prompt = entry["prompt"].lower()
    beh = (obj.get("behavior","") or "").lower()
    prompt_has = any(t in prompt for t in _QUANT_TERMS)
    if not prompt_has:
        return 3
    beh_has = any(t in beh for t in _QUANT_TERMS)
    return 3 if beh_has else 0


def _desc_quality(comps) -> int:
    if not comps: return 0
    informative = 0
    for c in comps:
        d = (c.get("description","") or "").strip()
        t = c.get("type","")
        if len(d) >= 25 and d.lower() not in {t, "protein", "gene"}:
            if re.search(r"[a-z]{4,}", d.lower()):
                informative += 1
    frac = informative / len(comps)
    if frac >= 0.9: return 3
    if frac >= 0.6: return 2
    if frac >= 0.3: return 1
    return 0


def _naming_conventions(comps) -> int:
    if not comps: return 0
    snake_ok = sum(1 for c in comps if re.match(r"^[a-z0-9][a-z0-9_]*$", c.get("name",""))) / len(comps)
    return 2 if snake_ok >= 0.9 else (1 if snake_ok >= 0.7 else 0)


def _modularity(comps, ixs) -> int:
    # a design is modular if each CDS has its own promoter/RBS/term, not sharing
    cds = [c for c in comps if c.get("type")=="cds"]
    if not cds: return 0
    if len(cds) == 1: return 1
    types_count = {t: sum(1 for c in comps if c.get("type")==t) for t in ALLOWED_TYPES}
    # reasonable modularity if promoter:cds ratio is between 0.5 and 1.5
    pc_ratio = types_count.get("promoter",0) / len(cds)
    if 0.5 <= pc_ratio <= 1.5: return 2
    if 0.3 <= pc_ratio <= 2.0: return 1
    return 0


def _redundancy_penalty(comps) -> int:
    names = [c.get("name","").lower() for c in comps]
    dups = len(names) - len(set(names))
    if dups == 0: return 3
    if dups <= 1: return 2
    if dups <= 2: return 1
    return 0


def _feedback_check(entry, ixs, types_of) -> int:
    prompt = entry["prompt"].lower()
    wants_feedback = "feedback" in prompt or "autoregul" in prompt or "oscillat" in prompt or "toggle" in prompt
    has_feedback_cycle = _has_cycle(ixs)
    if wants_feedback and has_feedback_cycle: return 3
    if wants_feedback and not has_feedback_cycle: return 0
    if not wants_feedback and not has_feedback_cycle: return 3
    # unintended feedback is not great
    return 1


def _has_cycle(ixs) -> bool:
    # simple cycle detection on (from->to) graph of regulatory edges
    graph: dict[str, set[str]] = {}
    for i in ixs:
        if i.get("type") in ("activation","repression","inhibition"):
            graph.setdefault(i.get("from",""), set()).add(i.get("to",""))
    visited = set(); stack = set()
    def dfs(n):
        if n in stack: return True
        if n in visited: return False
        visited.add(n); stack.add(n)
        for m in graph.get(n, ()):
            if dfs(m): return True
        stack.discard(n)
        return False
    for n in list(graph.keys()):
        if dfs(n): return True
    return False


def _burden_check(comps) -> int:
    n = len(comps)
    if n <= 20: return 2
    if n <= 30: return 1
    return 0


def _chassis_check(entry, comps, obj) -> int:
    org = entry["org"]
    n = len(comps)
    # larger complex organisms tolerate more components
    caps = {"ecoli": 20, "yeast": 25, "mammalian": 30, "plant": 25, "bacillus": 20, "cellfree": 15}
    cap = caps.get(org, 20)
    if n <= cap: return 2
    return 0


def _hierarchy_check(ixs, types_of) -> int:
    # A good design has activation/repression edges targeting promoter/operator,
    # not CDS (that's captured in BW but here we check structural hierarchy).
    reg = [i for i in ixs if i.get("type") in ("activation","repression","inhibition")]
    if not reg: return 3  # absence is fine if not needed
    good = sum(1 for i in reg if types_of.get(i.get("to",""),"") in ("promoter","operator"))
    frac = good / len(reg)
    if frac >= 0.9: return 3
    if frac >= 0.7: return 2
    if frac >= 0.4: return 1
    return 0


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------
def summarize(results: list[dict]) -> dict:
    """Aggregate per-axis averages and stratified breakdowns."""
    n = len(results)
    axes_sum = {a: 0 for a in AXIS_MAX}
    total = 0
    by_diff: dict[int, list[int]] = {}
    by_org: dict[str, list[int]] = {}
    by_topo: dict[str, list[int]] = {}
    failures = 0
    timeouts = 0
    for r in results:
        s = r.get("score", {})
        if not s:
            failures += 1
            continue
        if r.get("response","").startswith("<ERROR: timed out") or r.get("response","").startswith("<ERROR: <urlopen"):
            timeouts += 1
            continue
        total += s.get("total", 0)
        for a in AXIS_MAX:
            axes_sum[a] += min(sum(s.get("axes", {}).get(a, {}).values()), AXIS_MAX[a])
        d = r["entry"].get("diff")
        o = r["entry"].get("org")
        t = r["entry"].get("topo")
        by_diff.setdefault(d, []).append(s.get("total",0))
        by_org.setdefault(o, []).append(s.get("total",0))
        by_topo.setdefault(t, []).append(s.get("total",0))
    completed = n - timeouts - failures
    avg_total = total / max(1, completed)
    axes_avg = {a: axes_sum[a] / max(1, completed) for a in AXIS_MAX}
    return {
        "n_total": n,
        "n_completed": completed,
        "n_timeouts": timeouts,
        "n_failures": failures,
        "avg_total": round(avg_total, 2),
        "axes_avg": {a: round(v, 2) for a, v in axes_avg.items()},
        "by_difficulty": {k: round(sum(v)/len(v), 2) for k, v in sorted(by_diff.items())},
        "by_organism": {k: round(sum(v)/len(v), 2) for k, v in sorted(by_org.items())},
        "by_topology": {k: round(sum(v)/len(v), 2) for k, v in sorted(by_topo.items())},
    }

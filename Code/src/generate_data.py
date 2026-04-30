"""
Generate synthetic training data for fine-tuning.
Uses the circuit system prompt and a set of diverse circuit descriptions
to produce (instruction, response) pairs in JSONL format for mlx-lm LoRA.
"""

import json
from pathlib import Path

# Each entry: (natural language description, expected circuit JSON)
TRAINING_PAIRS = [
    # === CRISPR Systems ===
    (
        "A CRISPRi circuit where dCas9 with a single gRNA represses GFP expression from a constitutive promoter.",
        {
            "name": "CRISPRi_GFP_Repression",
            "description": "CRISPRi circuit using dCas9 and one gRNA to repress GFP expression.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None},
                {"id": "pU6", "type": "promoter", "name": "U6 Promoter", "sequence": None},
                {"id": "gRNA1", "type": "cds", "name": "gRNA targeting pConst", "sequence": None},
                {"id": "ter_grna", "type": "terminator", "name": "gRNA Terminator", "sequence": None},
                {"id": "pTet", "type": "promoter", "name": "TetR Promoter", "sequence": None},
                {"id": "rbs_dcas9", "type": "rbs", "name": "RBS for dCas9", "sequence": None},
                {"id": "dcas9", "type": "cds", "name": "dCas9", "sequence": None},
                {"id": "ter_dcas9", "type": "terminator", "name": "dCas9 Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "gfp"},
                {"type": "transcription", "from": "pU6", "to": "gRNA1"},
                {"type": "transcription", "from": "pTet", "to": "dcas9"},
                {"type": "repression", "from": "dcas9", "to": "pConst"}
            ]
        }
    ),
    (
        "A CRISPRa circuit where dCas9-VPR is guided by gRNA1 to activate mCherry expression from a minimal promoter, and gRNA2 activates luciferase from a second minimal promoter.",
        {
            "name": "CRISPRa_Dual_Activation",
            "description": "CRISPRa circuit using dCas9-VPR with two gRNAs to activate mCherry and luciferase.",
            "components": [
                {"id": "pCMV", "type": "promoter", "name": "CMV Promoter for dCas9-VPR", "sequence": None},
                {"id": "rbs_dcas9", "type": "rbs", "name": "RBS for dCas9-VPR", "sequence": None},
                {"id": "dcas9_vpr", "type": "cds", "name": "dCas9-VPR", "sequence": None},
                {"id": "ter_dcas9", "type": "terminator", "name": "dCas9-VPR Terminator", "sequence": None},
                {"id": "pU6_1", "type": "promoter", "name": "U6 Promoter for gRNA1", "sequence": None},
                {"id": "gRNA1", "type": "cds", "name": "gRNA1 targeting pMin1", "sequence": None},
                {"id": "ter_g1", "type": "terminator", "name": "gRNA1 Terminator", "sequence": None},
                {"id": "pU6_2", "type": "promoter", "name": "U6 Promoter for gRNA2", "sequence": None},
                {"id": "gRNA2", "type": "cds", "name": "gRNA2 targeting pMin2", "sequence": None},
                {"id": "ter_g2", "type": "terminator", "name": "gRNA2 Terminator", "sequence": None},
                {"id": "pMin1", "type": "promoter", "name": "Minimal Promoter for mCherry", "sequence": None},
                {"id": "rbs_mcherry", "type": "rbs", "name": "RBS for mCherry", "sequence": None},
                {"id": "mcherry", "type": "cds", "name": "mCherry", "sequence": None},
                {"id": "ter_mcherry", "type": "terminator", "name": "mCherry Terminator", "sequence": None},
                {"id": "pMin2", "type": "promoter", "name": "Minimal Promoter for Luciferase", "sequence": None},
                {"id": "rbs_luc", "type": "rbs", "name": "RBS for Luciferase", "sequence": None},
                {"id": "luc", "type": "cds", "name": "Luciferase", "sequence": None},
                {"id": "ter_luc", "type": "terminator", "name": "Luciferase Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pCMV", "to": "dcas9_vpr"},
                {"type": "transcription", "from": "pU6_1", "to": "gRNA1"},
                {"type": "transcription", "from": "pU6_2", "to": "gRNA2"},
                {"type": "activation", "from": "dcas9_vpr", "to": "pMin1"},
                {"type": "activation", "from": "dcas9_vpr", "to": "pMin2"},
                {"type": "transcription", "from": "pMin1", "to": "mcherry"},
                {"type": "transcription", "from": "pMin2", "to": "luc"}
            ]
        }
    ),
    (
        "A CRISPR kill switch where Cas9 cuts an essential gene when induced by IPTG.",
        {
            "name": "CRISPR_Kill_Switch",
            "description": "CRISPR kill switch that cleaves an essential gene upon IPTG induction.",
            "components": [
                {"id": "pLac", "type": "promoter", "name": "Lac Promoter (IPTG-inducible)", "sequence": None},
                {"id": "rbs_cas9", "type": "rbs", "name": "RBS for Cas9", "sequence": None},
                {"id": "cas9", "type": "cds", "name": "Cas9", "sequence": None},
                {"id": "ter_cas9", "type": "terminator", "name": "Cas9 Terminator", "sequence": None},
                {"id": "pU6", "type": "promoter", "name": "U6 Promoter", "sequence": None},
                {"id": "gRNA_ess", "type": "cds", "name": "gRNA targeting essential gene", "sequence": None},
                {"id": "ter_grna", "type": "terminator", "name": "gRNA Terminator", "sequence": None},
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "essential", "type": "cds", "name": "Essential Gene", "sequence": None},
                {"id": "ter_ess", "type": "terminator", "name": "Essential Gene Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "activation", "from": "pLac", "to": "cas9"},
                {"type": "transcription", "from": "pU6", "to": "gRNA_ess"},
                {"type": "repression", "from": "cas9", "to": "essential"},
                {"type": "transcription", "from": "pConst", "to": "essential"}
            ]
        }
    ),

    # === Logic Gates ===
    (
        "A genetic AND gate where GFP is expressed only when both arabinose and IPTG are present. The arabinose-inducible pBAD promoter drives T7 RNA polymerase, and the IPTG-inducible pLac promoter drives a T7 promoter-driven GFP.",
        {
            "name": "AND_Gate",
            "description": "Genetic AND gate: GFP expressed only with both arabinose and IPTG.",
            "components": [
                {"id": "pBAD", "type": "promoter", "name": "pBAD (arabinose-inducible)", "sequence": None},
                {"id": "rbs_t7", "type": "rbs", "name": "RBS for T7 RNAP", "sequence": None},
                {"id": "t7rnap", "type": "cds", "name": "T7 RNA Polymerase", "sequence": None},
                {"id": "ter_t7", "type": "terminator", "name": "T7 RNAP Terminator", "sequence": None},
                {"id": "pLac", "type": "promoter", "name": "pLac (IPTG-inducible)", "sequence": None},
                {"id": "rbs_sigma", "type": "rbs", "name": "RBS for sigma factor", "sequence": None},
                {"id": "sigma", "type": "cds", "name": "Sigma factor", "sequence": None},
                {"id": "ter_sigma", "type": "terminator", "name": "Sigma Terminator", "sequence": None},
                {"id": "pT7", "type": "promoter", "name": "T7 Promoter", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pBAD", "to": "t7rnap"},
                {"type": "transcription", "from": "pLac", "to": "sigma"},
                {"type": "activation", "from": "t7rnap", "to": "pT7"},
                {"type": "transcription", "from": "pT7", "to": "gfp"}
            ]
        }
    ),
    (
        "A genetic NOR gate: two repressors, LacI and TetR, each repress a common promoter driving GFP. GFP is only ON when neither IPTG nor aTc is present.",
        {
            "name": "NOR_Gate",
            "description": "Genetic NOR gate where GFP is expressed only when neither input is present.",
            "components": [
                {"id": "pConst1", "type": "promoter", "name": "Constitutive Promoter 1", "sequence": None},
                {"id": "rbs_laci", "type": "rbs", "name": "RBS for LacI", "sequence": None},
                {"id": "lacI", "type": "cds", "name": "LacI", "sequence": None},
                {"id": "ter_laci", "type": "terminator", "name": "LacI Terminator", "sequence": None},
                {"id": "pConst2", "type": "promoter", "name": "Constitutive Promoter 2", "sequence": None},
                {"id": "rbs_tetr", "type": "rbs", "name": "RBS for TetR", "sequence": None},
                {"id": "tetR", "type": "cds", "name": "TetR", "sequence": None},
                {"id": "ter_tetr", "type": "terminator", "name": "TetR Terminator", "sequence": None},
                {"id": "pHybrid", "type": "promoter", "name": "Hybrid Promoter (LacI+TetR repressible)", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst1", "to": "lacI"},
                {"type": "transcription", "from": "pConst2", "to": "tetR"},
                {"type": "repression", "from": "lacI", "to": "pHybrid"},
                {"type": "repression", "from": "tetR", "to": "pHybrid"},
                {"type": "transcription", "from": "pHybrid", "to": "gfp"}
            ]
        }
    ),
    (
        "A genetic NOT gate (inverter) where TetR constitutively represses GFP under a pTet promoter. Adding aTc relieves repression.",
        {
            "name": "NOT_Gate_Inverter",
            "description": "Genetic inverter: TetR represses GFP; aTc derepresses.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "rbs_tetr", "type": "rbs", "name": "RBS for TetR", "sequence": None},
                {"id": "tetR", "type": "cds", "name": "TetR", "sequence": None},
                {"id": "ter_tetr", "type": "terminator", "name": "TetR Terminator", "sequence": None},
                {"id": "pTet", "type": "promoter", "name": "pTet Promoter", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "tetR"},
                {"type": "repression", "from": "tetR", "to": "pTet"},
                {"type": "transcription", "from": "pTet", "to": "gfp"}
            ]
        }
    ),
    (
        "A genetic OR gate where GFP is expressed when either arabinose OR IPTG is present. pBAD and pLac both independently drive GFP expression.",
        {
            "name": "OR_Gate",
            "description": "Genetic OR gate: GFP expressed when either arabinose or IPTG is present.",
            "components": [
                {"id": "pBAD", "type": "promoter", "name": "pBAD (arabinose-inducible)", "sequence": None},
                {"id": "rbs_gfp1", "type": "rbs", "name": "RBS for GFP copy 1", "sequence": None},
                {"id": "gfp1", "type": "cds", "name": "GFP (copy 1)", "sequence": None},
                {"id": "ter_gfp1", "type": "terminator", "name": "GFP Terminator 1", "sequence": None},
                {"id": "pLac", "type": "promoter", "name": "pLac (IPTG-inducible)", "sequence": None},
                {"id": "rbs_gfp2", "type": "rbs", "name": "RBS for GFP copy 2", "sequence": None},
                {"id": "gfp2", "type": "cds", "name": "GFP (copy 2)", "sequence": None},
                {"id": "ter_gfp2", "type": "terminator", "name": "GFP Terminator 2", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pBAD", "to": "gfp1"},
                {"type": "transcription", "from": "pLac", "to": "gfp2"}
            ]
        }
    ),

    # === Oscillators ===
    (
        "A repressilator with three genes: lacI represses tetR, tetR represses cI, and cI represses lacI, forming a cycle that produces oscillations.",
        {
            "name": "Repressilator",
            "description": "Three-gene negative feedback oscillator: lacI → tetR → cI → lacI.",
            "components": [
                {"id": "pCI", "type": "promoter", "name": "cI-repressible Promoter", "sequence": None},
                {"id": "rbs_laci", "type": "rbs", "name": "RBS for LacI", "sequence": None},
                {"id": "lacI", "type": "cds", "name": "LacI", "sequence": None},
                {"id": "ter_laci", "type": "terminator", "name": "LacI Terminator", "sequence": None},
                {"id": "pLac", "type": "promoter", "name": "LacI-repressible Promoter", "sequence": None},
                {"id": "rbs_tetr", "type": "rbs", "name": "RBS for TetR", "sequence": None},
                {"id": "tetR", "type": "cds", "name": "TetR", "sequence": None},
                {"id": "ter_tetr", "type": "terminator", "name": "TetR Terminator", "sequence": None},
                {"id": "pTet", "type": "promoter", "name": "TetR-repressible Promoter", "sequence": None},
                {"id": "rbs_ci", "type": "rbs", "name": "RBS for cI", "sequence": None},
                {"id": "cI", "type": "cds", "name": "cI Repressor", "sequence": None},
                {"id": "ter_ci", "type": "terminator", "name": "cI Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pCI", "to": "lacI"},
                {"type": "repression", "from": "lacI", "to": "pLac"},
                {"type": "transcription", "from": "pLac", "to": "tetR"},
                {"type": "repression", "from": "tetR", "to": "pTet"},
                {"type": "transcription", "from": "pTet", "to": "cI"},
                {"type": "repression", "from": "cI", "to": "pCI"}
            ]
        }
    ),
    (
        "A dual-feedback oscillator where an activator A and a repressor R form a loop: A activates both itself and R, while R represses A.",
        {
            "name": "Dual_Feedback_Oscillator",
            "description": "Oscillator with positive and negative feedback loops.",
            "components": [
                {"id": "pA", "type": "promoter", "name": "Promoter for Activator A", "sequence": None},
                {"id": "rbs_a", "type": "rbs", "name": "RBS for Activator A", "sequence": None},
                {"id": "actA", "type": "cds", "name": "Activator A", "sequence": None},
                {"id": "ter_a", "type": "terminator", "name": "Activator A Terminator", "sequence": None},
                {"id": "pR", "type": "promoter", "name": "Promoter for Repressor R", "sequence": None},
                {"id": "rbs_r", "type": "rbs", "name": "RBS for Repressor R", "sequence": None},
                {"id": "repR", "type": "cds", "name": "Repressor R", "sequence": None},
                {"id": "ter_r", "type": "terminator", "name": "Repressor R Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "activation", "from": "actA", "to": "pA"},
                {"type": "activation", "from": "actA", "to": "pR"},
                {"type": "transcription", "from": "pA", "to": "actA"},
                {"type": "transcription", "from": "pR", "to": "repR"},
                {"type": "repression", "from": "repR", "to": "pA"}
            ]
        }
    ),

    # === Biosensors ===
    (
        "An arsenic biosensor where the ArsR protein normally represses GFP expression. When arsenic is present, it binds ArsR and relieves repression, turning on GFP.",
        {
            "name": "Arsenic_Biosensor",
            "description": "Arsenic biosensor using ArsR derepression to activate GFP.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter for ArsR", "sequence": None},
                {"id": "rbs_arsr", "type": "rbs", "name": "RBS for ArsR", "sequence": None},
                {"id": "arsR", "type": "cds", "name": "ArsR", "sequence": None},
                {"id": "ter_arsr", "type": "terminator", "name": "ArsR Terminator", "sequence": None},
                {"id": "pArs", "type": "promoter", "name": "Ars Promoter (ArsR-repressible)", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "arsR"},
                {"type": "repression", "from": "arsR", "to": "pArs"},
                {"type": "transcription", "from": "pArs", "to": "gfp"}
            ]
        }
    ),
    (
        "A mercury biosensor with signal amplification. MerR activates a promoter driving T7 RNAP when mercury is present. T7 RNAP then drives high-level GFP expression from a T7 promoter.",
        {
            "name": "Mercury_Biosensor_Amplified",
            "description": "Mercury biosensor with T7 RNAP amplification for high GFP output.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter for MerR", "sequence": None},
                {"id": "rbs_merr", "type": "rbs", "name": "RBS for MerR", "sequence": None},
                {"id": "merR", "type": "cds", "name": "MerR", "sequence": None},
                {"id": "ter_merr", "type": "terminator", "name": "MerR Terminator", "sequence": None},
                {"id": "pMer", "type": "promoter", "name": "Mercury-responsive Promoter", "sequence": None},
                {"id": "rbs_t7", "type": "rbs", "name": "RBS for T7 RNAP", "sequence": None},
                {"id": "t7rnap", "type": "cds", "name": "T7 RNA Polymerase", "sequence": None},
                {"id": "ter_t7", "type": "terminator", "name": "T7 RNAP Terminator", "sequence": None},
                {"id": "pT7", "type": "promoter", "name": "T7 Promoter", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "merR"},
                {"type": "activation", "from": "merR", "to": "pMer"},
                {"type": "transcription", "from": "pMer", "to": "t7rnap"},
                {"type": "activation", "from": "t7rnap", "to": "pT7"},
                {"type": "transcription", "from": "pT7", "to": "gfp"}
            ]
        }
    ),
    (
        "A quorum sensing biosensor where LuxR detects AHL and activates pLux to drive mCherry expression.",
        {
            "name": "Quorum_Sensing_Biosensor",
            "description": "AHL quorum sensing biosensor with mCherry reporter.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "rbs_luxr", "type": "rbs", "name": "RBS for LuxR", "sequence": None},
                {"id": "luxR", "type": "cds", "name": "LuxR", "sequence": None},
                {"id": "ter_luxr", "type": "terminator", "name": "LuxR Terminator", "sequence": None},
                {"id": "pLux", "type": "promoter", "name": "pLux Promoter (AHL-responsive)", "sequence": None},
                {"id": "rbs_mch", "type": "rbs", "name": "RBS for mCherry", "sequence": None},
                {"id": "mcherry", "type": "cds", "name": "mCherry", "sequence": None},
                {"id": "ter_mch", "type": "terminator", "name": "mCherry Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "luxR"},
                {"type": "activation", "from": "luxR", "to": "pLux"},
                {"type": "transcription", "from": "pLux", "to": "mcherry"}
            ]
        }
    ),

    # === Metabolic Pathways ===
    (
        "A lycopene biosynthesis pathway with three enzymes: CrtE converts GGPP to phytoene, CrtB converts phytoene to phytofluene, and CrtI converts phytofluene to lycopene. All expressed from a single operon.",
        {
            "name": "Lycopene_Pathway",
            "description": "Three-enzyme lycopene biosynthesis operon.",
            "components": [
                {"id": "pTrc", "type": "promoter", "name": "Trc Promoter", "sequence": None},
                {"id": "rbs_crte", "type": "rbs", "name": "RBS for CrtE", "sequence": None},
                {"id": "crtE", "type": "cds", "name": "CrtE (GGPP synthase)", "sequence": None},
                {"id": "rbs_crtb", "type": "rbs", "name": "RBS for CrtB", "sequence": None},
                {"id": "crtB", "type": "cds", "name": "CrtB (Phytoene synthase)", "sequence": None},
                {"id": "rbs_crti", "type": "rbs", "name": "RBS for CrtI", "sequence": None},
                {"id": "crtI", "type": "cds", "name": "CrtI (Phytoene desaturase)", "sequence": None},
                {"id": "ter_operon", "type": "terminator", "name": "Operon Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pTrc", "to": "crtE"},
                {"type": "transcription", "from": "pTrc", "to": "crtB"},
                {"type": "transcription", "from": "pTrc", "to": "crtI"}
            ]
        }
    ),
    (
        "A violacein biosynthesis pathway with five enzymes: VioA, VioB, VioC, VioD, and VioE expressed from two separate operons under pBAD and pTet promoters respectively.",
        {
            "name": "Violacein_Pathway",
            "description": "Five-enzyme violacein biosynthesis split across two operons.",
            "components": [
                {"id": "pBAD", "type": "promoter", "name": "pBAD Promoter", "sequence": None},
                {"id": "rbs_vioa", "type": "rbs", "name": "RBS for VioA", "sequence": None},
                {"id": "vioA", "type": "cds", "name": "VioA", "sequence": None},
                {"id": "rbs_viob", "type": "rbs", "name": "RBS for VioB", "sequence": None},
                {"id": "vioB", "type": "cds", "name": "VioB", "sequence": None},
                {"id": "rbs_vioe", "type": "rbs", "name": "RBS for VioE", "sequence": None},
                {"id": "vioE", "type": "cds", "name": "VioE", "sequence": None},
                {"id": "ter1", "type": "terminator", "name": "Operon 1 Terminator", "sequence": None},
                {"id": "pTet", "type": "promoter", "name": "pTet Promoter", "sequence": None},
                {"id": "rbs_vioc", "type": "rbs", "name": "RBS for VioC", "sequence": None},
                {"id": "vioC", "type": "cds", "name": "VioC", "sequence": None},
                {"id": "rbs_viod", "type": "rbs", "name": "RBS for VioD", "sequence": None},
                {"id": "vioD", "type": "cds", "name": "VioD", "sequence": None},
                {"id": "ter2", "type": "terminator", "name": "Operon 2 Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pBAD", "to": "vioA"},
                {"type": "transcription", "from": "pBAD", "to": "vioB"},
                {"type": "transcription", "from": "pBAD", "to": "vioE"},
                {"type": "transcription", "from": "pTet", "to": "vioC"},
                {"type": "transcription", "from": "pTet", "to": "vioD"}
            ]
        }
    ),
    (
        "A butanol production pathway: acetyl-CoA is converted through four enzymatic steps by AtoB, Hbd, Crt, and AdhE2 to produce butanol. All genes under a single IPTG-inducible promoter.",
        {
            "name": "Butanol_Pathway",
            "description": "Four-enzyme butanol biosynthesis from acetyl-CoA under IPTG control.",
            "components": [
                {"id": "pLac", "type": "promoter", "name": "pLac (IPTG-inducible)", "sequence": None},
                {"id": "rbs_atob", "type": "rbs", "name": "RBS for AtoB", "sequence": None},
                {"id": "atoB", "type": "cds", "name": "AtoB (thiolase)", "sequence": None},
                {"id": "rbs_hbd", "type": "rbs", "name": "RBS for Hbd", "sequence": None},
                {"id": "hbd", "type": "cds", "name": "Hbd (3-hydroxybutyryl-CoA dehydrogenase)", "sequence": None},
                {"id": "rbs_crt", "type": "rbs", "name": "RBS for Crt", "sequence": None},
                {"id": "crt", "type": "cds", "name": "Crt (crotonase)", "sequence": None},
                {"id": "rbs_adhe2", "type": "rbs", "name": "RBS for AdhE2", "sequence": None},
                {"id": "adhE2", "type": "cds", "name": "AdhE2 (aldehyde-alcohol dehydrogenase)", "sequence": None},
                {"id": "ter", "type": "terminator", "name": "Operon Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pLac", "to": "atoB"},
                {"type": "transcription", "from": "pLac", "to": "hbd"},
                {"type": "transcription", "from": "pLac", "to": "crt"},
                {"type": "transcription", "from": "pLac", "to": "adhE2"}
            ]
        }
    ),

    # === Mixed / Advanced ===
    (
        "A genetic toggle switch with two mutually repressing genes: LacI represses the pTet promoter driving TetR, and TetR represses the pLac promoter driving LacI. A GFP reporter is fused to LacI.",
        {
            "name": "Toggle_Switch_GFP",
            "description": "Bistable toggle switch with GFP reporter fused to LacI.",
            "components": [
                {"id": "pTet", "type": "promoter", "name": "pTet Promoter", "sequence": None},
                {"id": "rbs_laci", "type": "rbs", "name": "RBS for LacI", "sequence": None},
                {"id": "lacI", "type": "cds", "name": "LacI", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP (fused to LacI)", "sequence": None},
                {"id": "ter_laci", "type": "terminator", "name": "LacI-GFP Terminator", "sequence": None},
                {"id": "pLac", "type": "promoter", "name": "pLac Promoter", "sequence": None},
                {"id": "rbs_tetr", "type": "rbs", "name": "RBS for TetR", "sequence": None},
                {"id": "tetR", "type": "cds", "name": "TetR", "sequence": None},
                {"id": "ter_tetr", "type": "terminator", "name": "TetR Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pTet", "to": "lacI"},
                {"type": "repression", "from": "lacI", "to": "pLac"},
                {"type": "transcription", "from": "pLac", "to": "tetR"},
                {"type": "repression", "from": "tetR", "to": "pTet"}
            ]
        }
    ),
    (
        "A band-pass detector that only activates GFP at intermediate concentrations of AHL. Low AHL: LuxR inactive. High AHL: LuxR activates both GFP and a repressor that shuts GFP off.",
        {
            "name": "Band_Pass_Detector",
            "description": "AHL band-pass filter: GFP on only at intermediate AHL levels.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "rbs_luxr", "type": "rbs", "name": "RBS for LuxR", "sequence": None},
                {"id": "luxR", "type": "cds", "name": "LuxR", "sequence": None},
                {"id": "ter_luxr", "type": "terminator", "name": "LuxR Terminator", "sequence": None},
                {"id": "pLux", "type": "promoter", "name": "pLux (AHL-responsive)", "sequence": None},
                {"id": "rbs_gfp", "type": "rbs", "name": "RBS for GFP", "sequence": None},
                {"id": "gfp", "type": "cds", "name": "GFP", "sequence": None},
                {"id": "ter_gfp", "type": "terminator", "name": "GFP Terminator", "sequence": None},
                {"id": "pLux2", "type": "promoter", "name": "pLux High (high-threshold AHL-responsive)", "sequence": None},
                {"id": "rbs_rep", "type": "rbs", "name": "RBS for Repressor", "sequence": None},
                {"id": "repressor", "type": "cds", "name": "LacI Repressor", "sequence": None},
                {"id": "ter_rep", "type": "terminator", "name": "Repressor Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "luxR"},
                {"type": "activation", "from": "luxR", "to": "pLux"},
                {"type": "activation", "from": "luxR", "to": "pLux2"},
                {"type": "transcription", "from": "pLux", "to": "gfp"},
                {"type": "transcription", "from": "pLux2", "to": "repressor"},
                {"type": "repression", "from": "repressor", "to": "pLux"}
            ]
        }
    ),
    (
        "A population-level pulse generator using quorum sensing. LuxI produces AHL which accumulates. When AHL reaches threshold, LuxR-AHL activates a lysis gene that kills the cell, releasing contents and resetting AHL levels.",
        {
            "name": "Population_Pulse_Generator",
            "description": "Quorum-sensing synchronized lysis circuit for periodic payload release.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "rbs_luxi", "type": "rbs", "name": "RBS for LuxI", "sequence": None},
                {"id": "luxI", "type": "cds", "name": "LuxI (AHL synthase)", "sequence": None},
                {"id": "ter_luxi", "type": "terminator", "name": "LuxI Terminator", "sequence": None},
                {"id": "pConst2", "type": "promoter", "name": "Constitutive Promoter for LuxR", "sequence": None},
                {"id": "rbs_luxr", "type": "rbs", "name": "RBS for LuxR", "sequence": None},
                {"id": "luxR", "type": "cds", "name": "LuxR", "sequence": None},
                {"id": "ter_luxr", "type": "terminator", "name": "LuxR Terminator", "sequence": None},
                {"id": "pLux", "type": "promoter", "name": "pLux (AHL-responsive)", "sequence": None},
                {"id": "rbs_lysis", "type": "rbs", "name": "RBS for Lysis Gene", "sequence": None},
                {"id": "lysisE", "type": "cds", "name": "Lysis Gene E", "sequence": None},
                {"id": "ter_lysis", "type": "terminator", "name": "Lysis Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "luxI"},
                {"type": "transcription", "from": "pConst2", "to": "luxR"},
                {"type": "activation", "from": "luxR", "to": "pLux"},
                {"type": "transcription", "from": "pLux", "to": "lysisE"}
            ]
        }
    ),
    (
        "A light-inducible gene expression system where blue light activates the LOV domain protein EL222, which dimerizes and binds the pBLind promoter to drive YFP expression.",
        {
            "name": "Light_Inducible_YFP",
            "description": "Blue light-inducible YFP expression via EL222/LOV system.",
            "components": [
                {"id": "pConst", "type": "promoter", "name": "Constitutive Promoter", "sequence": None},
                {"id": "rbs_el222", "type": "rbs", "name": "RBS for EL222", "sequence": None},
                {"id": "el222", "type": "cds", "name": "EL222 (LOV domain protein)", "sequence": None},
                {"id": "ter_el222", "type": "terminator", "name": "EL222 Terminator", "sequence": None},
                {"id": "pBLind", "type": "promoter", "name": "pBLind (blue light-responsive)", "sequence": None},
                {"id": "rbs_yfp", "type": "rbs", "name": "RBS for YFP", "sequence": None},
                {"id": "yfp", "type": "cds", "name": "YFP", "sequence": None},
                {"id": "ter_yfp", "type": "terminator", "name": "YFP Terminator", "sequence": None}
            ],
            "interactions": [
                {"type": "transcription", "from": "pConst", "to": "el222"},
                {"type": "activation", "from": "el222", "to": "pBLind"},
                {"type": "transcription", "from": "pBLind", "to": "yfp"}
            ]
        }
    ),
]


SYSTEM_PROMPT = """You are an expert synthetic biologist. Given a natural language description of a genetic circuit, output a JSON object describing the circuit components and interactions.

The JSON must follow this schema:
{
  "name": "string — circuit name",
  "description": "string — brief description",
  "components": [
    {
      "id": "string — unique identifier",
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

Respond with valid JSON only, no explanation."""


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
    out_dir = Path(__file__).parent

    # Split: 80% train, 10% valid, 10% test
    n = len(TRAINING_PAIRS)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train.jsonl": TRAINING_PAIRS[:train_end],
        "valid.jsonl": TRAINING_PAIRS[train_end:valid_end],
        "test.jsonl": TRAINING_PAIRS[valid_end:],
    }

    for filename, pairs in splits.items():
        path = out_dir / filename
        with open(path, "w") as f:
            for desc, circuit in pairs:
                entry = build_chat_entry(desc, circuit)
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(pairs)} examples to {path}")


if __name__ == "__main__":
    main()

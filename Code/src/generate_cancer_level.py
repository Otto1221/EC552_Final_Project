"""
Generate cancer-level therapeutic circuit examples targeting dataset gaps.
Focus: multi-module designs with dense interaction networks, heavy activation/repression,
30+ components, 20+ interactions per circuit.

Targets weak areas: drug delivery, inflammation, aging, wound healing, antibodies,
closed-loop control, combination therapies, and advanced CAR-T/cell therapy.
"""

import json
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Every description is engineered for MAXIMUM complexity:
# - 30-60+ components
# - 20-40+ interactions
# - Heavy on activation AND repression (the model's weak spot)
# - Multi-module with cross-module regulation
# ---------------------------------------------------------------------------

CANCER_LEVEL_CIRCUITS = [

    # === ADVANCED MULTI-MODULE CAR-T (the holy grail) ===

    "A fifth-generation CAR-T with five integrated modules for solid tumor killing. Module 1: an anti-GD2 CAR (scFv-CD28-4-1BB-CD3zeta-IL2Rbeta fusion) under EF1a promoter constitutively arms the T-cell — the IL2Rbeta domain triggers JAK/STAT signaling upon antigen engagement. Module 2: an NFAT-responsive promoter (activated by CAR signaling) drives a bicistronic cassette of IL-15 superagonist (ALT-803) and anti-PD-1 scFv via P2A peptide — simultaneously boosting T-cell persistence and blocking checkpoint. Module 3: a hypoxia-responsive HRE promoter drives VEGF-Trap (soluble VEGFR decoy) to starve the tumor vasculature — this only activates in the hypoxic tumor core. Module 4: constitutive PGK promoter drives a dominant-negative TGF-beta receptor II (dnTGFBRII) fused to a 4-1BB costimulatory domain via a flexible linker — converting immunosuppressive TGF-beta into a costimulatory signal. Module 5: a rapamycin-inducible safety system — constitutive expression of FKBP-iCasp9 is dimerized by rapamycin, triggering apoptosis. Each module has its own RBS and terminator. An mCherry reporter under NFAT monitors activation.",

    "A logic-gated dual-checkpoint-resistant CAR-T for pancreatic cancer. Module 1: a synNotch receptor with anti-mesothelin scFv under EF1a promoter detects tumor cells. Cleavage releases Gal4-VP64 which activates a 5xUAS promoter. Module 2: UAS drives an anti-HER2 CAR (scFv-CD28-CD3zeta) — the T-cell only arms against HER2 when mesothelin is present (AND gate). Module 3: UAS also drives CRISPR-Cas9 with two U6-driven gRNAs targeting PD-1 (PDCD1) and LAG-3 (LAG3) simultaneously — checkpoint genes are knocked out only in tumor-proximal cells. Module 4: NFAT-responsive promoter drives secreted IL-12p70 (heterodimer of p35 and p40 via IRES) for immune potentiation and M1 macrophage polarization. Module 5: constitutive PGK promoter drives CD47 surface expression for don't-eat-me protection from host macrophages. Module 6: an AP1903-inducible iCasp9 safety switch. GFP reporter under UAS tracks spatial activation.",

    "A self-amplifying CAR-T with metabolic armor for glioblastoma. Module 1: anti-EGFRvIII CAR (scFv-IgG4hinge-CD28TM-4-1BB-CD3zeta) driven by MSCV promoter. Module 2: NFAT-responsive promoter drives a positive feedback cassette — IL-7 and CCL19 chemokine via T2A peptide, recruiting additional T-cells AND enhancing survival (IL-7 feeds back to JAK/STAT → more NFAT activity, creating amplification). Module 3: constitutive EF1a drives MCT1 (monocarboxylate transporter 1, which imports lactate as fuel) and GPX4 (glutathione peroxidase, which resists ferroptosis) via P2A — metabolic reprogramming for the harsh tumor microenvironment. Module 4: a TGF-beta-responsive SMAD-binding element drives a decoy IL-6 receptor that sequesters tumor-derived IL-6, blocking STAT3-mediated immunosuppression. Module 5: the CAR construct includes a truncated CD34 tag for magnetic bead selection. Module 6: a tetracycline-inducible TRE promoter drives Fas ligand for activation-induced death as a safety mechanism. Module 7: constitutive PGK drives an shRNA cassette targeting PTPN2 phosphatase to enhance TCR/CAR signaling sensitivity.",

    "An allogeneic off-the-shelf universal CAR-T with immune evasion. Module 1: anti-CD19 CAR (FMC63-CD8hinge-CD28-CD3zeta) under EF1a promoter. Module 2: CRISPR multiplex editing — Cas9 under CMV promoter with four U6-driven gRNAs: gRNA-1 knocks out TRAC (prevents GvHD), gRNA-2 knocks out B2M (eliminates MHC-I to avoid host rejection), gRNA-3 knocks out CIITA (eliminates MHC-II), gRNA-4 knocks out PD-1. Module 3: constitutive PGK drives HLA-E (non-classical MHC that inhibits NK cell killing via NKG2A). Module 4: constitutive SV40 drives CD47 (don't-eat-me signal blocks macrophage phagocytosis). Module 5: NFAT-responsive promoter drives IL-21 and CCL21 via P2A (immune recruitment only during active killing). Module 6: constitutive expression of a single-domain anti-CD52 antibody fused to a GPI anchor for surface display — enables alemtuzumab-mediated depletion of host T/NK cells that might reject the graft. Module 7: AP1903-inducible iCasp9 kill switch.",

    # === ENGINEERED BACTERIA FOR CANCER ===

    "A triple-payload tumor-colonizing bacterium with synchronized release and immune activation. Chassis: E. coli Nissle 1917 with three therapeutic arms and two sensors. Sensor 1: LuxI/LuxR quorum sensing — constitutive pJ23106 drives LuxI producing AHL. At tumor-density threshold, AHL-LuxR activates pLux. Sensor 2: a hypoxia sensor — FNR-responsive pFNR promoter activates only in anaerobic tumor cores. Arm 1: pLux drives phiX174 lysis gene E — at quorum threshold, cells lyse and release intracellular payloads. Arm 2: constitutive pJ23119 drives intracellular accumulation of anti-CTLA4 nanobody (VHH, no secretion signal — released upon lysis). Arm 3: pFNR AND pLux dual-input promoter (synthetic AND gate requiring both hypoxia AND quorum) drives theta-defensin retrocyclin (antimicrobial + immunostimulatory peptide). Module 4: constitutive expression of ClyA pore-forming toxin with a cleavable periplasmic signal peptide — slowly leaks during growth, massively released during lysis. Module 5: an arabinose-inducible pBAD drives MazF toxin as external kill switch. Module 6: a DAP auxotrophy (dapA deletion) ensures death outside the tumor (where DAP is not supplemented). A GFP reporter under pLux tracks oscillation cycles. An mCherry reporter under pFNR confirms hypoxia.",

    "An engineered Salmonella VNP20009 for combination immunotherapy with genetic safeguards. Module 1: a constitutive pTac drives flagellin B (FljB) — a TLR5 agonist that activates innate immunity in the tumor microenvironment. Module 2: a hypoxia-responsive pepT promoter activates a Flp recombinase that excises a FRT-flanked transcriptional stop cassette, permanently activating a downstream pConst driving anti-CD47 nanobody-OmpA fusion for surface display — blocking the don't-eat-me signal only in hypoxic tumor regions. Module 3: a quorum sensing module (EsaI/EsaR with AHL) where high density activates pEsa driving STING agonist (cyclic di-GMP synthase DGC) — triggering type I interferon in tumor-associated dendritic cells. Module 4: a second QS circuit (LasI/LasR, orthogonal) drives lysis protein SRRz from phage lambda for periodic population reset and payload release. Module 5: a temperature-sensitive CI857 repressor constitutively expressed represses pR-driven CcdB toxin — cells die if they escape body temperature. Module 6: an essential gene thyA is on a plasmid under IPTG control — thymidine auxotrophy without inducer. Module 7: constitutive chromosomal lacZ-alpha fusion with beta-galactosidase reporter for blue/white colony screening. Antibiotic-free selection via an operator-repressor titration system.",

    # === CLOSED-LOOP DRUG DELIVERY CIRCUITS ===

    "A closed-loop smart drug delivery circuit for doxorubicin-resistant breast cancer. Module 1: engineered HEK293 cells encapsulated in alginate beads. A synthetic estrogen receptor element (ERE) promoter senses circulating estradiol levels — high estradiol (indicating ER+ tumor activity) activates the circuit. ERE drives expression of a P450 enzyme CYP3A4 that converts an inactive cyclophosphamide prodrug into its active form (4-hydroxycyclophosphamide) locally. Module 2: an NF-kB-responsive promoter (activated by tumor-secreted TNF-alpha and IL-1beta in the microenvironment) drives expression of a bispecific T-cell engager (BiTE) with anti-HER2 x anti-CD3 specificity, recruiting endogenous T-cells. Module 3: a negative feedback loop — the same NF-kB promoter also drives expression of IL-10 (anti-inflammatory) which feeds back to dampen NF-kB via the IL-10R-JAK1-STAT3 pathway, preventing cytokine storm. Module 4: a hypoxia-responsive HRE promoter drives angiostatin (anti-angiogenic) to starve the tumor. Module 5: a constitutive CMV promoter drives herpes simplex thymidine kinase (HSV-TK) as a safety switch — ganciclovir administration kills the cells. Module 6: a tetracycline-inducible TRE promoter drives secreted GFP-luciferase fusion for non-invasive bioluminescence imaging. Each module has its own poly-A signal and insulator elements (cHS4).",

    "A nanoparticle-based programmable drug release circuit for metastatic melanoma. Module 1: a tumor-homing peptide iRGD is displayed on an engineered M13 phage coat protein. Module 2: the phage carries a genetic circuit — a survivin promoter (active only in cancer cells) drives T7 RNA polymerase for signal amplification. Module 3: T7 promoter drives a cytosine deaminase (CD) that converts 5-fluorocytosine (5-FC) prodrug to 5-fluorouracil (5-FU) chemotherapy locally within the tumor. Module 4: T7 also drives a secreted anti-VEGF nanobody (bevacizumab mimic) for anti-angiogenic therapy. Module 5: T7 drives a TRAIL (TNF-related apoptosis-inducing ligand) for selective cancer cell apoptosis via death receptors DR4/DR5. Module 6: a negative feedback timer — T7 also drives LacI which slowly accumulates and represses a lacO operator upstream of the T7 RNAP gene, creating a self-limiting circuit that shuts down after ~48 hours to prevent off-target toxicity. Module 7: T7 drives a secreted NanoLuc luciferase as a blood-detectable pharmacodynamic biomarker. Each therapeutic gene has its own RBS, and all coding sequences include optimized Kozak sequences for mammalian expression.",

    "A pH-responsive targeted drug delivery circuit for acidic tumor microenvironments. Module 1: engineered E. coli with a CadC acid sensor — at pH < 6.5 (tumor acidity), CadC activates the cadBA promoter driving a master activator AraC-VP16 fusion. Module 2: AraC-VP16 activates pBAD-VP16 driving diphtheria toxin fragment A (DTA) — a potent protein synthesis inhibitor that kills surrounding tumor cells (bystander effect). Module 3: AraC-VP16 also activates pBAD-VP16 driving a heparin-binding EGF-like growth factor (HB-EGF) decoy receptor — blocking tumor autocrine growth signaling. Module 4: constitutive pJ23119 drives surface display of a ClyA-anti-EGFR affibody fusion for tumor cell binding. Module 5: a neutral pH sensor (CpxR, active at pH 7.4) drives expression of MazF toxin — cells that escape the acidic tumor into normal tissue are killed. Module 6: an IPTG-inducible secondary kill switch with CcdB. Module 7: a DAP auxotrophy for environmental containment. GFP reporter under CadC-responsive promoter and mCherry under CpxR-responsive promoter provide dual pH visualization.",

    # === INFLAMMATION & AUTOIMMUNE CIRCUITS ===

    "A multi-sensor anti-inflammatory circuit for rheumatoid arthritis. Module 1: engineered mesenchymal stem cells (MSCs) with a TNF-alpha-responsive NF-kB promoter (8xNF-kB binding sites) driving IL-1 receptor antagonist (IL-1Ra) — directly blocking the IL-1 inflammatory cascade. Module 2: the same NF-kB promoter drives a soluble TNF receptor II (sTNFRII-Fc fusion) decoy that sequesters excess TNF-alpha — creating negative feedback (less TNF → less NF-kB → less decoy production, self-regulating). Module 3: an IL-6-responsive STAT3 element drives expression of sgp130Fc (soluble gp130 decoy that specifically blocks IL-6 trans-signaling while preserving classical signaling). Module 4: a constitutive EF1a promoter drives IDO1 (indoleamine 2,3-dioxygenase) which depletes tryptophan locally, suppressing effector T-cell proliferation. Module 5: a hypoxia-responsive HRE promoter drives VEGF for tissue repair — only active in damaged hypoxic joints. Module 6: a mechanical stress-responsive promoter (from the TRPV4 mechanosensitive channel gene) drives lubricin (PRG4) secretion for joint lubrication. Module 7: a tetracycline-OFF system — doxycycline removes tTA from TRE-driven BMP-7, stopping cartilage regeneration when sufficient. A GFP reporter under NF-kB and a BFP reporter under HRE track inflammation and hypoxia states.",

    "A closed-loop cytokine storm prevention circuit for sepsis. Module 1: engineered macrophages with an NF-kB-responsive promoter driving IL-10 (anti-inflammatory) — activated by TNF-alpha and IL-1beta. Module 2: a STAT1-responsive promoter (activated by IFN-gamma) drives PD-L1 surface expression — suppressing hyperactive T-cells in the vicinity. Module 3: a dual-input AND gate — both NF-kB AND STAT1 must be active (indicating severe inflammation) to drive expression of a soluble IL-6R-Fc decoy AND sTNFRII-Fc via bicistronic P2A cassette. Module 4: a negative feedback controller — IL-10 produced by Module 1 activates STAT3, which drives a STAT3-responsive promoter expressing SOCS3 (suppressor of cytokine signaling 3) that feeds back to dampen NF-kB and STAT1 signaling, creating homeostatic regulation. Module 5: constitutive PGK drives A20 (TNFAIP3, a ubiquitin-editing enzyme that terminates NF-kB signaling) providing baseline anti-inflammatory tone. Module 6: a severity-gated apoptosis circuit — very high NF-kB (indicating uncontrollable inflammation) drives FasL expression, causing the engineered macrophage to kill neighboring hyper-inflammatory cells. Module 7: iCasp9 under constitutive promoter for AP1903-mediated safety shutdown. mCherry under NF-kB and GFP under STAT3 for dual reporter monitoring.",

    "An engineered Treg circuit for inflammatory bowel disease with gut homing and mucosal healing. Module 1: CD4+ T-cells transduced with a lentiviral construct. EF1a promoter drives FOXP3 (master Treg transcription factor) with a positive feedback loop — FOXP3 binds its own CNS2 enhancer element, creating stable Treg commitment. Module 2: constitutive PGK drives CCR9 (chemokine receptor for gut homing to CCL25+ intestinal epithelium) and integrin alpha4beta7 (for MAdCAM-1 binding on gut endothelium) via P2A peptide. Module 3: an NFAT-responsive promoter (activated by TCR engagement at inflammation sites) drives a triple therapeutic cassette: IL-10, TGF-beta3, and IL-35 (Ebi3-p35 heterodimer) via T2A and P2A peptides. Module 4: NFAT also drives KGF (keratinocyte growth factor) for epithelial barrier repair and trefoil factor TFF3 for mucosal healing. Module 5: constitutive expression of CTLA-4 at high levels for competitive inhibition of CD80/CD86 on antigen-presenting cells. Module 6: a tissue-damage sensor — a HMGB1-responsive promoter (recognizing damage-associated molecular patterns) drives additional IL-10 burst production. Module 7: a rapamycin-inducible iCasp9 for safety.",

    # === AGING / SENESCENCE CIRCUITS ===

    "A senolytic circuit that selectively destroys senescent cells in aging tissue. Module 1: a p16INK4a promoter (specifically active in senescent cells) drives a master activator — split T7 RNAP N-terminal fragment. Module 2: a p21/WAF1 promoter (another senescence marker) drives the T7 RNAP C-terminal fragment. Only cells expressing BOTH p16 AND p21 (confirmed senescent) produce functional T7 RNAP (AND gate). Module 3: T7 promoter drives a pro-apoptotic cassette — BIM (BCL2L11) and PUMA (BBC3) via P2A peptide, overwhelming anti-apoptotic BCL-2 family members. Module 4: T7 also drives secreted FOXO4-DRI peptide (disrupts FOXO4-p53 interaction in senescent cells, triggering p53-dependent apoptosis). Module 5: T7 drives a secreted GDF11 (growth differentiation factor 11, a rejuvenation factor) — surviving non-senescent neighboring cells receive pro-regenerative signals. Module 6: a negative safety gate — a constitutive CMV promoter drives BCL-XL (anti-apoptotic), but T7 drives a miR-targeting BCL-XL mRNA (miR-shRNA cassette), specifically removing the safety brake only in double-positive senescent cells. Module 7: a tetracycline-inducible HSV-TK suicide gene for external control. Module 8: T7 drives secreted NanoLuc for blood-based monitoring of senolytic activity.",

    "A telomere maintenance and cellular rejuvenation circuit for aging reversal. Module 1: a tetracycline-inducible TRE promoter drives hTERT (human telomerase reverse template) for controlled telomere extension — only active when doxycycline is administered. Module 2: a constitutive EF1a promoter drives TERC (telomerase RNA component) constitutively at low levels. Module 3: a p53-responsive promoter drives a dominant-negative p53 (p53DD) that transiently blocks senescence checkpoints during telomere extension — but includes an ssrA-equivalent mammalian PEST degradation tag for rapid clearance when p53 signaling normalizes. Module 4: a DNA damage-responsive promoter (p21/CDKN1A promoter) drives SIRT6 (NAD-dependent deacetylase involved in DNA repair and metabolic regulation) for enhanced genome maintenance. Module 5: constitutive PGK drives Yamanaka factors OCT4 and KLF4 (but NOT SOX2 or MYC to avoid full reprogramming/oncogenesis) at very low levels via weak RBS equivalents for partial epigenetic rejuvenation. Module 6: a myc-responsive promoter (E-box elements) drives a dominant-negative MYC (OmoMYC) as an oncogene safety brake — if any cell activates endogenous MYC, OmoMYC blocks it. Module 7: a constitutive CMV drives miR-302 cluster (represses cell cycle inhibitors and promotes repair). Module 8: an ARF/p14 promoter drives herpes thymidine kinase (HSV-TK) — if cells become oncogenic (ARF activation), ganciclovir kills them.",

    # === WOUND HEALING & REGENERATION ===

    "An advanced wound healing circuit with four-phase temporal programming. Module 1: Phase 1 (immediate, constitutive) — EF1a promoter drives antimicrobial peptide LL-37 (cathelicidin) and CXCL12 (SDF-1, recruits stem cells to wound) via P2A, providing immediate antimicrobial and stem cell recruitment. Module 2: Phase 2 (inflammation, first 24h) — an NF-kB-responsive promoter activated by wound DAMPs drives MCP-1 (CCL2, monocyte recruitment) and TNF-alpha (pro-inflammatory, M1 macrophage activation) via T2A. A delayed negative feedback: NF-kB also drives IL-10 with a slow-folding 5'UTR secondary structure that delays translation by ~12 hours, initiating the transition to anti-inflammatory phase. Module 3: Phase 3 (proliferation, day 2-7) — an IL-10-responsive STAT3 promoter (activated by Phase 2 IL-10 feedback) drives VEGF-A (angiogenesis), PDGF-BB (fibroblast proliferation), and KGF (keratinocyte migration) via triple P2A cassette. STAT3 also drives TGF-beta3 (scarless wound healing isoform, not TGF-beta1 which causes scarring). Module 4: Phase 4 (remodeling, day 7+) — a collagen-responsive promoter (activated by accumulating collagen matrix) drives MMP-2 (matrix metalloproteinase for collagen remodeling) and TIMP-1 (tissue inhibitor of metalloproteinases for balanced remodeling) with different RBS strengths (MMP-2 strong, TIMP-1 moderate) to achieve a 2:1 ratio. Module 5: an oxygen sensor — FIH (factor inhibiting HIF) responsive promoter tracks wound oxygenation recovery, driving GFP as a healing progress reporter. Module 6: a mechanical tension sensor — Piezo1 channel-responsive promoter drives decorin (anti-fibrotic proteoglycan that neutralizes TGF-beta1) specifically in high-tension scar areas. Module 7: HSV-TK suicide gene under constitutive promoter for ganciclovir-mediated cell elimination after healing.",

    "A diabetic wound healing circuit with infection sensing and glucose regulation. Module 1: engineered keratinocytes with a glucose-responsive ChREBP element driving GLUT1 (glucose transporter, normalizes local hyperglycemia) and hexokinase II (accelerates glucose metabolism). Module 2: a TLR4-responsive NF-kB promoter (activated by bacterial LPS in infected wounds) drives human beta-defensin-3 (HBD3, antimicrobial) and cathelicidin LL-37 via P2A. Module 3: NF-kB also activates a time-delayed module — NF-kB drives a slow-maturing Cre recombinase (with added N-terminal destabilization domain requiring trimethoprim to stabilize), which upon stabilization flips a loxP-flanked transcriptional stop, permanently activating a downstream PGK promoter driving VEGF-A and angiopoietin-1 (Ang-1) for coordinated angiogenesis. Module 4: constitutive EF1a drives nerve growth factor (NGF) for diabetic neuropathy repair and sensory nerve regeneration. Module 5: a hypoxia-responsive HRE promoter drives SDF-1/CXCL12 (stem cell recruitment to the hypoxic wound bed). Module 6: constitutive low-level expression of EGF (epidermal growth factor) from a weak promoter (pJ23117 equivalent) provides baseline epithelialization support. Module 7: a collagen-responsive element drives lysyl oxidase (LOX) for proper collagen crosslinking — ensuring strong scar tissue. Module 8: iCasp9 safety switch under constitutive promoter.",

    # === ANTIBODY / NANOBODY ENGINEERING CIRCUITS ===

    "A living antibody factory circuit for continuous bispecific antibody production against solid tumors. Module 1: engineered CHO cells with a constitutive CMV promoter driving anti-HER2 VHH (nanobody, heavy chain only) fused to anti-CD3 scFv via a flexible (G4S)3 linker — creating a bispecific T-cell engager (BiTE) that redirects endogenous T-cells to HER2+ tumor cells. Module 2: a second constitutive EF1a promoter drives anti-PD-L1 VHH nanobody fused to human IgG1 Fc domain — creating a checkpoint inhibitor antibody. Module 3: an ER stress-responsive UPR element (UPRE) drives GRP78 chaperone and PDI disulfide isomerase via P2A — enhancing protein folding capacity when antibody production overwhelms the ER. Module 4: a tetracycline-inducible TRE promoter drives anti-VEGF VHH-Fc (bevacizumab biosimilar) — inducible anti-angiogenic therapy. Module 5: constitutive PGK drives a furin-cleavable IL-2 mutein (IL-2 superkine with reduced CD25 binding, preferentially activating CD8+ T-cells over Tregs) — fused to albumin-binding domain for extended half-life. Module 6: a miR-21-responsive element (miR-21 is elevated in cancer) acts as a cancer proximity detector — when the cells sense tumor-derived exosomal miR-21, it drives additional anti-CD47 nanobody production (blocking the don't-eat-me signal). Module 7: HSV-TK safety gene under constitutive promoter. All antibody sequences include IL-2 signal peptide for secretion.",

    "A multi-format antibody platform circuit producing four different antibody formats simultaneously. Module 1: CMV promoter drives an anti-EGFR IgG heavy chain and light chain from a bicistronic IRES cassette — assembling into full IgG1. Module 2: EF1a drives an anti-HER2 scFv-Fc (single-chain antibody with Fc effector function) — a smaller format with better tumor penetration. Module 3: PGK drives an anti-PD-L1 nanobody (VHH) fused to albumin-binding peptide — extending half-life without Fc. Module 4: a tetracycline-inducible TRE drives an anti-CTLA4 x anti-4-1BB bispecific diabody (two scFvs in tandem) — inducible costimulatory checkpoint modulator. Module 5: constitutive expression of BIP chaperone and calnexin from SV40 promoter assists folding of all four antibody formats. Module 6: an IRE1-responsive element (activated by ER stress from high-level secretion) drives XBP1s transcription factor which activates genes for ER expansion — adaptive capacity increase. Module 7: a production rate controller — constitutive expression of a synthetic miRNA targeting all four antibody mRNAs at their 3'UTRs, balanced to maintain stable production levels without overloading the secretory pathway. Degradation-tagged GFP under CMV and mCherry under EF1a serve as production proxies.",

    # === CRISPR COMBINATION THERAPY ===

    "A multiplexed CRISPR circuit for simultaneous oncogene disruption and tumor suppressor activation in non-small cell lung cancer. Module 1: a tumor-specific hTERT promoter drives SaCas9 (Staphylococcus aureus Cas9, smaller than SpCas9) for gene disruption. Module 2: three U6-driven gRNAs target three oncogenes simultaneously — gRNA-1 targets KRAS G12D hotspot, gRNA-2 targets EGFR exon 19, gRNA-3 targets MET exon 14 splice site. Module 3: a separate CMV promoter drives dCas9-VP64-p65-Rta (VPR, a potent transcriptional activator) for CRISPRa gene activation. Module 4: three H1-driven gRNAs direct VPR to activate tumor suppressor promoters — gRNA-4 activates p53 (TP53), gRNA-5 activates RB1, gRNA-6 activates PTEN. Module 5: a p53-responsive element (activated by restored p53 from Module 4) drives TRAIL for apoptosis of neighboring untransduced cancer cells (bystander effect). Module 6: a constitutive PGK promoter drives a dominant-negative mutant of DNMT1 (dn-DNMT1) that globally reduces DNA methylation, reactivating silenced tumor suppressors epigenetically. Module 7: an ARF-responsive promoter drives BFP reporter confirming tumor suppressor pathway restoration. Module 8: a miR-34a expression cassette under CMV (p53-regulated miRNA that suppresses MYC, CDK6, SIRT1) for multi-target post-transcriptional tumor suppression. All cassettes flanked by cHS4 insulators to prevent silencing.",

    "A CRISPR base editing combination circuit for hereditary cancer syndromes. Module 1: an adenine base editor ABE8e (TadA8e-nCas9-D10A) under a liver-specific albumin (ALB) promoter for hepatocyte-specific editing. Module 2: a U6-driven gRNA targeting BRCA1 c.5382insC frameshift — ABE converts the premature stop codon back to wild-type, restoring BRCA1 function. Module 3: a second U6-driven gRNA targeting TP53 R175H hotspot mutation — ABE corrects the oncogenic mutation G>A back to wild-type. Module 4: a cytosine base editor CBE4 (APOBEC1-nCas9-UGI-UGI) under a separate TRE inducible promoter for orthogonal C-to-T editing. Module 5: an H1-driven gRNA directs CBE to create a premature stop codon in oncogenic PIK3CA H1047R (functionally disrupting the gain-of-function mutation). Module 6: a DNA damage-responsive p21 promoter drives anti-CRISPR AcrIIA4 protein — if editing causes excessive DNA damage, anti-CRISPR shuts down both editors (negative safety feedback). Module 7: a constitutive PGK drives UGI (uracil glycosylase inhibitor) at high levels to improve CBE editing efficiency. Module 8: a CMV-driven GFP-2A-puromycin resistance cassette for selection of edited cells. Module 9: a self-inactivating module — a gRNA targeting the ABE and CBE coding sequences themselves, which slowly accumulates editing in cis, causing the editors to self-destruct after sufficient therapeutic editing.",

    # === SYNTHETIC IMMUNE SYSTEM ===

    "A synthetic innate immune sensor circuit for broad-spectrum cancer immunosurveillance. Module 1: engineered NK cells with a constitutive EF1a promoter driving a chimeric NKG2D receptor with enhanced affinity for MICA/MICB stress ligands (overexpressed on many cancers). Module 2: an NFAT-responsive promoter (activated by NKG2D engagement) drives a cytokine cocktail — IL-15 superagonist (ALT-803 format: IL-15 mutant bound to sushi domain of IL-15Ra fused to IgG1 Fc), IFN-gamma, and GM-CSF via triple T2A peptide, potently activating the entire local immune response. Module 3: NFAT also drives BiTE secretion — anti-EPCAM x anti-CD3 for recruiting endogenous T-cells to epithelial tumors. Module 4: constitutive PGK drives a dominant-negative ADAM17 (prevents NKG2D ligand shedding by tumor cells — tumors normally cleave MICA/MICB to evade NK cells). Module 5: constitutive SV40 drives CXCR3 chemokine receptor for trafficking toward CXCL9/10/11-producing inflamed tumors. Module 6: a synthetic promoter responsive to tumor-derived TGF-beta (via SMAD-binding elements) paradoxically drives TRAIL and FasL — converting the immunosuppressive signal into a direct killing signal. Module 7: a hypoxia-responsive HRE promoter drives HIF2-alpha (which sustains NK cell cytotoxicity under hypoxic tumor conditions, unlike HIF1-alpha which impairs NK function). Module 8: rapamycin-inducible iCasp9 safety switch.",

    "An engineered dendritic cell vaccine circuit with multi-antigen presentation and adjuvant co-expression. Module 1: monocyte-derived dendritic cells transduced with a CMV promoter driving a polyepitope string — 20 tumor neoantigen peptides (each 25-mer with predicted MHC-I and MHC-II binding) separated by furin/cathepsin cleavage sites and flanked by PADRE universal T-helper epitope, all fused to the LAMP-1 lysosomal targeting signal for enhanced MHC-II presentation. Module 2: a second CMV promoter drives calreticulin (CRT) fused to the same neoantigen string but with a signal peptide for ER targeting — CRT acts as an eat-me signal when surface-exposed, enhancing DC phagocytosis. Module 3: constitutive EF1a drives a constitutively active STING variant (STING-V155M) that triggers type I interferon production without exogenous ligand — endogenous adjuvant. Module 4: constitutive PGK drives CD40L (CD40 ligand) that activates the DC's own CD40 in an autocrine/paracrine loop — mimicking T-helper cell licensing and driving IL-12 production. Module 5: a tetracycline-inducible TRE drives Flt3L (FMS-like tyrosine kinase 3 ligand) for expanding endogenous DCs in vivo. Module 6: constitutive expression of CCR7 (lymph node homing receptor) ensures DC migration to lymph nodes. Module 7: an NF-kB-responsive promoter drives IL-12p70 (bioactive heterodimer) only upon DC maturation — preventing premature cytokine release. Module 8: iCasp9 under SV40 promoter for safety.",

    # === TUMOR MICROENVIRONMENT REMODELING ===

    "A tumor microenvironment reprogramming circuit that converts immunosuppressive to immunostimulatory. Module 1: engineered macrophages with a constitutive MSCV promoter driving a chimeric TLR4-CD3zeta receptor that triggers M1 polarization upon LPS binding (abundant in tumor necrosis). Module 2: an NF-kB-responsive promoter drives iNOS (inducible nitric oxide synthase) for tumor cell killing via nitric oxide AND drives IRF5 transcription factor for M1 commitment via P2A. Module 3: a HIF1-responsive element drives arginase-1 shRNA (knocking down M2 marker arginase-1) — specifically in hypoxic tumor regions where TAMs accumulate. Module 4: constitutive PGK drives a soluble VEGF-Trap (aflibercept analog, sVEGFR1-Fc) that sequesters VEGF, reducing tumor angiogenesis AND reducing VEGF-mediated immunosuppression. Module 5: an NFAT-responsive element (from Module 1 CD3zeta signaling) drives secreted CXCL9 and CXCL10 chemokines via P2A — recruiting CD8+ T-cells and NK cells into the tumor. Module 6: constitutive SV40 drives anti-CSF1R nanobody that blocks the CSF1R receptor on neighboring M2 macrophages, forcing them out of M2 polarization. Module 7: a TGF-beta-responsive SMAD element paradoxically drives a dominant-negative TGF-beta receptor that acts as a decoy, sequestering tumor-derived TGF-beta. Module 8: AP1903-inducible iCasp9 for safety. GFP under NF-kB and mCherry under HIF1 for dual phenotype/hypoxia monitoring.",

    # === METABOLIC REPROGRAMMING FOR CANCER ===

    "A tumor metabolism disruption circuit targeting the Warburg effect. Module 1: engineered T-cells with a constitutive EF1a promoter driving a glucose transporter GLUT1 dominant-negative mutant (GLUT1-DN surface displayed) that competes with tumor cells for glucose uptake — starving the tumor while feeding the T-cell. Module 2: constitutive PGK drives PDK1 kinase inhibitor (dichloroacetate mimic peptide) fused to a cell-penetrating TAT peptide for secretion into neighboring tumor cells — forcing them from glycolysis back to oxidative phosphorylation (reversing Warburg effect, inducing ROS and apoptosis). Module 3: an NFAT-responsive promoter (upon antigen recognition) drives IDO1 inhibitor peptide (1-methyl-tryptophan synthesis via tryptophan hydroxylase enzyme) — blocking tumor immunosuppression via tryptophan depletion. Module 4: constitutive expression of MCT4 (monocarboxylate transporter 4) for lactate export — preventing intracellular acidification from tumor-derived lactate. Module 5: a lactate-responsive element drives adenosine deaminase (ADA) — converting immunosuppressive adenosine (abundant in acidic tumors) to inosine. Module 6: constitutive expression of CD73 blocking nanobody (anti-CD73 VHH) — preventing extracellular adenosine generation by tumor ectoenzymes. Module 7: a hypoxia-responsive HRE drives LDHA shRNA — knocking down lactate dehydrogenase in the engineered T-cell under hypoxic conditions to prevent metabolic exhaustion. Module 8: iCasp9 safety switch. Module 9: an anti-CD19 CAR under MSCV promoter as the primary tumor-targeting module.",

    # === ONCOLYTIC VIRUS CIRCUITS ===

    "A conditionally replicating oncolytic adenovirus with immune activation modules. Module 1: the E1A gene (essential for viral replication) is placed under a tumor-specific survivin promoter — the virus can only replicate in survivin-expressing cancer cells. Module 2: the E1B-55K gene (which normally inhibits p53) is deleted — in normal cells with functional p53, the virus cannot complete its replication cycle, but in p53-mutant cancer cells it can. Module 3: a CMV promoter inserted in a deleted E3 region drives GM-CSF (granulocyte-macrophage colony-stimulating factor) for dendritic cell recruitment and maturation. Module 4: a second CMV cassette in E3 drives a membrane-bound anti-CD3 scFv that decorates the surface of infected cancer cells, directly engaging endogenous T-cells for killing. Module 5: a hypoxia-responsive HRE promoter drives RANTES/CCL5 for immune cell chemotaxis into hypoxic tumor cores. Module 6: the fiber knob protein is modified with an RGD peptide insertion for enhanced integrin-mediated tumor cell entry (broadens tropism beyond CAR receptor). Module 7: a miR-Let7-responsive element in the E1A 3'UTR — Let7 is abundant in normal cells (suppresses E1A translation) but downregulated in cancer cells (allows E1A expression), providing an additional layer of tumor selectivity. Module 8: an shRNA cassette driven by a VA-RNA promoter targets PD-L1 mRNA in infected tumor cells, preventing immune checkpoint activation.",

    # === GENE THERAPY FOR GENETIC CANCERS ===

    "A combination gene therapy for Li-Fraumeni syndrome with cancer prevention and surveillance. Module 1: AAV9 vector with a constitutive CAG promoter driving wild-type TP53 cDNA — restoring p53 function in haploinsufficient cells. Module 2: a p53-responsive element (p21/CDKN1A promoter) drives secreted Gaussia luciferase (GLuc) — a blood-detectable biomarker that reports p53 pathway activity (if p53 restoration is successful, GLuc goes up). Module 3: a constitutive PGK promoter drives MDM2 at carefully calibrated low levels (weak RBS) — sufficient to prevent p53 overactivity in normal cells while insufficient to suppress p53 in transforming cells where p53 demand is high. This creates a homeostatic buffer. Module 4: an E2F-responsive element (activated in aberrantly proliferating cells) drives TRAIL — if any cell escapes p53 control and proliferates, E2F-driven TRAIL induces apoptosis. Module 5: the same E2F element drives secreted alkaline phosphatase (SEAP) — a second blood-detectable biomarker that rises when cells are proliferating abnormally, serving as a cancer early warning system. Module 6: a constitutive CMV promoter drives miR-34a cluster (direct p53 transcriptional target normally) — reinforcing cell cycle arrest and apoptosis through post-transcriptional regulation of CDK6, MYC, BCL-2, and SIRT1. Module 7: a MYC-responsive promoter (E-box elements) drives ARF (p14ARF) which sequesters MDM2, releasing p53 — creating a synthetic tumor suppressor checkpoint. Module 8: all cassettes flanked by cHS4 insulator elements.",

    # === ADVANCED BIOSENSORS FOR CANCER ===

    "A multi-analyte liquid biopsy circuit for early cancer detection with signal amplification. Module 1: engineered HEK293 cells with an NF-kB-responsive promoter (activated by tumor-derived exosomal miR-155, which is processed to activate TLR7/8 → NF-kB) driving T7 RNAP for 1000-fold signal amplification. Module 2: T7 promoter drives secreted NanoLuc-HiBiT fragment that complements with LgBiT in a companion blood test reagent — luminescent signal detectable at femtomolar concentrations. Module 3: a STAT3-responsive element (activated by tumor-derived IL-6 in blood) drives a second reporter — secreted placental alkaline phosphatase (SEAP), measurable by standard ELISA. Module 4: a hypoxia-responsive HRE element (activated by tumor-associated exosomes carrying HIF1-alpha) drives a third reporter — secreted beta-hCG fragment detectable by pregnancy test strips (accessible point-of-care). Module 5: a synthetic promoter responsive to cell-free tumor DNA (cfDNA) — a dCas9-VPR with a panel of 10 U6-driven gRNAs targeting common tumor suppressor promoter methylation patterns. When unmethylated cfDNA (from lysed tumor cells) is taken up, the gRNAs activate a GFP-luciferase dual reporter. Module 6: a normalizer module — constitutive CMV drives secreted SEAP from a separate, NF-kB-independent cassette as an internal control. Module 7: a time-stamp module — each detection event also activates a Cre recombinase that flips a permanent memory switch (loxP-flanked cassette), recording cumulative cancer exposure. Module 8: iCasp9 safety switch.",

    # === COMBINATION PLATFORMS ===

    "A next-generation tumor-infiltrating lymphocyte (TIL) enhancement circuit with seven functional modules. Module 1: ex vivo expanded TILs transduced with a lentiviral construct. Constitutive MSCV promoter drives a costimulatory switch receptor — PD-1 ectodomain fused to CD28 intracellular domain, converting inhibitory PD-L1 engagement into costimulatory signaling. Module 2: constitutive EF1a drives a dominant-negative TGF-beta receptor II (dnTGFBRII) — blocking TGF-beta-mediated exhaustion. Module 3: NFAT-responsive promoter drives IL-15 superagonist (N72D IL-15 mutant-sushi domain-Fc fusion) for autocrine T-cell survival without exogenous cytokines. Module 4: NFAT also drives a secreted anti-CTLA4 scFv-Fc — localized checkpoint blockade only at sites of active tumor killing. Module 5: constitutive PGK drives CXCR3 overexpression for enhanced chemotaxis toward CXCL9/10/11-rich inflamed tumor regions. Module 6: a HIF1-responsive element (activated in hypoxic tumor cores) drives GLUT3 (high-affinity glucose transporter) and LDHA (lactate dehydrogenase A) via P2A — metabolic adaptation to compete with tumor cells for glucose. Module 7: constitutive expression of anti-TIGIT nanobody (VHH) fused to GPI anchor for surface display — blocking TIGIT inhibitory receptor. Module 8: an AP20187 (CID)-inducible rapamycin-resistant mTOR variant — enabling selective expansion of engineered TILs with CID while rapamycin controls host immune cells. Module 9: iCasp9 under SV40 for safety shutdown.",

    "A synthetic gene circuit for converting fibroblasts into tumor-killing immune cells (direct reprogramming). Module 1: a doxycycline-inducible TRE promoter drives PU.1 and IRF8 (master myeloid transcription factors) via P2A — initiating fibroblast-to-macrophage conversion. Module 2: constitutive EF1a drives BATF3 and IRF4 (dendritic cell specification factors) at lower levels via weak IRES — biasing the conversion toward a DC-like antigen-presenting phenotype. Module 3: once reprogramming is complete (monitored by a CD11c promoter that activates in myeloid cells), CD11c drives an anti-HER2 CAR-phagocytosis receptor (scFv-CD8hinge-FcRgamma-CD3zeta) for targeted tumor engulfment. Module 4: CD11c also drives constitutively active STING (cGAS-STING pathway constitutively on) via P2A — triggering type I interferon production for immune activation. Module 5: an NF-kB-responsive element drives IL-12p70 (bioactive heterodimer assembled via IRES) — potent Th1 immune activation. Module 6: a maturation sensor — MHC-II promoter (CIITA-responsive, activated upon DC maturation) drives CCR7 for lymph node migration AND cross-presentation machinery (TAP1, TAP2, tapasin) via polycistronic cassette. Module 7: a safety circuit — the original fibroblast-specific FSP1 promoter drives HSV-TK, so any cells that revert to fibroblast identity are killed by ganciclovir. Module 8: constitutive PGK drives puromycin resistance for selection of successfully transduced cells.",

    # === NEXT-GEN KILL SWITCHES ===

    "A multi-layered fail-safe circuit for any engineered therapeutic cell with five independent kill mechanisms. Layer 1: rapamycin-inducible — constitutive EF1a drives FKBP-iCasp9, dimerized and activated by rapamycin (or AP1903). Layer 2: ganciclovir-activated — constitutive PGK drives HSV-TK, phosphorylates ganciclovir into toxic nucleotide analog causing DNA chain termination. Layer 3: antibody-mediated — constitutive SV40 drives truncated EGFR (EGFRt) surface marker that enables cetuximab-mediated ADCC/CDC killing by host immune system. Layer 4: small molecule-controlled essential gene — the endogenous DHFR gene is knocked out and replaced with an IPTG-inducible version (TRE-DHFR with rtTA from constitutive CMV) — without doxycycline, cells cannot synthesize thymidine and die. Layer 5: an autonomous dead-man switch — constitutive expression of a synthetic transcription factor (synTF) that REPRESSES a toxin promoter (pSynTF drives LacI which represses pLac-CcdB-MazF dual toxin). The synTF gene has a short half-life mRNA (AU-rich 3'UTR elements) requiring continuous transcription. If the cell acquires mutations silencing synTF expression, LacI drops, toxins are derepressed, and the cell self-destructs. A GFP reporter under the synTF-responsive promoter monitors dead-man switch integrity. Each kill layer is insulated with cHS4 insulators and uses different poly-A signals.",

    # === MICROBIOME ENGINEERING FOR CANCER ===

    "An engineered gut bacterium for colorectal cancer immunotherapy with tumor sensing and four therapeutic payloads. Module 1: E. coli Nissle 1917 chassis with a synthetic colorectal cancer sensor — a chimeric two-component system where the sensor kinase recognizes tumor-shed carcinoembryonic antigen (CEA) via an anti-CEA nanobody fused to the EnvZ extracellular domain, and the response regulator OmpR activates the ompC promoter. Module 2: pOmpC drives anti-PD-L1 nanobody (VHH) with PelB secretion signal — localized checkpoint inhibition at the tumor surface. Module 3: pOmpC also drives a ClyA-heparin cofactor II (HCII) fusion pore that displays HCII on the bacterial surface — HCII cleaves and activates latent TGF-beta trap (LAP domain) to neutralize immunosuppressive TGF-beta. Module 4: constitutive pJ23119 drives butyrate biosynthesis operon (tesB, ato genes) — producing the SCFA butyrate which inhibits HDAC (histone deacetylase) in cancer cells, reactivating silenced tumor suppressors. Module 5: a bile acid sensor (based on the VtrA/VtrC system) detects secondary bile acids (elevated in CRC) and drives additional anti-CTLA4 nanobody production. Module 6: quorum sensing (LuxI/LuxR) synchronized lysis circuit — at high density, phiX174 lysis gene releases all accumulated intracellular nanobodies in a bolus. Module 7: double auxotrophy — DAP (dapA deletion) and thymidine (thyA deletion) ensure cells die outside the supplemented gut environment. Module 8: a constitutive mCherry reporter tracks colonization, GFP under pOmpC tracks tumor proximity.",

    # === EXOSOME-BASED DELIVERY ===

    "An engineered exosome-producing cell circuit for targeted cancer therapy delivery. Module 1: HEK293T cells with a constitutive CMV promoter driving Lamp2b-RVG peptide fusion (lysosomal membrane protein fused to rabies virus glycoprotein peptide for brain-targeting exosomes — for glioblastoma). Module 2: constitutive EF1a drives a TRAIL-CD63 fusion — TRAIL is displayed on exosome surface via the tetraspanin CD63, enabling apoptosis of cancer cells upon exosome contact. Module 3: a loading module — constitutive PGK drives a Cas9 protein fused to a CD63-binding nanobody (Nb-CD63) that is sorted into exosomes. U6 drives a gRNA targeting MGMT (DNA repair gene that causes temozolomide resistance in glioblastoma), which is co-loaded into exosomes via a packaging signal. Module 4: constitutive SV40 drives a miR-124 expression cassette (anti-oncogenic miRNA targeting STAT3, ROCK2, and SOS1) with an exosome-targeting zipcode motif in the 3'UTR for selective exosome loading. Module 5: a synthetic NF-kB-responsive promoter drives additional exosome production during inflammation — the UPR element also drives Rab27a and nSMase2 (exosome biogenesis factors) via P2A, increasing exosome yield. Module 6: a tetracycline-inducible TRE drives anti-VEGF nanobody-Lamp2b fusion — inducible anti-angiogenic exosomes. Module 7: constitutive expression of VSV-G (vesicular stomatitis virus glycoprotein) for enhanced exosome-cell fusion. Module 8: iCasp9 safety switch for the producer cells.",

    # === SYNTHETIC BIOLOGY FOR BLOOD CANCERS ===

    "A multi-target CAR-NK circuit for relapsed/refractory acute myeloid leukemia. Module 1: an NK-92 cell line (or primary NK cells) with constitutive MSCV promoter driving a tri-specific CAR — anti-CD33 scFv, anti-CD123 nanobody (VHH), and anti-CLL-1 scFv connected by flexible linkers, with NKG2D transmembrane domain, 2B4 costimulatory domain, and CD3zeta signaling domain. This targets three AML antigens simultaneously to prevent escape. Module 2: constitutive EF1a drives a soluble NKG2D-Fc decoy that blocks sMICA/sMICB (shed by AML cells to evade NK killing) from reaching host NK cells. Module 3: NFAT-responsive promoter drives a bicistronic cassette of IL-15 (NK cell survival) and IL-21 (NK cell activation and proliferation) via T2A. Module 4: constitutive PGK drives CXCR4 (receptor for CXCL12/SDF-1, expressed by bone marrow stroma) for bone marrow homing where AML resides. Module 5: a hypoxia-responsive HRE promoter drives HIF2-alpha (which sustains NK cytotoxicity under hypoxia, unlike HIF1-alpha) — critical since the bone marrow niche is hypoxic. Module 6: constitutive SV40 drives an anti-CD47 scFv fused to GPI anchor — surface-displayed CD47 blocker enhances phagocytosis of opsonized AML cells. Module 7: constitutive expression of anti-TIM3 nanobody — blocking TIM3 checkpoint that is particularly important in AML immune evasion. Module 8: a rapamycin-inducible iCasp9 with an mCherry-P2A fusion (mCherry reports kill switch integrity).",

    # === PRECISION MEDICINE CIRCUITS ===

    "A patient-specific neoantigen-responsive circuit for personalized cancer immunotherapy. Module 1: primary T-cells transduced with a synthetic Notch (synNotch) receptor whose extracellular domain is a patient-specific anti-neoantigen scFv (computationally designed from tumor whole-exome sequencing) fused to a Notch core and Gal4-VP64 intracellular domain, driven by EF1a promoter. Module 2: 5xUAS promoter (activated by released Gal4-VP64) drives a second synNotch receptor — this one targeting HLA-A2/neoantigen-peptide complex (TCR-mimic scFv), releasing LexA-p65. Module 3: lexAop promoter (requiring Module 2 activation) drives the final therapeutic payload — a cytotoxic cassette of perforin and granzyme B via T2A, plus secreted IFN-gamma. This triple-layered synNotch cascade ensures ultra-specific killing ONLY of cells displaying the exact neoantigen. Module 4: UAS promoter also drives a secreted IL-2 mutein (superkine, reduced CD25 binding) — autocrine T-cell survival that only activates near the neoantigen+ tumor. Module 5: constitutive PGK drives a dominant-negative SHP-1 (blocking the most upstream TCR inhibitory phosphatase) for enhanced signaling sensitivity. Module 6: constitutive expression of anti-LAG3 nanobody for checkpoint resistance. Module 7: a constitutive CMV drives a barcode RNA (unique per patient construct) that enables PCR tracking of engineered cells in blood samples. Module 8: AP1903-inducible iCasp9 safety. Module 9: a NFAT-responsive GFP for activation monitoring.",

    # === ANTI-METASTASIS CIRCUITS ===

    "A circulating tumor cell (CTC) hunter-killer circuit that patrols the bloodstream. Module 1: engineered NK cells with constitutive EF1a driving a bispecific CAR — anti-EpCAM scFv and anti-N-cadherin nanobody (marking epithelial-mesenchymal transition CTCs) joined by a flexible linker, with 2B4-CD3zeta signaling domain. This targets both epithelial and mesenchymal CTCs. Module 2: constitutive PGK drives E-selectin ligand PSGL-1 (CD162) and L-selectin (CD62L) via P2A — enabling the NK cells to patrol the vasculature by rolling on endothelium like natural immune cells. Module 3: NFAT-responsive promoter drives TRAIL (membrane-bound) — upon CTC engagement, the NK cell deploys TRAIL which kills DR4/DR5+ CTCs. Module 4: NFAT also drives secreted granzyme B-anti-EpCAM immunotoxin fusion — released locally to kill bystander CTCs. Module 5: constitutive SV40 drives an anti-MMP9 scFv (blocking matrix metalloproteinase-9 that CTCs use for extravasation into metastatic sites). Module 6: a synthetic promoter responsive to platelet-derived TGF-beta (platelets coat CTCs for immune evasion) drives a platelet-disaggregation factor (apyrase, which degrades ADP) — stripping the platelet cloak from CTCs. Module 7: constitutive expression of DNAM-1 (CD226) for enhanced recognition of PVR/Nectin-2 on tumor cells. Module 8: rapamycin-inducible iCasp9. Module 9: constitutive BFP reporter for in vivo tracking.",

    # === EPIGENETIC REPROGRAMMING ===

    "An epigenetic reprogramming circuit that reverses cancer cell identity to normal. Module 1: a lentiviral construct with a tumor-specific survivin promoter driving dCas9-TET1 catalytic domain (DNA demethylase) — active only in cancer cells. Module 2: six U6-driven gRNAs direct TET1 to demethylate the promoters of six key tumor suppressors: p16/CDKN2A, MLH1 (mismatch repair), BRCA1, RB1, VHL, and APC — reactivating these silenced genes. Module 3: a separate EF1a promoter drives dCas9-p300 (histone acetyltransferase) for activating chromatin. Module 4: four H1-driven gRNAs direct p300 to enhancers of differentiation genes: GATA3 (luminal breast differentiation), CDX2 (intestinal differentiation), MITF (melanocyte differentiation), and MYOD1 (muscle differentiation) — pushing cancer cells toward terminally differentiated, non-dividing states. Module 5: a constitutive PGK drives EZH2 dominant-negative (blocks Polycomb repressive complex) — globally reducing H3K27me3 repressive marks on tumor suppressor loci. Module 6: a p53-responsive promoter (activated by demethylated/reactivated p53 pathway from Module 2) drives TRAIL — if cancer cells don't differentiate, restored p53 triggers apoptosis via TRAIL. Module 7: a CMV promoter drives miR-200 cluster (reverses EMT: represses ZEB1, ZEB2, SNAIL) and miR-34a (suppresses stem cell genes: CD44, BMI1, NOTCH1) via a polycistronic cassette. Module 8: a MYC-responsive element (E-box) drives an anti-MYC shRNA — self-dampening oncogene feedback. Module 9: HSV-TK under constitutive promoter for ganciclovir safety.",

    # === ADVANCED DRUG PRODUCTION IN VIVO ===

    "An implantable cell-based factory for continuous anti-cancer antibody production with regulated output. Module 1: encapsulated CHO cells in an immunoprotective alginate-PLL-alginate microcapsule. CMV promoter drives pembrolizumab (anti-PD-1) heavy and light chains from a bicistronic IRES cassette — continuous anti-PD-1 production without repeated infusions. Module 2: a tetracycline-responsive TRE promoter drives atezolizumab (anti-PD-L1) heavy and light chains — orthogonal checkpoint that can be turned on/off by oral doxycycline. Module 3: constitutive EF1a drives bevacizumab (anti-VEGF) single-chain format (scFv-Fc) at calibrated low levels (weak RBS) for continuous anti-angiogenic baseline therapy. Module 4: an inflammation-responsive NF-kB element drives ipilimumab mimic (anti-CTLA4 scFv-Fc) — checkpoint inhibition that auto-scales with tumor-associated inflammation. Module 5: constitutive PGK drives BiP chaperone and PDI for ER folding support. Module 6: an IRE1-responsive UPR element drives XBP1s which activates ER expansion genes — adaptive secretory capacity that scales with production demand. Module 7: a growth rate controller — constitutive expression of p21 (cell cycle inhibitor) at calibrated levels maintains the cells in a slow-dividing, high-producing state rather than fast-growing low-producing. Module 8: HSV-TK under SV40 for ganciclovir-mediated destruction of the implant if needed.",

    # === STEM CELL-BASED CANCER THERAPY ===

    "A neural stem cell-based circuit for glioblastoma with tumor-tropic homing and multi-modal therapy. Module 1: human neural stem cells (NSCs, which naturally home to brain tumors) transduced with a constitutive EF1a promoter driving cytosine deaminase-UPRT fusion (CD-UPRT) — converts 5-FC prodrug to 5-FU directly within the tumor, AND converts 5-FU to 5-FUMP for enhanced incorporation. Module 2: a HIF1-responsive HRE promoter drives TRAIL (preferentially active in hypoxic GBM core) — triggering DR4/DR5-mediated apoptosis of GBM stem cells which reside in hypoxic niches. Module 3: constitutive PGK drives an anti-EGFRvIII scFv fused to a carboxylesterase (CE) enzyme — the scFv targets EGFRvIII+ GBM cells and the CE locally converts irinotecan (CPT-11) prodrug to its active SN-38 metabolite, achieving three orthogonal prodrug strategies. Module 4: a GBM-conditioned medium responsive promoter (based on CXCL12/SDF-1 receptor CXCR4) drives secretion of IL-12 for immune activation. Module 5: constitutive CMV drives an shRNA targeting MGMT (O6-methylguanine methyltransferase) in neighboring tumor cells via exosome-mediated transfer — resensitizing GBM to temozolomide chemotherapy. Module 6: a VEGF-responsive promoter (activated by tumor-derived VEGF) drives endostatin (anti-angiogenic) — autoregulated anti-angiogenic therapy. Module 7: constitutive expression of a dominant-negative Notch1 (DN-MAML) — blocking Notch signaling that maintains GBM stem cell self-renewal. Module 8: a tetracycline-inducible HSV-TK for safety.",

    # === RADIATION SENSITIZER CIRCUIT ===

    "A radiation-responsive circuit that amplifies radiotherapy effectiveness with immune activation. Module 1: a tumor-targeting AAV with a synthetic radiation-responsive promoter (CArG/Egr-1 elements, activated by ionizing radiation-induced ROS) driving TNF-alpha — the original TNFerade concept enhanced with modern circuit design. Module 2: the same radiation-responsive promoter drives a STING agonist (constitutively active cGAS variant) — triggering type I interferon and converting immunogenic cell death from radiation into a systemic immune response. Module 3: constitutive weak PGK promoter drives a dominant-negative DNA-PKcs (DNA repair kinase) at low basal levels — sensitizing tumor cells to radiation by partially impairing DNA double-strand break repair. Module 4: radiation-responsive element drives Smac/DIABLO (mitochondrial apoptosis activator) — overcoming IAP-mediated apoptosis resistance common in radio-resistant tumors. Module 5: a constitutive CMV drives shRNA targeting RAD51 (homologous recombination repair gene) — further impairing tumor DNA repair specifically. Module 6: the radiation-responsive promoter also drives a secreted bispecific anti-PD-L1 x anti-4-1BB antibody — combining checkpoint blockade with costimulation at the irradiated tumor site. Module 7: a p53-responsive element drives PUMA (pro-apoptotic BH3-only protein) — amplifying p53-dependent apoptosis after radiation damage. Module 8: a constitutive SV40 drives GFP-luciferase fusion for bioluminescence monitoring of vector persistence. Each module has cHS4 insulator elements.",

    # === TUMOR VACCINE CIRCUIT ===

    "An in-situ tumor vaccination circuit that converts dying tumor cells into vaccines. Module 1: a tumor-targeting oncolytic herpes simplex virus (HSV) with a tumor-specific NESTIN promoter driving ICP34.5 (neurovirulence gene, required for replication) — restricting viral replication to nestin+ tumor cells. Module 2: a CMV promoter in the virus drives calreticulin (CRT) fused to the tumor cell surface via a GPI anchor — the eat-me signal enhances phagocytosis of dying infected cells by dendritic cells. Module 3: a separate CMV cassette drives HMGB1 (alarmin, danger signal that activates TLR4 on DCs) fused to a secretion signal. Module 4: a third therapeutic cassette drives ATP biosynthesis enzyme (adenylate kinase) to boost extracellular ATP release during cell death — ATP is a find-me signal for DCs via P2Y2 receptor. Together, CRT + HMGB1 + ATP create all three immunogenic cell death (ICD) signals. Module 5: a viral immediate-early promoter drives a polyepitope string of 15 predicted neoantigens with cathepsin cleavage sites for cross-presentation on MHC-I. Module 6: constitutive viral expression of GM-CSF for DC recruitment. Module 7: an NF-kB-responsive element (activated during viral infection) drives OX40L and 4-1BBL costimulatory ligands via P2A for T-cell activation. Module 8: an anti-PD-L1 shRNA cassette under VA promoter to prevent upregulation of the checkpoint in infected cells.",
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

CRITICAL RULES:
1. Every CDS MUST have a transcription interaction (from a promoter) AND a translation interaction (from an RBS)
2. For EVERY regulatory relationship described (activates, represses, drives, inhibits, blocks, converts signal), include the corresponding activation or repression interaction
3. Include ALL components explicitly mentioned — for circuits with 7+ modules, you should have 30-60+ components
4. Feedback loops MUST be represented as complete interaction chains (A represses B, B activates C, C represses A)
5. Use descriptive snake_case IDs (e.g., anti_pd1_scfv_cds, nfat_promoter, icasp9_cds)
6. Include operators explicitly when binding sites are mentioned
7. Include 'other' type for signal peptides, P2A/T2A peptides, degradation tags, ITRs, insulators, linkers, GPI anchors
8. For multi-module circuits, EVERY module must have complete promoter→RBS→CDS→terminator structure
9. MAXIMIZE activation and repression interactions — every described regulatory relationship MUST appear as an interaction

Respond with valid JSON only, no explanation."""


def generate_with_gpt(descriptions: list[str], api_key: str) -> list[dict]:
    """Generate circuit JSONs using GPT-5.4 API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    results = []
    failed = []

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
                max_completion_tokens=8192,  # doubled for these massive circuits
                temperature=0.7,
            )
            output = response.choices[0].message.content
            elapsed = time.time() - t0

            cleaned = output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)

            # Validate
            valid_types = {"activation", "repression", "transcription", "translation"}
            comp_ids = {c['id'] for c in parsed.get('components', [])}

            # Remove bad interactions
            parsed['interactions'] = [
                ix for ix in parsed.get('interactions', [])
                if ix.get('type') in valid_types
                and ix.get('from') in comp_ids
                and ix.get('to') in comp_ids
            ]

            n_comp = len(parsed['components'])
            n_inter = len(parsed['interactions'])
            inter_types = sorted(set(ix['type'] for ix in parsed['interactions']))

            if n_comp < 10:
                print(f"    SKIP: Only {n_comp} components (need 10+)")
                failed.append(i)
                continue

            if n_inter < 8:
                print(f"    SKIP: Only {n_inter} interactions (need 8+)")
                failed.append(i)
                continue

            results.append({"description": desc, "circuit": parsed})
            print(f"    OK ({elapsed:.1f}s) — {n_comp} comps, {n_inter} inters ({', '.join(inter_types)})")
        except json.JSONDecodeError as e:
            print(f"    FAILED (JSON): {e}")
            failed.append(i)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(i)

        time.sleep(0.3)

    # Retry failed ones once with lower temp and explicit instruction
    if failed:
        print(f"\n  Retrying {len(failed)} failed examples...")
        for idx in failed:
            desc = descriptions[idx]
            print(f"  [retry {idx+1}] {desc[:60]}...")
            try:
                response = client.chat.completions.create(
                    model="gpt-5.4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": desc},
                    ],
                    max_completion_tokens=8192,
                    temperature=0.5,
                )
                output = response.choices[0].message.content
                cleaned = output.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("```json").strip("```").strip()
                parsed = json.loads(cleaned)
                valid_types = {"activation", "repression", "transcription", "translation"}
                comp_ids = {c['id'] for c in parsed.get('components', [])}
                parsed['interactions'] = [
                    ix for ix in parsed.get('interactions', [])
                    if ix.get('type') in valid_types
                    and ix.get('from') in comp_ids and ix.get('to') in comp_ids
                ]
                if len(parsed['components']) >= 10 and len(parsed['interactions']) >= 8:
                    results.append({"description": desc, "circuit": parsed})
                    print(f"    OK (retry)")
                else:
                    print(f"    SKIP (retry): {len(parsed['components'])} comps, {len(parsed['interactions'])} inters")
            except Exception as e:
                print(f"    FAILED again: {e}")
            time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-key", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / 'scraped'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    descs = CANCER_LEVEL_CIRCUITS
    print(f"=== Generating {len(descs)} CANCER-LEVEL circuit examples via GPT-5.4 ===\n")

    results = generate_with_gpt(descs, args.openai_key)

    # Save raw
    raw_path = output_dir / 'cancer_level_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as training JSONL
    jsonl_path = output_dir / 'cancer_level_training.jsonl'
    with open(jsonl_path, 'w') as f:
        for item in results:
            training_example = {
                "messages": [
                    {"role": "system", "content": "You are a genetic circuit design assistant. Convert natural language circuit descriptions into structured JSON with components and interactions."},
                    {"role": "user", "content": item["description"]},
                    {"role": "assistant", "content": json.dumps(item["circuit"])},
                ]
            }
            f.write(json.dumps(training_example) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"Generated: {len(results)}/{len(descs)}")
    print(f"Saved to: {jsonl_path}")

    # Stats
    all_comps = [len(r['circuit']['components']) for r in results]
    all_inters = [len(r['circuit']['interactions']) for r in results]
    inter_type_counts = {}
    for r in results:
        for ix in r['circuit']['interactions']:
            t = ix['type']
            inter_type_counts[t] = inter_type_counts.get(t, 0) + 1

    print(f"Avg components: {sum(all_comps)/len(all_comps):.1f}")
    print(f"Min components: {min(all_comps)}")
    print(f"Max components: {max(all_comps)}")
    print(f"Avg interactions: {sum(all_inters)/len(all_inters):.1f}")
    print(f"Min interactions: {min(all_inters)}")
    print(f"Max interactions: {max(all_inters)}")
    print(f"Interaction types: {inter_type_counts}")


if __name__ == "__main__":
    main()

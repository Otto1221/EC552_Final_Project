"""
Generate complex circuit examples based on real published systems and
advanced therapeutic designs via GPT-5.4.

Target: ~300 examples covering diverse complex circuit topologies.
"""

import json
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Descriptions organized by category
# ---------------------------------------------------------------------------

PUBLISHED_CIRCUITS = [
    # === Gardner et al. Toggle Switch variants ===
    "A genetic toggle switch (Gardner et al. 2000): LacI represses pTet driving TetR, while TetR represses pLac driving LacI. The bistable switch is flipped by IPTG (inactivates LacI, switching to TetR-high state) or aTc (inactivates TetR, switching to LacI-high state). GFP is co-expressed with LacI from pLac as a state reporter. Each gene has its own RBS and shares a B0015 terminator.",

    "An improved toggle switch with ultrasensitivity: LacI with cooperative binding to two tandem lacO operators represses TetR expression. TetR with cooperative binding to two tandem tetO operators represses LacI expression. Both arms include positive autoregulation (LacI weakly activates its own promoter, TetR weakly activates its own promoter) to sharpen the switching threshold. GFP and mCherry reporters indicate each state.",

    "A three-state toggle switch using three mutually repressing transcription factors: LacI represses CI, CI represses TetR, and TetR represses LacI, but unlike the repressilator, each repressor also activates the next-next repressor (LacI activates TetR indirectly via double repression), creating three stable states rather than oscillation. Each state has a unique reporter (GFP, mCherry, BFP).",

    # === Elowitz Repressilator variants ===
    "The repressilator (Elowitz & Leibler 2000): three repressors in a cyclic negative feedback loop. TetR represses pLac driving LacI, LacI represses pCI driving CI, CI represses pTet driving TetR. Each node has its own RBS and terminator. A separate reporter plasmid has GFP driven by pTet (reports TetR-low phase). All three repressors carry ssrA degradation tags for faster oscillation dynamics.",

    "A dual-feedback repressilator with quorum coupling (Danino et al. 2010): the standard repressilator (TetR→LacI→CI→TetR) is augmented with a quorum sensing module. LuxI is co-expressed with one repressilator node, producing AHL. AHL activates LuxR which drives expression of the same node's repressor on neighboring cells, synchronizing oscillations across the population. GFP reporter on pTet.",

    "A tunable repressilator with degradation control: the standard three-node repressilator (LacI→TetR→CI→LacI) where each repressor is fused to a tunable degradation tag. An orthogonal protease (TEV) expressed from an IPTG-inducible promoter cleaves the degradation tags at a controllable rate, allowing external tuning of oscillation period. Each node has a different fluorescent reporter.",

    # === Synthetic biology logic gates ===
    "A two-input AND gate: input A (IPTG) derepresses pLac driving the T7 RNA polymerase N-terminal half fused to a leucine zipper. Input B (arabinose) activates pBAD driving the T7 RNAP C-terminal half fused to a complementary leucine zipper. Only when both halves are present do they dimerize into functional T7 RNAP, which drives GFP from a T7 promoter. Each component has its own RBS and terminator.",

    "A two-input NOR gate: both IPTG and arabinose independently drive repressors (LacI from pBAD and AraC from pLac in a cross-wired configuration such that either input represses the output). The output GFP is driven from a promoter that is repressed by both CI (expressed from pLac, induced by IPTG) and TetR (expressed from pBAD, induced by arabinose). GFP is only ON when neither input is present.",

    "A two-input NAND gate: LacI (constitutive) represses pLac-CI and TetR (constitutive) represses pTet-CI. The output promoter pCI drives GFP and is repressed by CI. CI is only produced when BOTH repressors (LacI, TetR) are inactive (IPTG + aTc). So GFP is ON unless both inputs are present. Each module has explicit RBS, operators, and terminators.",

    "A three-input majority gate: three inputs (IPTG, arabinose, aTc) each drive a different activator. Any two activators together can activate the output promoter through an OR of three AND gates. Implemented using three split T7 RNAP pairs — each pair is activated by a different combination of two inputs. All three pairs target the same T7 promoter driving GFP.",

    "An XOR gate using a dual-repression cascade: Input A (IPTG) drives expression of activator LuxR AND repressor TetR. Input B (arabinose) drives expression of activator AraC AND repressor LacI. The output GFP is driven by a hybrid promoter requiring activation (by LuxR OR AraC) but inhibited by repression (by TetR AND LacI together via tandem operators). When only one input is present, activation dominates. When both are present, the cross-repressors block output.",

    # === Danino Lab — Synchronized Lysis Circuits ===
    "A synchronized lysis circuit (Danino et al. 2016): LuxI is constitutively expressed, producing AHL. When population density reaches a threshold, AHL activates LuxR which drives pLux expressing a phage lysis gene (E-gene from phiX174). Lysis kills most cells, releasing therapeutic payload (pre-accumulated intracellular protein). A few surviving cells regrow and repeat the cycle. GFP reporter on pLux monitors the oscillation. The therapeutic payload ClyA (cytotoxin) is expressed from a constitutive promoter.",

    "A dual-strain synchronized lysis circuit for combination therapy: Strain A uses LuxI/LuxR quorum sensing to trigger lysis releasing anti-PD-L1 nanobody. Strain B uses RhlI/RhlR (orthogonal QS) to trigger lysis releasing anti-CTLA4 nanobody. Both strains are mixed and injected into tumor. Each strain has its own QS system, lysis cassette, and therapeutic payload. Antibiotic resistance markers (AmpR for A, KanR for B) enable selective culture.",

    # === Wendell Lim Lab — synNotch circuits ===
    "A synNotch-CAR two-step activation circuit (Roybal et al. 2016): a synNotch receptor with anti-GFP scFv extracellular domain detects GFP on target cells. Cleavage releases Gal4-VP64 transcription factor which activates a UAS promoter driving an anti-CD19 CAR (scFv-CD28-CD3zeta). Only cells contacting GFP+ targets express the CAR. A constitutive EF1a promoter drives the synNotch receptor. The CAR module includes a 4-1BB costimulatory domain.",

    "A multi-antigen synNotch logic circuit: synNotch-A (anti-HER2) releases Gal4-VP64 activating UAS-driven anti-mesothelin CAR. synNotch-B (anti-EpCAM) releases LexA-VP64 activating lexAop-driven IL-21 cytokine. The CAR kills tumor cells while IL-21 boosts immune response. Both synNotch receptors are driven by constitutive promoters (EF1a and PGK). A safety iCasp9 kill switch under TRE promoter is activated by doxycycline.",

    "A three-input synNotch computation circuit: three synNotch receptors detect three different surface antigens (CD19, HER2, EGFR). Each releases a different transcription factor (Gal4-VP64, LexA-p65, QF-AD). A synthetic AND gate promoter requires all three TFs for activation, driving a therapeutic payload (perforin and granzyme B). An OR gate promoter (activated by any single TF) drives a GFP reporter for tumor contact detection.",

    # === CRISPR therapeutic circuits ===
    "A CRISPR-based gene therapy for sickle cell disease: dCas9-KRAB is expressed from an erythroid-specific beta-globin LCR promoter to repress BCL11A, a fetal hemoglobin silencer. Simultaneously, Cas9 (from a separate module with its own U6-driven gRNA) disrupts the BCL11A erythroid enhancer permanently. A third module expresses a corrected beta-globin gene from its natural promoter. All cassettes are flanked by insulator elements (cHS4).",

    "A CRISPR-Cas13 RNA knockdown circuit for viral therapy: Cas13d (RfxCas13d) expressed from a constitutive CMV promoter with three guide RNAs (driven by separate U6 promoters) targeting three conserved regions of SARS-CoV-2 RNA. A fourth guide targets influenza PB2. An interferon-stimulated response element (ISRE) promoter drives additional Cas13d during active infection (positive feedback through interferon signaling). A GFP reporter under the ISRE promoter monitors activation.",

    "A CRISPRi-based genetic circuit with layered regulation: dCas9-KRAB expressed from a constitutive promoter targets three different promoters simultaneously using three gRNAs from U6 promoters. Target 1: represses TetR (which represses output GFP). Target 2: represses LacI (which represses mCherry). Target 3: represses an intermediate activator AraC. The net effect: GFP is ON (TetR repressed), mCherry is ON (LacI repressed), but AraC-dependent BFP is OFF.",

    "A base editing therapeutic circuit: an adenine base editor (ABE8e = TadA8e-nCas9-D10A) expressed from a liver-specific TTR promoter converts A-to-G at a specific position in PCSK9, introducing a loss-of-function mutation to lower cholesterol. A self-limiting module uses a gRNA that also targets the ABE expression cassette itself — once editing is complete, the ABE disables its own promoter. Includes a woodchuck hepatitis virus posttranscriptional regulatory element (WPRE) and polyadenylation signal.",

    # === Collins Lab circuits ===
    "A genetic counter using recombinases (Friedland et al. 2009): three sequentially activated memory modules. Module 1: arabinose induces Cre recombinase which flips a promoter between loxP sites, permanently activating Module 2. Module 2: now-active promoter drives Flp recombinase which flips a second promoter between FRT sites, activating Module 3. Module 3: drives PhiC31 integrase which flips a final cassette between attB/attP sites activating GFP. Each count state has a unique reporter (BFP=1, mCherry=2, GFP=3). Recombinases have ssrA degradation tags to prevent leaky counting.",

    "A synthetic gene oscillator with tunable period (Stricker et al. 2008): a hybrid pLac/ara promoter drives both AraC (activator) and LacI (repressor) in a dual-feedback loop. AraC activates its own promoter (positive feedback) while LacI represses it (negative feedback with delay due to LacI maturation). The imbalance between fast activation and slow repression generates oscillations. IPTG tunes the repression strength, controlling period. GFP reporter is co-expressed with LacI.",

    "A riboregulator-based logic circuit (Callura et al. 2010): four small RNAs (taRNAs) each activate translation of a different cognate crRNA-regulated mRNA. The mRNAs encode LacI, TetR, CI, and GFP. Upstream logic: IPTG controls taRNA-1 production, arabinose controls taRNA-2. taRNA-1 activates LacI mRNA translation, LacI represses taRNA-3 transcription. taRNA-3 would activate CI, so without taRNA-3, CI is off, and pCI drives GFP.",

    # === Voigt Lab — Cello designs ===
    "A Cello-designed 3-input circuit: three inputs (IPTG, arabinose, aTc) are processed through a layered NOT/NOR gate topology. Layer 1: IPTG drives PhlF repressor, arabinose drives SrpR repressor. Layer 2: PhlF represses BM3R1 expression, SrpR represses HlyIIR expression. Layer 3: BM3R1 and HlyIIR together repress YFP output. aTc directly represses an additional TetR gate that feeds into layer 2. Each gate is insulated with ribozyme insulators (RiboJ) and strong terminators.",

    "A Cello-designed 2-bit half adder: two inputs (IPTG and arabinose). Sum output (XOR logic): implemented by a series of NOR gates — IPTG drives AmtR, arabinose drives BetI, both feed into an intermediate NOR producing PhlF only when neither is present. A second path drives output when exactly one input is present. Carry output (AND logic): a separate NOR-NOR cascade produces carry only when both inputs are present. Sum is reported by YFP, carry by BFP. All gates use Cello-characterized repressor-promoter pairs.",

    # === Metabolic engineering circuits ===
    "An artemisinin biosynthesis pathway (Ro et al. 2006): five enzymes in the mevalonate pathway are expressed in two operons. Operon 1 (constitutive pTrc promoter): AtoB, HMGS, tHMGR each with their own RBS. Operon 2 (IPTG-inducible pLac): MK, PMK, PMD, IDI, FPS each with their own RBS. A third module expresses amorphadiene synthase (ADS) from a separate arabinose-inducible pBAD promoter. Downstream: CYP71AV1 and CPR for oxidation to artemisinic acid. Each operon has a B0015 terminator.",

    "A dynamic metabolic valve circuit: a malonyl-CoA sensor (FapR/FapO system) detects pathway intermediate buildup. High malonyl-CoA causes FapR to dissociate from fapO operators, derepressing expression of downstream fatty acid pathway enzymes (FabB, FabF, FabH each with own RBS from a single pFapO promoter). Simultaneously, a constitutive antisense RNA module (driven by pJ23119) knocks down competing pathway enzymes. A GFP reporter under a second fapO operator reports sensor state.",

    "A co-culture synthetic consortium for vitamin B12 production: Strain A (E. coli) expresses the aerobic B12 pathway (cobA through cobT, 8 enzymes in two operons under T7 promoter control). Strain B (Propionibacterium) expresses the anaerobic-specific steps. Cross-feeding is mediated by a cobinamide transporter (btuB) on Strain A. Population ratio is maintained by a mutual dependency: Strain A produces tryptophan (trpE operon) required by Strain B (trp auxotroph), while Strain B produces histidine (hisA operon) required by Strain A (his auxotroph).",

    # === Cell-free circuits ===
    "A cell-free transcription-translation circuit for diagnostics: a toehold switch RNA sensor detects Zika virus RNA. Upon binding, the toehold hairpin opens, exposing an RBS that enables translation of LacZ reporter. A T7 promoter drives the toehold switch transcript. A second module uses a CRISPR-Cas12a system: Cas12a with a Zika-targeting guide activates collateral ssDNA cleavage of a quenched fluorescent probe. Both detection modules are freeze-dried on paper and activated by adding water and sample.",

    "A cell-free genetic oscillator: a T7 RNAP positive feedback loop (T7 promoter drives T7 RNAP gene) coupled with sigma-28 negative feedback (T7 also drives sigma-28 which competes for core RNAP, reducing T7 RNAP activity). A ClpXP protease module degrades both T7 RNAP and sigma-28 at different rates, creating oscillatory dynamics. Energy regeneration is provided by a creatine kinase module. GFP and mCherry reporters on T7 and sigma-28 promoters respectively.",

    # === Published CAR-T designs ===
    "A Kymriah-inspired CAR construct: the CD19-targeting CAR consists of an anti-CD19 scFv (FMC63), a CD8alpha hinge and transmembrane domain, a 4-1BB costimulatory domain, and a CD3zeta signaling domain, all as a single CDS. Expression is driven by an EF1a promoter with a Kozak sequence. A separate cassette co-expresses truncated EGFR (EGFRt) as a safety/tracking marker via a T2A self-cleaving peptide from the same transcript. The construct is flanked by lentiviral LTRs for genomic integration.",

    "A fourth-generation CAR-T (TRUCK) circuit: an anti-HER2 CAR (scFv-CD28-CD3zeta) is driven by an EF1a promoter. Downstream, an NFAT-responsive promoter (6xNFAT-IL2mp) drives expression of IL-12 for autocrine/paracrine immune stimulation — this promoter is only active upon CAR engagement. A third module: constitutive PGK promoter drives a dominant-negative TGF-beta receptor (dnTGFBRII) to resist tumor immunosuppression. A rapamycin-inducible iCasp9 kill switch (FRB-FKBP-Casp9 fusion) under a constitutive promoter provides safety control.",

    "An armored CAR-T with checkpoint resistance: anti-CD19 CAR driven by EF1a promoter. A CRISPR module knocks out PD-1: Cas9 from a CMV promoter with a U6-driven gRNA targeting PDCD1. A third cassette secretes anti-CTLA4 scFv from an NFAT-responsive promoter (only during target engagement). A fourth module: constitutive expression of IL-7 receptor alpha chain for enhanced persistence. An EGFRt tracking marker is co-expressed with the CAR via P2A peptide.",

    "A logic-gated CAR-T for solid tumors: an AND gate requiring two antigens. Module 1: constitutive EF1a promoter drives a synNotch receptor targeting mesothelin. Upon mesothelin engagement, the intracellular Gal4-VP64 domain is released. Module 2: Gal4-VP64 activates a 5xUAS promoter driving an anti-HER2 CAR (scFv-4-1BB-CD3zeta). Module 3: the same UAS promoter also drives IL-21 secretion for immune potentiation. Module 4: a TRE promoter (doxycycline-inducible) drives iCasp9 for safety. Only cells in contact with mesothelin+ cells express the HER2 CAR.",

    # === Tumor-targeting bacteria (published) ===
    "A tumor-targeting E. coli circuit (Din et al. 2016 Nature): constitutive expression of LuxI produces AHL. At high density, AHL-LuxR activates pLux driving: (1) phage lysis gene E for synchronized population lysis, (2) haemolysin E (HlyE/ClyA) cytotoxin as therapeutic payload accumulated intracellularly. Post-lysis, surviving cells (stochastic non-lysing fraction ~1%) regrow and repeat. A pDawn light-sensor module (YF1/FixJ) optionally controls LuxI expression for external triggering. An ampicillin resistance marker enables selection.",

    "An engineered Salmonella for tumor immunotherapy: a hypoxia sensor (pepT promoter, activated under low oxygen in tumor core) drives a Flp recombinase. Flp excises a transcriptional terminator flanked by FRT sites, permanently activating a downstream pTac promoter driving flagellin B (FljB, a potent TLR5 agonist for immune activation). A second module: constitutive pJ23100 drives an anti-CD47 nanobody (blocking the don't-eat-me signal) fused to an OmpA surface display anchor. A third module: arabinose-inducible MazF toxin as external kill switch.",

    # === Gene therapy (published/clinical) ===
    "A Luxturna-inspired gene therapy construct: RPE65 coding sequence driven by a chicken beta-actin (CBA) promoter with CMV enhancer (CAG promoter). The construct includes: a beta-globin intron for enhanced expression, a Kozak consensus sequence before RPE65, and a bovine growth hormone polyadenylation signal. The entire cassette is flanked by AAV2 inverted terminal repeats (ITRs). A woodchuck hepatitis virus posttranscriptional regulatory element (WPRE) enhances mRNA stability.",

    "A Zolgensma-inspired gene therapy: SMN1 (survival motor neuron 1) coding sequence driven by a CBA promoter with CMV enhancer. A chimeric intron (from CMV/beta-globin) enhances expression. An SV40 polyadenylation signal terminates the transcript. The cassette is flanked by AAV9 ITRs for CNS tropism. A microRNA target sequence (miR-122 binding sites in the 3'UTR) detunes expression in liver (where AAV9 also transduces) to reduce hepatotoxicity.",

    "A self-complementary AAV gene therapy for hemophilia B: a codon-optimized Factor IX Padua variant (FIX-R338L) with enhanced activity is driven by a liver-specific LP1 promoter (ApoE enhancer + human alpha-1-antitrypsin promoter). A minute virus of mice (MVM) intron enhances nuclear export. The construct includes a synthetic polyadenylation signal. The cassette fits within the reduced scAAV packaging limit (~2.2kb). Wild-type AAV2 ITRs flank the construct, with one ITR carrying a deletion (delta-ITR) for self-complementary replication.",

    # === Biosafety circuits (published) ===
    "A synthetic auxotrophy kill switch (Mandell et al. 2015): the essential gene for diaminopimelate synthesis (dapA) is deleted from the chromosome and placed on a plasmid under control of an IPTG-inducible promoter. Without IPTG, no DAP is synthesized and cells die (DAP is required for cell wall synthesis). A second layer: a GFP-LVA fusion (GFP with LAA degradation tag) is constitutively expressed as a viability reporter. A third layer: an amber suppressor tRNA driven by an arabinose-inducible promoter is required to translate an essential gene (infA) containing premature amber stops.",

    "A deadman kill switch (Chan et al. 2016): a genetic timer that kills cells unless periodically reset. The circuit uses a toggle-like architecture: LacI represses expression of a toxin (EcoRI restriction enzyme). IPTG must be continuously supplied to keep LacI active. Without IPTG, LacI levels drop and the toxin is expressed. A second arm: TetR represses a second toxin (CcdB). aTc keeps TetR active. Both inputs must be maintained. Each arm has a positive feedback loop on the survival state to resist noise-induced switching. GFP reporter indicates the safe state.",

    "A PASSCODE kill switch (Chan et al. 2017): a multi-input dependent survival circuit. Three environmental signals (IPTG, arabinose, aTc) are required simultaneously. IPTG maintains LacI repression of toxin-1 (CcdB). Arabinose maintains AraC activation of an essential gene (folA). aTc maintains TetR repression of toxin-2 (EcoRI). Loss of any single input triggers cell death through a different mechanism. Each module has explicit operators, RBS sequences, and terminators. A GFP reporter under a promoter requiring all three TFs indicates full-survival state.",

    # === Optogenetic circuits (published) ===
    "An optogenetic toggle switch: blue light activates EL222 (LOV domain TF) which drives expression of LacI. LacI represses pLac driving TetR. In the dark, TetR is expressed and represses pTet driving LacI. This creates a light-responsive bistable switch. Blue light pushes the switch to LacI-high state (GFP reporter on pLac), darkness allows relaxation to TetR-high state (mCherry reporter on pTet). EL222 is constitutively expressed from a pJ23119 promoter.",

    "A CcaS/CcaR green light sensor circuit: CcaS (membrane histidine kinase) is constitutively expressed and phosphorylates CcaR (response regulator) under green light. Phospho-CcaR activates the cpcG2 promoter driving GFP output. Under red light, CcaS reverses phosphorylation (phosphatase mode), turning off GFP. Phycocyanobilin (PCB) chromophore is provided by co-expression of ho1 and pcyA biosynthesis genes from a constitutive promoter. All components have individual RBS and terminators.",

    "A two-channel optogenetic pattern generator: Channel 1 uses blue-light-activated EL222 driving LacI, which represses pLac-GFP. Channel 2 uses red-light-activated PhyB-PIF3 dimerization: constitutive PhyB expression and pConst-PIF3-VP16. Under red light, PhyB-PIF3-VP16 complex activates a UAS promoter driving mCherry. Each channel is independently addressable with different wavelengths. A T7-based amplification loop (EL222 also drives T7 RNAP which amplifies the GFP-repression signal through pT7-LacI) provides signal gain in Channel 1.",

    # === Quorum sensing circuits ===
    "An orthogonal two-channel quorum sensing circuit: Channel 1 uses LuxI/LuxR with AHL (3OC6HSL). Channel 2 uses RhlI/RhlR with C4-HSL. Each channel independently controls a different reporter: pLux drives GFP, pRhl drives mCherry. Cross-talk is minimized by using engineered LuxR and RhlR variants with improved specificity. Both LuxI and RhlI are under separate inducible promoters (pBAD for LuxI, pTet for RhlI). LuxR and RhlR are constitutively expressed.",

    "A quorum sensing-based population density controller: LuxI (constitutive) produces AHL. At low density, AHL is diluted and LuxR is inactive. At high density, AHL-LuxR activates pLux driving: (1) a growth-arresting gene SulA (inhibits cell division), and (2) a toxin CcdB. An antitoxin CcdA is constitutively expressed at a level that neutralizes low CcdB but is overwhelmed at high expression. This creates a population density ceiling. GFP under pLux reports density state.",

    # === RNA circuits ===
    "A toehold switch cascade for multi-level signal processing: Input RNA-A activates Toehold Switch 1, which translates TF-1 (an activator). TF-1 drives production of Trigger RNA-B from a TF-1-responsive promoter. RNA-B activates Toehold Switch 2, translating TF-2. TF-2 drives Trigger RNA-C. RNA-C activates Toehold Switch 3, translating GFP output. Each toehold switch has a unique RNA hairpin structure and trigger sequence. A T7 promoter drives the initial input RNA-A production in response to an external signal.",

    "A small transcription activating RNA (STAR) logic circuit: two input STARs (STAR-A induced by IPTG, STAR-B induced by arabinose) each activate translation of a different target mRNA. STAR-A activates LacI mRNA, STAR-B activates TetR mRNA. LacI represses pLac-GFP, TetR represses pTet-mCherry. A third STAR (STAR-C, constitutive) activates a bifunctional antisense RNA that can be sequestered by STAR-A. The net logic: IPTG activates LacI (represses GFP) AND sequesters the antisense (activates BFP). Arabinose activates TetR (represses mCherry).",

    # === More CAR-T variants ===
    "A switchable universal CAR (UniCAR) system: a constitutive EF1a promoter drives expression of a CAR with an anti-FITC scFv (instead of anti-tumor). Separately, bispecific adapter molecules (FITC conjugated to anti-CD19 scFv) are administered as drugs. The adapter bridges the CAR-T to CD19+ tumor cells. Removing the adapter turns off killing. An NFAT-responsive promoter drives IL-2 for T-cell expansion only during active engagement. A separate EGFRt safety marker enables antibody-mediated depletion (cetuximab).",

    "A self-driving CAR-T with metabolic reprogramming: an anti-CD19 CAR under EF1a promoter. Module 2: NFAT-responsive promoter drives PGC1-alpha (metabolic regulator) to enhance mitochondrial biogenesis and oxidative phosphorylation in the tumor microenvironment. Module 3: constitutive PGK promoter drives a dominant-negative form of HIF-1alpha to prevent exhaustion-associated hypoxic signaling. Module 4: a TGF-beta-responsive promoter paradoxically drives a costimulatory signal (4-1BBL) — the immunosuppressive tumor environment actually boosts T-cell activation.",

    "A dual-targeting tandem CAR: two CARs on the same T-cell. CAR-1: anti-CD19 scFv-CD28-CD3zeta driven by EF1a promoter. CAR-2: anti-CD22 scFv-4-1BB-CD3zeta driven by PGK promoter, linked to CAR-1 via a P2A self-cleaving peptide in a single polycistronic transcript. An NFAT-responsive promoter drives secretion of a bispecific T-cell engager (BiTE) targeting CD20, recruiting bystander T-cells. A tetracycline-inducible iCasp9 provides safety.",

    # === Immunotherapy circuits ===
    "A synthetic Treg (regulatory T-cell) circuit for autoimmune disease: a constitutive EF1a promoter drives an anti-MOG CAR (targeting myelin for multiple sclerosis). CAR engagement activates an NFAT promoter driving: (1) FOXP3 master Treg TF for immunosuppressive phenotype, (2) IL-10 anti-inflammatory cytokine, (3) CTLA-4 for suppressing neighboring immune cells. FOXP3 feeds back to activate its own endogenous promoter (positive feedback for stable Treg commitment). An iCasp9 kill switch on a separate constitutive promoter.",

    "An NK cell-based immunotherapy circuit: a constitutive EF1a promoter drives an anti-GPC3 CAR (targeting liver cancer) with NKG2D transmembrane domain (native NK activating receptor), 2B4 costimulatory domain, and CD3zeta signaling domain. Module 2: IL-15/IL-15Ra complex is constitutively co-expressed (PGK promoter) for enhanced NK cell persistence without exogenous cytokine support. Module 3: a soluble NKG2D-Fc decoy (from NFAT-responsive promoter) blocks tumor NKG2D ligand shedding which normally suppresses NK cells.",

    "A checkpoint-armored macrophage (CAR-M) circuit: an anti-HER2 CAR-phagocytosis receptor (scFv-CD8 hinge-FcRgamma) is driven by a constitutive CMV promoter in macrophages. Module 2: a SIRPalpha knockout construct using CRISPR (CMV-Cas9 + U6-gRNA targeting SIRPA) removes the don't-eat-me checkpoint. Module 3: a constitutive promoter drives pro-inflammatory cytokines (TNF-alpha and IFN-gamma) via a bicistronic cassette with P2A peptide for M1 polarization. Module 4: an NF-kB-responsive promoter drives iNOS for additional tumor killing via nitric oxide.",

    # === Advanced kill switches ===
    "An addiction-based biocontainment circuit: an essential gene (thyA for thymidylate synthase) is placed under control of a promoter that requires a synthetic non-natural amino acid (pAzF) for activation. Without pAzF supplementation, thyA is not expressed and cells die. A second layer: an orthogonal ribosome (ribo-X) is required to translate a second essential gene (infA) that has been recoded to only work with ribo-X. Ribo-X expression requires a second synthetic molecule. A third layer: a programmed population limiter using a density-dependent toxin (CcdB under pLux/LuxR quorum sensing) prevents uncontrolled growth.",

    "A temperature-sensitive kill switch: a thermosensitive CI857 repressor is constitutively expressed and represses pR driving the CcdB toxin. At normal body temperature (37C), CI857 is active and CcdB is repressed. If cells escape the body and encounter lower temperatures, CI857 still works (more stable at low temp). But if cells encounter fever temperatures (>39C), CI857 unfolds, derepressing CcdB and killing the cell. A second arm: a cold-sensitive TlpA39 repressor (active <30C) represses a second toxin (MazF). If cells are between 30-39C, both toxins are repressed (safe range).",

    # === Tissue engineering ===
    "A morphogen gradient circuit for neural tube patterning: a source cell constitutively secretes Sonic Hedgehog (Shh) with a GFP tag from CMV promoter. Receiving cells have three reporter modules at different concentration thresholds. High Shh: a high-affinity Gli-binding promoter (8xGli-BS) drives NKX2.2-BFP. Medium Shh: a medium-affinity promoter (4xGli-BS) drives OLIG2-mCherry. Low Shh: a low-affinity promoter (2xGli-BS) drives PAX6-YFP. Each receiving cell module has a negative feedback: each TF represses the promoters of adjacent fates (NKX2.2 represses OLIG2 promoter, OLIG2 represses both neighbors).",

    "A synthetic Wnt signaling circuit for organoid patterning: a doxycycline-inducible TRE promoter drives Wnt3a secretion in source cells. Receiving cells have a TCF/LEF-responsive promoter (7xTCF) driving Lgr5-GFP (stem cell marker). A negative feedback: Wnt-responsive DKK1 (Wnt antagonist) is also driven by 7xTCF, creating a self-limiting gradient. A second module: BMP4 under a constitutive promoter drives differentiation. BMP4-responsive pSMAD promoter drives CDX2-mCherry (differentiation marker). The balance between Wnt and BMP4 determines stem vs differentiated fate.",

    "A sequential differentiation timer using a cascade of TFs: constitutive Oct4 maintains pluripotency. Doxycycline (stage trigger) activates TRE-driven Brachyury (mesoderm). Brachyury activates a Brachyury-responsive promoter driving Mesp1 (cardiac progenitor) AND a delayed self-repressor (Brachyury activates expression of miR-430 which degrades Brachyury mRNA after a delay). Mesp1 activates Mesp1-responsive promoter driving Nkx2.5 (cardiomyocyte). Each stage has a reporter: Oct4→GFP, Brachyury→BFP, Mesp1→YFP, Nkx2.5→mCherry. Each transition is irreversible due to positive autoregulation at each stage.",

    # === Metabolic disease therapy ===
    "A glucose-responsive insulin circuit (Ye et al. 2011): a calcium-responsive NFAT promoter drives insulin expression. Upstream: a chimeric glucose receptor (glucose transporter GLUT2 fused to a calcium channel TRPM2 variant) imports calcium upon glucose binding. High glucose → calcium influx → NFAT activation → insulin secretion. Negative feedback: secreted insulin activates an insulin receptor (INSR) on the same cell, which through IRS-PI3K-Akt pathway activates a FoxO-responsive promoter driving a repressor (KRAB domain fused to a DNA-binding protein targeting the NFAT promoter). Includes all signaling components and a GFP reporter under NFAT for monitoring.",

    "A urate-responsive circuit for gout therapy: a urate oxidase (UOx) enzyme converts uric acid to allantoin. Expression is controlled by a urate-responsive riboswitch in the 5'UTR that enhances translation when urate is present. A constitutive promoter drives the riboswitch-UOx mRNA. A second module: xanthine dehydrogenase (XDH) expressed from a HIF-responsive promoter activates only during inflammatory hypoxia in gouty joints. A secretion signal peptide (PelB) enables enzyme release. The construct is delivered in an engineered Lactobacillus for gut-based therapy with a bile salt-responsive promoter controlling colonization genes.",

    "A phenylketonuria combination therapy: Module 1: phenylalanine ammonia lyase (PAL) converts phenylalanine to trans-cinnamic acid, expressed from a constitutive promoter in engineered E. coli Nissle 1917. Module 2: phenylalanine transporter (PheP) overexpressed from a strong RBS to increase uptake. Module 3: a phenylalanine-responsive riboswitch controls expression of additional PAL copies — when Phe is high, more PAL is produced. Module 4: a tetracycline-inducible L-amino acid deaminase (LAAD) provides an alternative degradation pathway. Module 5: a FNR-responsive promoter drives the entire therapeutic cassette specifically under the anaerobic conditions of the gut. GFP reporter under the riboswitch for monitoring.",

    # === Microbiome circuits ===
    "A synthetic probiotic for C. difficile infection: an engineered Lactobacillus reuteri detects C. difficile toxin TcdB via a TcdB-responsive promoter (using a chimeric receptor). Detection activates: (1) endolysin LysCD (C. difficile-specific bacteriocin), (2) alanine racemase (Alr) which produces D-alanine that inhibits C. difficile germination, (3) a bile salt hydrolase (BSH) that modifies bile acids to inhibit C. difficile growth. A constitutive promoter drives mucus-binding protein (MUB) for gut colonization. A thymidine auxotrophy (thyA deletion) ensures containment — cells die without thymidine supplementation.",

    "A gut inflammation sentinel: engineered E. coli Nissle 1917 with three inflammation sensors. Sensor 1: a tetrathionate-responsive TtrS/TtrR drives GFP (early inflammation). Sensor 2: a nitrate-responsive NarX/NarL drives mCherry (medium inflammation). Sensor 3: a calprotectin-responsive promoter drives BFP (severe inflammation). Each sensor output is also connected to a recombinase-based memory module: Sensor 1 activates Cre (flips loxP memory), Sensor 2 activates Flp (flips FRT memory), creating a permanent record of inflammation history. A therapeutic arm: combined sensor 2+3 activation drives IL-10 secretion via a two-input AND gate.",

    # === Neurological therapy ===
    "A closed-loop deep brain stimulation circuit: channelrhodopsin-2 (ChR2) expressed from a CaMKII promoter (excitatory neurons) enables optical stimulation. A calcium indicator (GCaMP6) co-expressed from the same promoter reports neural activity. An engineered luciferase (NanoLuc) driven by a c-Fos activity-dependent promoter provides light output proportional to seizure activity, which feeds back to activate ChR2 in inhibitory interneurons (via a VGAT promoter driving halorhodopsin for inhibition). The balance creates a closed-loop: seizure activity → NanoLuc light → halorhodopsin inhibition → reduced seizure → reduced NanoLuc → reduced inhibition.",

    "A gene therapy for Duchenne muscular dystrophy: a micro-dystrophin (truncated functional dystrophin fitting in AAV) is driven by a muscle-specific CK8 promoter. A second cassette: micro-utrophin (dystrophin homolog) under a MCK promoter provides redundant structural support. A third module: follistatin (myostatin antagonist) under a CMV promoter promotes muscle growth. The constructs are in two separate AAV vectors with split-intein reconstitution: AAV1 carries the 5' half of micro-dystrophin fused to IntN under CK8 promoter, AAV2 carries the 3' half fused to IntC under CK8 promoter. Full-length protein is reconstituted by intein-mediated trans-splicing only in cells transduced by both vectors.",

    # === Wound healing / Regeneration ===
    "An engineered probiotic bandage for chronic wounds: constitutive expression of antimicrobial peptide cecropin-A provides broad-spectrum antimicrobial protection. A hypoxia sensor (FNR) detects wound hypoxia and drives VEGF for angiogenesis. A pH sensor (CadC, activated at acidic wound pH) drives PDGF-BB for fibroblast recruitment. A quorum sensing density limiter (LuxI/LuxR → SulA growth arrest at high density) prevents overgrowth. Wound healing progress is monitored by an oxygen-responsive GFP reporter (shifting from FNR-active to aerobic). A temperature-sensitive CI857 kill switch ensures cells die if they escape the wound environment (below 30C).",

    # === More advanced regulation ===
    "A genetic band-pass filter with tunable window: an IPTG concentration-dependent circuit. At low IPTG: LacI is active, represses pLac-GFP (output OFF). At medium IPTG: LacI is partially titrated, GFP is partially expressed, but a second repressor TetR driven by a high-cooperativity pLac variant (with three tandem lacO operators, requiring lower LacI to derepress) is not yet expressed (output ON). At high IPTG: the high-cooperativity pLac variant derepresses TetR, which represses GFP through tetO operators (output OFF). Net result: GFP only at intermediate IPTG. The band-pass window width is tuned by adjusting TetR RBS strength.",

    "A coherent feedforward loop with AND-gate logic: an input signal (arabinose) activates AraC which directly activates pBAD driving GFP. AraC also activates a slow intermediate (LuxR expression from pBAD2 with a weak RBS and ssrA degradation tag). LuxR requires AHL (provided by constitutive LuxI) to become active and also activates GFP from a hybrid pBAD-pLux promoter. Both AraC AND active LuxR are needed for full GFP expression. This creates a delay filter: transient arabinose pulses don't activate GFP (LuxR hasn't accumulated yet), but sustained arabinose does.",

    "An incoherent feedforward loop pulse generator: IPTG simultaneously activates two paths. Path 1 (fast): pLac directly drives GFP through a strong RBS. Path 2 (slow): pLac drives LacI-LVA (LacI with LAA degradation tag) through a weak RBS. LacI-LVA represses a second pLac (with tandem lacO operators) driving GFP through a different RBS. Net effect: IPTG addition causes fast GFP activation followed by slow repression, creating a pulse. Pulse width is controlled by the degradation rate of LacI-LVA. mCherry under constitutive promoter serves as normalization control.",

    "A mutual activation positive feedback switch: AraC (activated by arabinose) drives pBAD expressing LuxR. LuxR (activated by AHL from constitutive LuxI) drives pLux expressing AraC. Once triggered by transient arabinose, the mutual activation maintains both TFs in the ON state permanently (hysteresis). A GFP reporter on pLux indicates the ON state. Breaking the loop requires either degradation tags on both TFs or expression of a dominant-negative AraC variant from a tetracycline-inducible safety switch.",

    # === More therapeutic circuits ===
    "An engineered red blood cell for enzyme replacement therapy: during erythropoiesis, a GATA1-responsive promoter (active during RBC differentiation) drives expression of adenosine deaminase (ADA) for ADA-SCID. The enzyme is trapped inside the RBC upon enucleation. A sortase-mediated surface display module anchors a PEG-binding peptide for extended circulation. A second enzyme (phenylalanine ammonia lyase for PKU) is co-expressed via P2A peptide. A GFP-based maturation reporter under a beta-globin promoter tracks successful differentiation.",

    "An anti-fibrotic gene therapy for liver fibrosis: a hepatic stellate cell (HSC)-specific GFAP promoter drives expression of relaxin-2 (anti-fibrotic hormone). A second module: TGF-beta-responsive promoter (SMAD-binding elements) drives a dominant-negative TGF-beta receptor type II (dnTGFBRII) to block pro-fibrotic signaling. A third module: a MMP-2 metalloprotease under a constitutive promoter degrades excess collagen. A miR-29 expression cassette (driven by a CMV promoter) represses COL1A1 and COL3A1 (collagen genes). A GFP reporter under the GFAP promoter confirms HSC-specific expression.",

    "A diabetes treatment using engineered beta cells: a glucose-responsive GLP1 promoter drives expression of proinsulin. The proinsulin is processed by co-expressed PC1/3 and PC2 convertases (from constitutive promoters) into mature insulin in secretory granules. A GLP1R (GLP-1 receptor) amplification loop: secreted insulin activates neighboring beta cells via insulin receptor, which activates IRS2 promoting beta cell survival. A KATP channel sensor links glucose metabolism to membrane depolarization, triggering calcium-dependent insulin granule exocytosis. A constitutive ABCC8 (SUR1 sulfonylurea receptor) expression enables pharmacological control with sulfonylurea drugs.",

    # === Cancer detection ===
    "A circulating tumor cell (CTC) detection circuit: an EpCAM-responsive synNotch receptor on engineered T-cells detects CTCs. Activation releases a Gal4-VP64 TF that activates UAS-driven secreted NanoLuc luciferase (detectable in blood with a simple luminescence assay). A second synNotch targeting HER2 releases a LexA-VP16 TF activating lexAop-driven SEAP (secreted alkaline phosphatase, detectable via standard blood test). Both reporters persist in blood for hours after CTC contact, enabling liquid biopsy. An iCasp9 kill switch prevents the engineered T-cells from attacking (they only detect and report).",

    "A synthetic biomarker amplification circuit for early cancer detection: a weak tumor-associated promoter (survivin promoter, active in cancer cells) drives T7 RNA polymerase. T7 RNAP amplifies the signal 1000-fold by driving a T7 promoter expressing a secreted biomarker (human chorionic gonadotropin fragment, detectable by standard pregnancy test). A second amplification stage: T7 also drives LuxI which produces AHL in a positive feedback loop (AHL-LuxR drives more T7). A timer module using a delayed-action repressor (ssrA-tagged TetR accumulation) eventually shuts down the amplification to prevent false sustained signals. Delivered via tumor-targeting AAV with integrin-binding peptide on capsid.",

    # === Drug production ===
    "An engineered yeast for opioid biosynthesis (Galanie et al. 2015-inspired): a 20+ enzyme pathway from glucose to thebaine. Module 1: tyrosine overproduction (ARO4-K229L feedback-resistant DAHP synthase, ARO7-T226I feedback-resistant chorismate mutase from constitutive promoters). Module 2: dopamine pathway (TyrH tyrosine hydroxylase and DODC DOPA decarboxylase from pGAL1 galactose-inducible promoter). Module 3: norcoclaurine synthase (NCS) from pTEF constitutive. Module 4: methyltransferases 6OMT and CNMT from pGPD. Module 5: CYP80B1 and CPR from pGAL10. Module 6: salutaridine synthase (SalSyn), salutaridine reductase (SalR), salutaridinol acetyltransferase (SalAT) from pGAL1. Flux sensors using GFP under intermediate-responsive promoters at key branchpoints.",

    # === More complex regulatory circuits ===
    "A winner-take-all competition circuit for cell-fate decision: three mutually inhibitory TFs form a tristable network. LacI (from pCI promoter), CI (from pTet promoter), TetR (from pLac promoter). Each represses the next. Additionally, each TF positively autoregulates by weakly activating its own promoter through an intermediate activator. LacI drives AraC which activates pBAD-LacI amplification. CI drives LuxR which with constitutive AHL activates pLux-CI amplification. TetR drives RhlR which with constitutive C4-HSL activates pRhl-TetR amplification. Each state has a reporter (GFP, mCherry, BFP). The winning state depends on initial conditions and noise.",

    "A genetic memory register with read-write capability: a serine integrase (PhiC31) flips a DNA segment between attB and attP sites, switching between two stable states. State A: promoter facing forward drives GFP. State B: promoter facing reverse drives mCherry. Writing State A→B: arabinose-inducible PhiC31 integrase flips the segment. Writing State B→A: IPTG-inducible PhiC31 recombination directionality factor (RDF) + PhiC31 reverses the flip. Reading: GFP fluorescence = State A, mCherry = State B. A second independent memory bit uses Bxb1 integrase with its own attB/attP sites and reporters (BFP/YFP), giving a 2-bit memory (4 possible states).",

    "A genetic amplitude modulation circuit: a constitutive promoter drives an mRNA with a theophylline-responsive riboswitch in the 5'UTR controlling GFP translation efficiency. Theophylline concentration directly modulates GFP protein level (analog output, not digital). A second channel: a tetracycline-responsive riboswitch controls mCherry translation. A third channel: an adenine-responsive riboswitch controls BFP. Each riboswitch provides independent analog control of its reporter. A normalization module: constitutive RFP (no riboswitch) provides a reference signal for ratiometric measurements.",

    # === Combination therapies ===
    "A multi-modal cancer therapy bacterium: engineered EcN with five therapeutic arms. Arm 1: hypoxia sensor (FNR) drives anti-PD-L1 nanobody (checkpoint inhibitor). Arm 2: lactate sensor (LldR) drives STING agonist (cdGAMP synthase, cGAS, for innate immune activation). Arm 3: constitutive promoter drives CD47 nanobody (blocks don't-eat-me signal). Arm 4: quorum sensing (LuxI/LuxR) synchronized lysis releases all accumulated intracellular therapeutics. Arm 5: arabinose-inducible MazF kill switch for safety. A GFP reporter under pLux monitors population density dynamics.",

    "A three-pronged autoimmune therapy: engineered Tregs with three functional modules. Module 1: anti-MOG CAR (for MS) or anti-insulin CAR (for T1D) under EF1a promoter enables targeting of autoimmune sites. Module 2: NFAT-responsive promoter (activated by CAR engagement) drives immunosuppressive payload — IL-10 and TGF-beta from a bicistronic cassette with P2A peptide. Module 3: constitutive PGK promoter drives CTLA-4-Ig fusion protein that is surface-displayed to suppress neighboring effector T-cells. A stability module: FOXP3 under a constitutive promoter with a positive feedback loop (FOXP3 activates its own CNS2 enhancer) maintains Treg identity. Safety: iCasp9 under tetracycline-inducible TRE promoter.",

    # === Biosensor circuits ===
    "A multiplexed heavy metal biosensor array: five independent sensor modules. Sensor 1: MerR/pMer detects mercury → GFP. Sensor 2: ArsR/pArs detects arsenic → mCherry. Sensor 3: CadC/pCad detects cadmium → BFP. Sensor 4: CueR/pCopA detects copper → YFP. Sensor 5: ZntR/pZnt detects zinc → mTurquoise. Each sensor has its own RBS and terminator. A normalization module: constitutive RFP provides a reference for ratiometric quantification. A signal amplification module: each sensor also drives T7 RNAP from a weak RBS, and a T7 promoter drives additional reporter copies for 10x signal boost.",

    "A two-input analog-to-digital converter: continuous IPTG concentration is converted to a 3-bit digital output. Bit 0 (LSB): a low-threshold pLac variant (single lacO, easily derepressed) drives GFP — ON above ~10uM IPTG. Bit 1: a medium-threshold pLac (two lacO operators, cooperative LacI binding) drives mCherry — ON above ~100uM IPTG. Bit 2 (MSB): a high-threshold pLac (three lacO operators, highly cooperative) drives BFP — ON above ~1mM IPTG. Each output is sharpened by a positive feedback loop (each reporter also drives additional copies of itself through a secondary promoter). All eight states (000 through 111) produce distinct fluorescence patterns.",

    # === Additional complex circuits ===
    "A genetic PID controller for precise gene expression: Proportional: the error signal (difference between desired and actual protein level, sensed by a protein-responsive riboswitch) directly drives output gene expression. Integral: a slow-accumulating integrator protein (ssrA-tagged, long half-life variant) driven by the same error signal provides integral control. Derivative: a fast-folding activator and a slow-folding repressor both driven by the error signal create a derivative term (responds to rate of change). All three terms converge on a hybrid promoter with binding sites for all three regulators, driving the target gene GFP. A reference signal is set by a constitutive RFP expression level.",

    "A synthetic predator-prey ecosystem: Predator cells (E. coli strain A) constitutively express an AHL-responsive lysis gene (under pLux). Prey cells (E. coli strain B) constitutively produce AHL via LuxI. At high prey density, AHL accumulates and activates predator lysis, releasing a bacteriocin (colicin) that kills prey. Declining prey → declining AHL → predator survival → predator growth → when prey regrows, cycle repeats. Prey has colicin immunity gene (constitutive). Predator has GFP reporter, prey has mCherry reporter. Both strains have orthogonal antibiotic resistance (AmpR vs KanR).",

    "A synthetic stripe-forming circuit (Schaerli et al. 2014-inspired): a morphogen gradient is created by a source cell constitutively producing AHL. Receiving cells have a concentration-dependent response: at low AHL: LuxR is inactive, a constitutive low-level CI repressor keeps mCherry OFF and GFP ON (green stripe, far from source). At medium AHL: LuxR-AHL activates pLux driving both mCherry (red stripe) AND a high-threshold repressor TetR. TetR represses GFP. At high AHL: very strong pLux activation drives excess TetR which represses both GFP and mCherry, and activates BFP from a TetR-activated promoter (blue stripe, near source). Result: blue-red-green stripes.",
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
2. Regulation targets should be promoters or operators, NOT other CDS directly
3. Include ALL components explicitly mentioned in the description
4. For complex circuits with many modules, include all modules with their complete promoter-RBS-CDS-terminator structure
5. For feedback loops, show the full regulatory chain
6. Use descriptive snake_case IDs (e.g., laci_cds, plac_promoter, gfp_reporter)
7. Include operators explicitly when the description mentions binding sites
8. Include 'other' type for degradation tags, signal peptides, self-cleaving peptides, ITRs, insulators, recombinase sites

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
                max_completion_tokens=4096,
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

            if n_comp < 4:
                print(f"    SKIP: Only {n_comp} components")
                failed.append(i)
                continue

            if n_inter < 2:
                print(f"    SKIP: Only {n_inter} interactions")
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

    # Retry failed ones once
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
                    max_completion_tokens=4096,
                    temperature=0.5,  # lower temp for retry
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
                if len(parsed['components']) >= 4 and len(parsed['interactions']) >= 2:
                    results.append({"description": desc, "circuit": parsed})
                    print(f"    OK (retry)")
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

    descs = PUBLISHED_CIRCUITS
    print(f"=== Generating {len(descs)} complex circuit examples via GPT-5.4 ===\n")

    results = generate_with_gpt(descs, args.openai_key)

    # Save raw
    raw_path = output_dir / 'complex_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)

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

    training_path = output_dir / 'complex_training.jsonl'
    with open(training_path, 'w') as f:
        for entry in training:
            f.write(json.dumps(entry) + '\n')

    # Stats
    total_comps = sum(len(r['circuit']['components']) for r in results)
    total_inters = sum(len(r['circuit']['interactions']) for r in results)
    from collections import Counter
    inter_types = Counter()
    for r in results:
        for ix in r['circuit']['interactions']:
            inter_types[ix['type']] += 1

    print(f"\n=== SUMMARY ===")
    print(f"Generated: {len(results)}/{len(descs)}")
    print(f"Saved to: {training_path}")
    print(f"Avg components: {total_comps/max(len(results),1):.1f}")
    print(f"Avg interactions: {total_inters/max(len(results),1):.1f}")
    print(f"Interaction types: {dict(inter_types)}")


if __name__ == '__main__':
    main()

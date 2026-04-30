# Biology validation — Cell D (LoRA 8-bit + Chen prompt) on 11 hardest items

Judge: Claude Opus 4.7 (Newgenes session). Items: 11 (all diff=5 + all cycle-topologies: toggle, feedback, oscillator).

## Aggregate

| Metric | Score |
|---|---|
| Structural rubric (sbol_eval_v2) avg | **91.8** |
| Biology rubric (expert review) avg | **71.4** |
| Gap | **-20.4 pts** |

## Per-item verdicts

| # | Prompt | Topo | Struct | Bio | Verdict |
|---|---|---|---|---|---|
| 1 | CRISPRi-based toggle | toggle | 96 | 50 | logic inversion (self-repression not cross) |
| 2 | Mammalian TetR/PipR toggle | toggle | 93 | 40 | logic inversion (constitutive CMV readouts) |
| 3 | Elowitz repressilator | oscillator | 99 | 90 | correct |
| 4 | TetR negative autoregulation | feedback | 93 | 92 | correct |
| 5 | Beta-carotene (CrtE/B/I/Y) | pathway | 82 | 85 | correct (simplified operon RBS) |
| 6 | CRISPRa dCas9-VP64 | crispr | 90 | 85 | correct (chassis override to mammalian) |
| 7 | Dual-input CcdA/CcdB kill switch | kill | 88 | 50 | NAND logic wrong |
| 8 | E. coli Nissle tumor-targeting | pathway | 94 | 88 | correct |
| 9 | Yeast GAL80-KO + GAL4-VP64 | pathway | 93 | 80 | correct (simplified) |
| 10 | Cas13 RNA sensor | crispr | 92 | 60 | wrong sensing mode (direct vs collateral) |
| 11 | Cas12a TX-TL DNA sensor | crispr | 92 | 65 | reporter misrepresented as CDS |

## Strengths

- **Feedback loops** (autoregulation): canonical TetR self-repression implemented correctly.
- **Oscillators** (repressilator): 3-node cyclic repression graph built accurately.
- **Metabolic pathways** (β-carotene, GAL, Nissle): correct enzyme order, chassis-appropriate parts, right regulatory logic.
- **Chassis override**: when prompt label conflicts with biological reality (E. coli + VP64), model correctly chose mammalian.

## Weaknesses

- **Toggle switches** (both attempts inverted): produces a cyclic graph on paper but implements self-repression or dangling operators — no bistability. Structural rubric can't detect this.
- **Multi-input logic** (kill switch NAND): model parses "both inducers required" as separate TUs rather than a conjunction; resulting circuit has the opposite phenotype.
- **Novel sensing modalities** (Cas13 collateral, Cas12a DETECTR): model defaults to direct-cleavage semantics and cannot represent the trigger-RNA / ssDNA-reporter distinctions natively.

## Implication for the presentation

The 20-point gap is the honest answer to "are these designs buildable?" — they are for simple feedback/pathway topologies, but the model has a systematic blind spot for logic primitives (toggle, NAND, collateral cleavage). The structural rubric measures well-formedness; the biology rubric measures *functional* fidelity. Future work: add a 7th axis to `sbol_eval_v2` that checks whether the behavior text's stated logic is actually implemented by the interaction signs.

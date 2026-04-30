# The 80/20 Redundancy — Prompt Engineering vs Fine-Tuning

## The setup

We ran a controlled 2×2 on an identical 34-prompt subset (s3), four cells:

|   | Default prompt | Chen 2026 prompt |
|---|---|---|
| **Q4 base, no LoRA** | A | B |
| **Q8 LoRA, same base** | C | D |

Axis rubric (max 100): Structural Validity (20) + Behavioral Wiring (20) + Biology Accuracy (20) + Prompt Fidelity (20) + Design Quality (10) + Engineering Realism (10).

## The result

| Cell | Total | ΔA |
|---|---:|---:|
| **A** (baseline)    | 89.71 |    — |
| **B** (+Chen only)  | 92.68 | +2.97 |
| **C** (+LoRA only)  | 92.21 | +2.50 |
| **D** (+both)       | 93.18 | +3.47 |

**Prompt engineering alone (+2.97) and LoRA alone (+2.50) are nearly equivalent.**
Stacking both is only +0.50 better than either single intervention.

## Interpretation

On easy-to-medium circuits at Q8 quant, a careful prompt and a fine-tuned model
buy you **roughly the same thing**: a cleaner JSON schema discipline and a
slightly better biology-accuracy score. They are **~80% redundant**.

## Where LoRA pulls ahead — the complexity tail

Run the same ablation at **Q3 quant on Jetson**, where the model is compressed
4× harder, and the picture changes:

| Diff | LoRA mean | bare mean | Δ | bare min | catastrophes |
|:---:|---:|---:|---:|---:|:---:|
| d1 | 90.75 | 90.25 | +0.50 | 83 | 0 |
| d2 | 90.80 | 86.50 | +4.30 | **0** | 1× parse fail |
| d3 | 89.60 | 89.30 | +0.30 | 80 | 0 |
| d4 | 90.35 | 89.45 | +0.90 | 78 | 0 |
| d5 | **86.50** | **80.35** | **+6.15** | **0** | 1× parse fail |

LoRA's floor is **78** across all 100 prompts. Bare's floor is **0**.

The +2.43 point mean-gap under-sells the real value: LoRA **eliminates tail
risk** — zero parse failures, zero truncations, zero sub-50 scores —
even on CRISPR, quorum-sensing, and metabolic-pathway designs at Q3 quant.

## Per-topology gaps (s3 + Opus)

|  | A | B | C | D | Opus | D−A | Opus−D |
|---|---:|---:|---:|---:|---:|---:|---:|
| reporter   | 89.0 | 94.4 | 94.1 | 94.6 | 98.2 | +5.6 | +3.6 |
| inducible  | 89.6 | 90.7 | 90.3 | 91.1 | 99.0 | +1.6 | +7.9 |
| biosensor  | 91.2 | 93.2 | 90.0 | 92.0 | 100  | +0.8 | +8.0 |
| gate       | 92.2 | 96.2 | 96.5 | 97.5 | 99.1 | +5.2 | +1.6 |
| oscillator | 92.0 | 99.0 | 96.0 | 99.0 | 100  | +7.0 | +1.0 |
| crispr     | 84.7 | 87.3 | 87.7 | **91.3** | 100 | **+6.7** | +8.7 |
| pathway    | 87.3 | 89.0 | 90.7 | 89.7 | 100  | +2.3 | +10.3 |

The biggest Jetson→Opus gaps are on **pathway, biosensor, inducible, crispr** —
topologies requiring either deep biology knowledge (pathway) or precise
regulatory wiring (crispr). LoRA closes CRISPR the most (+6.7), which is
also where frontier models have the largest headroom.

## One caveat: the rubric is structural, not biological

Our 100-point scale puts 80 points on structural + wiring correctness and only
20 points on biology (BA + DQ partly). A model that emits clean JSON with
plausible-looking component names scores high even if the biology is a bit off.
This favors the fine-tuned model, which is disciplined about schema.

The Opus → Jetson gap on biology-heavy topologies (pathway, biosensor) is
probably **larger in practice** than the 10-point score gap suggests.

## The takeaway

> 1. **Prompt engineering and LoRA are ~80% redundant on easy circuits.**
>    Either one gets you most of the way; stacking both is marginal.
>
> 2. **LoRA is not redundant where it matters most:** hard circuits under
>    aggressive quantization. It eliminates tail risk that prompts cannot
>    fix, because the failure modes are single-token JSON bugs.
>
> 3. **The frontier gap is structural-discipline + biology:** Opus's
>    uplift over our LoRA is concentrated on pathways (metabolic knowledge)
>    and biosensors (regulatory precision). These are plausibly closable
>    with more training data, not more model.
>
> 4. **At $1,200 one-time vs $6.80/100-call, the Jetson stack is the right
>    platform for any domain where privacy or volume matters.**

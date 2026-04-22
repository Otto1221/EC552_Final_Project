# Complex-Circuit Tail Risk — What the Average Hides

**Jetson UD-Q3_K_M + LoRA vs the same model, bare (no LoRA).**

Scores average out to a small gap (89.6 vs 87.2, Δ = +2.43). That number is **misleading**.
The real story is in the *distribution*, and it sharpens as circuits get harder.

## Per-difficulty breakdown (n=20 per row)

| Diff | LoRA mean | bare mean | Δ | LoRA min | **bare min** | LoRA <50 | **bare <50** | LoRA <70 | **bare <70** |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|---:|:---:|
| d1  | 90.75 | 90.25 | +0.50 | 85 | 83 | 0 | 0 | 0 | 0 |
| d2  | 90.80 | 86.50 | +4.30 | 82 | **0**  | 0 | **1** | 0 | **1** |
| d3  | 89.60 | 89.30 | +0.30 | 82 | 80 | 0 | 0 | 0 | 0 |
| d4  | 90.35 | 89.45 | +0.90 | 86 | 78 | 0 | 0 | 0 | 0 |
| d5  | 86.50 | 80.35 | +6.15 | 78 | **0** | 0 | **1** | 0 | **1** |

- Easy circuits (d1, d3): essentially a tie.
- **Medium-hard (d2, d5): bare catastrophically fails at least once per tier.**
- LoRA's **floor across all 100 prompts is 78**. Bare's floor is **0**.

## The two bare catastrophes

| Prompt (abbrev.) | Diff | Bare | LoRA | What went wrong (bare) |
|---|:---:|---:|---:|---|
| "Nitrate-responsive GFP via NarL/PnarK" | d2 | **0** | 94 | **Missing comma** between two JSON array entries — entire parse fails |
| "Population-level consensus via AHL quorum sensing" | d5 | **0** | 93 | Emitted a `// Note:` comment inside JSON — parser rejects |

Both bare designs were *biologically reasonable*. Bare scored zero because of **one bad token each**. The scoring pipeline is strict by design: no parse → every structural axis is zero → total is zero.

## Where LoRA actually earns its keep (d5 axis deltas)

| Axis | Max | LoRA | bare | Δ |
|---|:---:|---:|---:|---:|
| SV  Structural validity | 20 | 20.00 | 18.95 | **+1.05** |
| BW  Behavioral wiring   | 20 | 17.40 | 15.35 | **+2.05** |
| BA  Biology accuracy    | 20 | 15.75 | 15.10 | +0.65 |
| PF  Prompt fidelity     | 20 | 15.65 | 14.55 | +1.10 |
| DQ  Design quality      | 10 |  8.60 |  7.85 | +0.75 |
| REP Engineering         | 10 |  9.10 |  8.55 | +0.55 |

- **SV** gap (+1.05) = bare's parse failure knocked structural validity by ~5% on d5.
- **BW** gap (+2.05) = the *biggest* per-axis delta. On hard circuits, bare wires transcription / translation / regulation edges incorrectly more often.

## The presentation-ready framing

> The bare Gemma is mostly competent. It gets **easy circuits right** and the
> average looks acceptable. But when the prompt asks for a quorum-sensing
> consensus circuit or a nitrate-responsive inducer, it produces a
> biologically-plausible design that **silently drops a comma** or
> **inserts a `//` comment** — and the entire structured output is unusable.
>
> The LoRA closes that gap to **zero parse failures in 100 prompts** and
> raises the *minimum* score across all difficulties from 0 to 78.
>
> **Average score hides tail risk. In structured generation, tail risk is the whole problem.**

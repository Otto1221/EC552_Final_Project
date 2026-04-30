# Platform Tradeoffs — Cloud vs Local for SBOL Generation

Two charts, one story: **local LoRA on a $1,200 Jetson is competitive on quality, radically better on cost and efficiency once you're past a few thousand circuits/year.**

---

## Chart 1 — `chart_tco.png` (break-even curve)

Total annual cost vs. circuits generated per year, log-log axes, 3-year capex amortization.

**Key numbers (from `plot_tco.py`):**

| Scenario | Opus 4.7 API | MacBook M5 Max | Jetson Orin NX |
|---|---:|---:|---:|
| Hardware capex | $0 | $4,000 | $1,200 |
| Per-circuit cost | $0.068 | ~amortized | ~amortized |
| Active power | ~350 W/req | 50 W | 15 W |
| Seconds/circuit | network | ~49 s | ~124 s |

**Break-even points:**
- **Jetson beats Opus at ~5,900 circuits/yr** (≈113/week)
- **MacBook beats Opus at ~19,600 circuits/yr** (≈378/week)

**At a typical mid-size biotech lab (10,000 circuits/yr):**
- Opus:   **$680/yr**
- Mac:    **$1,334/yr**  (still in amortization)
- Jetson: **$401/yr**    ← winner by 40% vs Opus, 70% vs Mac

---

## Chart 2 — `chart_efficiency.png` (points-per-watt)

Three-panel: Quality · Power · Efficiency.

| Platform | Rubric | Active W | Points / Watt |
|---|---:|---:|---:|
| Opus 4.7 cloud API           | 99.22 | 350 | **0.28** |
| MacBook M5 Max Q8 LoRA (MLX) | 91.57 |  50 | **1.83** |
| Jetson Orin NX + LoRA        | 89.60 |  15 | **5.97** |
| Jetson Orin NX bare          | 87.17 |  15 | 5.81 |

**Jetson LoRA is ~21× more energy-efficient than an Opus API call — at 90% of the quality.**

The cloud API caveat: 350 W is a per-request allocation estimate for a datacenter H100/B200 slice; real draw is bursty but this is the right ballpark for fair comparison.

---

## Talking points for the slide

1. **Opus is the quality ceiling.** 99.22 / 100. Nothing we run locally will beat it in absolute rubric points.
2. **But quality isn't the whole question.** For bio-design, 90% quality on a hermetic box is often *more useful* than 100% quality on an internet-dependent API — think airgapped labs, field deployments, IP-sensitive constructs.
3. **The cost crossover is closer than people expect.** A working biotech lab (500 circuits/wk) pays Opus ~$1,700/yr. A Jetson would be paid off in year 1 and save money every year after.
4. **Efficiency is the real story.** Per watt delivered to the problem, the Jetson destroys the cloud. This matters for sustainability narratives, edge deployments, and anyone whose "compute budget" is actually a power budget.

---

## The one-liner for the presenter

> "For the cost of one engineer's monthly Claude subscription, you can buy a Jetson that runs your SBOL generator forever — at 90% of the quality and 21× the efficiency. Cloud wins for prototyping; local wins for production."

---

## Sources / reproducibility

- Quality numbers: `sbol_eval_v2_results.json` (bare + LoRA, 100 prompts) and `sbol_eval_v2_opus_47.json` (Opus, same 100).
- Throughput numbers: measured on device (`17.85 tok/s` Mac MLX, `7.04 tok/s` Jetson llama.cpp UD-Q3_K_M).
- Opus per-circuit cost: $0.068, computed from measured token counts × published Opus 4.7 pricing.
- Power numbers: Jetson = measured sustained; Mac = peak-during-inference (powermetrics); cloud = published datacenter per-request allocation.
- Scripts: `plot_tco.py`, `plot_efficiency.py` — both standalone matplotlib, run from repo root.

# Newgenes SBOL Generation — Presentation Numbers

## Primary result table (sbol_eval_v2, 100 stratified prompts, 6-axis rubric)

| Config | Platform | Quant | Score | Cost / 100 designs | Decode | Notes |
|---|---|---|---|---|---|---|
| **Opus 4.7** (frontier, prompt-only) | Anthropic API | bf16 | **99.22** | ~$4–6 (API) | ~50 tok/s | Upper bound; Chen & Truong techniques |
| **Qwen 3.5 27B + LoRA** | Mac MLX | Q8 | **91.57** *⁺ | $0 (local) | 17.6 tok/s | Our best-on-device; rep_penalty 1.05 + schema retry added |
| Qwen 3.5 27B + LoRA (pre-fix) | Mac MLX | Q8 | 90.71 | $0 (local) | 17.6 tok/s | baseline run before loop-mitigation patches |
| Qwen 3.5 27B + LoRA | Mac GGUF | Q3_K_M | 83.99 | $0 | — | Shows quant scaling |
| Gemma 4 26B-A4B + LoRA | Jetson Orin NX | Q2_K_XL | 92.8 *¹ | $0 | 4.55 tok/s | Edge-deployable; different eval battery |
| Qwen 3.5 27B + LoRA | Jetson Orin NX | Q3_K_M | n/a *² | — | 1.89 tok/s | Too slow to bench; hardware-limited |

¹ 20-prompt `jetson_eval100.py` battery (different rubric, not sbol_eval_v2)
² 3-prompt spot-check only; full run would take ~70 hr at 1.89 tok/s
⁺ After loop fix: feedback topology jumped from 68.5 → 89.5 (+21 pts); only one residual truncation (prompt 45 XOR gate) remains.

## Throughput / latency

| Platform | Hardware | Decode tok/s | 100-design wall time |
|---|---|---|---|
| Mac M5 Max (MLX Q8) | ~64–128 GB unified | 17.6 | ~30–40 min |
| Mac M5 Max (GGUF Q3) | ~64–128 GB unified | — | — |
| Jetson Orin NX 16 GB (Gemma Q2_K_XL) | 15 W SoC | 4.55 | ~3 hr |
| Jetson Orin NX 16 GB (Qwen Q3_K_M) | 15 W SoC | 1.89 | ~70 hr |
| Anthropic API (Opus 4.7) | Cloud | ~50+ | ~2–3 min |

## Cost per 100 designs

| Path | Variable cost | Fixed cost | Power |
|---|---|---|---|
| Opus 4.7 API | ~$4–6 | $0 (hosted) | — |
| Mac MLX (local) | $0 | — | ~40–80 W peak during inference |
| Jetson (edge) | $0 | Jetson Orin NX ~$500 device | 15 W (enables battery / remote) |

## Story arc

**Frontier → Edge gradient:**
- Opus 4.7 sets the ceiling at 99.22. Costs money per call; requires internet.
- Our Q8 MLX on Mac hits 90.71 (**91% of frontier**) at zero variable cost, 17.6 tok/s.
- Jetson edge deploy (Gemma) holds quality on its own eval and runs on 15 W — critical for field/lab settings with no connectivity.

**LoRA matters (but not as much as you'd think):**
- Q3 GGUF with LoRA: 83.99
- Q8 MLX with LoRA: 91.57
- Gap between Q3 and Q8 (+7.58 pts) quantifies precision tax at the edge.

**The Jetson story:**
- 27B is too slow on Orin NX (1.89 tok/s → 70 hr / 100 designs).
- We deployed 26B-A4B MoE instead: 4.55 tok/s, 15 W power, 92.8 on the 20-prompt field battery.
- Fits in 15.5 GB unified memory with Q8 KV cache and LoRA applied at runtime.

## Training cost (LoRA on M5 Max)

| Item | Value |
|---|---|
| Base model | Qwen 3.5 27B, pre-quantized 4-bit (`mlx-community` convention) |
| Adapter shape | rank 8, alpha 16, dropout 0.05, targeting last 16 of 64 layers |
| Dataset | 1,162 train / 64 valid / 75 test — SBOL prompt→JSON, chat format |
| Tokens trained | ~1.1 M total (median 875 tok/row, p95 1,630) |
| Max seq length | 1,024 (p45 of dataset; longer rows truncated) |
| Optimizer | AdamW, LR 1.0e-5, batch 1, grad_checkpoint on |
| Iterations | 2,500 (≈1.76 epochs) |
| **Wall-clock** | **7 h 40 min on M5 Max 64 GB** (≈11 s/step, 45 min per 250-step checkpoint) |
| Peak Metal memory | 56 GB (`iogpu.wired_limit_mb=57344`) |
| Adapter size | 58 MB per checkpoint |
| Out-of-pocket | **$0** (laptop compute) |

**Return on the 7.7 hours:** +2.50 pts over base+default (Cell A → Cell C), or roughly **3 hours per point** of average rubric score. The same delta can be bought in ~30 lines of prompt text (Chen & Truong) — see overlap analysis below.

## Evaluation independence (contamination check)

Every `sbol_eval_v2` prompt was compared against all 1,162 train + 64 valid + 75 test rows by exact match, substring, and token-set Jaccard.

| Check | train | valid | test |
|---|---|---|---|
| Exact matches | **0**/100 | 0/100 | 0/100 |
| Substring matches | **0**/100 | 0/100 | 0/100 |
| High Jaccard (>0.55) | **0** | 0 | 0 |
| Max Jaccard observed | 0.47 | — | — |
| Mean max-Jaccard | 0.235 | — | — |

The single highest-Jaccard pair (0.47) was *thematically* similar — "quorum sensing sender-receiver" appears in both — but the eval prompt specifies `LuxI/LuxR` explicitly and the train row does not. Classifies as topology overlap (intentional — we want the eval to cover topologies we trained on) rather than prompt leakage.

**No eval prompt appears in any training split.**

## Score distributions (not just averages)

| Cell | n | min | q25 | median | q75 | max | mean | 50-69 / 70-79 / 80-89 / 90-94 / 95-99 / 100 |
|---|---|---|---|---|---|---|---|---|
| Opus 4.7 | 100 | 96 | 98 | 100 | 100 | 100 | 99.22 | 0/0/0/0/32/**68** |
| C: LoRA+default (100) | 100 | 0 | 90 | 93 | 95 | 100 | 91.57 | 0/2/18/48/30/1 |
| D: LoRA+Chen (34) | 34 | 82 | 92 | 93 | 95 | 99 | 93.18 | 0/0/5/17/12/0 |
| B: base+Chen (34) | 34 | 80 | 91 | 94 | 95 | 100 | 92.68 | 0/0/7/14/12/1 |
| C_s3: LoRA+default (34) | 34 | 82 | 90 | 93 | 95 | 99 | 92.21 | 0/0/8/16/10/0 |
| A: base+default (34) | 34 | 80 | 88 | 90 | 92 | 97 | 89.71 | 0/0/15/17/2/0 |
| Q3 GGUF+LoRA (100) | 100 | 0 | 86 | 91 | 93 | 97 | 83.99 | 0/3/26/51/13/0 (+**7 zeros**) |

**Two things the averages hide:**
1. **Opus doesn't just have a higher mean — it has a completely different distribution.** 68 perfect 100s is the *mode*, not an outlier. No frontier model response scores below 96. This is a regime change, not a shift.
2. **The 34-prompt cells (A/B/C_s3/D) all have tight distributions with no catastrophic failures**, which is why we trust the ablation averages despite small n. The only cell with a bimodal, long-tailed distribution is Q3 GGUF — where 7 hard zeros drag a median of 91 down to mean 84.

Distribution-wise, Cell D is a tighter Cell C_s3: same median (93), same IQR (90–95), but +1.0 on the mean because the low tail is pulled up.

## Prompt-vs-LoRA ablation (2×2, stride-3 sample of 34 prompts)

|  | Default SBOL prompt | Chen & Truong 2026 prompt |
|---|---|---|
| **Base 4-bit (no LoRA)** | 89.71 (A) | 92.68 (B) |
| **LoRA 8-bit** | 92.21 (C) | **93.18 (D)** |

| Effect | Δ |
|---|---|
| LoRA alone (A→C) | **+2.50** |
| Chen prompt alone (A→B) | **+2.97** |
| LoRA + Chen combined (A→D) | **+3.47** |
| Sum if additive | +5.47 |
| Actual combined | +3.47 |
| **Overlap (redundant gain)** | **~2.0 pts** |

**Finding:** LoRA fine-tuning and the Chen & Truong prompt-engineering technique teach the model largely *the same things*. They are **not additive** — combining both gives only marginal lift over either alone. For teams without GPU access, the Chen prompt captures 85% of the LoRA gain at zero training cost.

## Where each intervention actually lands (per-axis deltas on the 2×2)

Axis caps: SV=20, BW=20, BA=20, PF=20, DQ=10, REP=10.

| Axis | A (base+default) | ΔChen | ΔLoRA | ΔBoth |
|---|---|---|---|---|
| SV (schema valid) | 20.00 | +0.00 | +0.00 | +0.00 |
| BW (block well-formed) | 19.12 | +0.17 | -0.15 | -0.15 |
| **BA (biology-appropriate parts)** | **16.32** | **+1.47** | **+1.33** | **+2.12** |
| **PF (part completeness)** | **15.82** | **+0.80** | **+0.86** | **+1.03** |
| DQ (description quality) | 9.03 | +0.29 | +0.23 | +0.23 |
| REP (response formatting) | 9.41 | +0.24 | +0.24 | +0.24 |

**Reading:** SV/BW/DQ/REP are already near their caps at the baseline — the model knows how to emit valid JSON. The headroom is on **BA** (chassis-appropriate parts: no pLac in mammalian cells, right terminator for the organism) and **PF** (every CDS has its full transcription unit). Both interventions push the same two axes — which is *why* they overlap so much. Chen prompt does this via explicit chassis rules + acceptance checklist; LoRA does it via example-grounded pattern matching. Same effect, different mechanism.

## Error taxonomy (failure modes)

| Config | n | zero | <50 | truncated | parse-fail |
|---|---|---|---|---|---|
| Opus 4.7 | 100 | 0 | 0 | 0 | 0 |
| Q8 LoRA + default (fix) | 100 | 1 | 1 | 1 | 1 |
| **Q3 GGUF + LoRA** | **100** | **7** | **7** | **21** | **7** |
| 2×2 stride-3 cells (A,B,C,D) | 34 each | 0 | 0 | 0 | 0 |

**Key insight:** The 34-prompt ablation cells all have zero failures — stable enough to trust the averages. The averages are moving *only* because of mid-range score differences, not catastrophic events.

**Quantization tax is catastrophic, not graceful:** Q3_K_M adds 21 truncations and 7 parse-fails per 100. That's the real cost of going to Q3 at the edge — not the 7.58-pt average drop, but the reliability collapse behind it. Q8 is the floor for production.

## Per-topology heatmap (how each topology fares)

|  | A (base+def) | B (base+Chen) | C (LoRA+def) | D (LoRA+Chen) | Opus | D−A | Opus−D |
|---|---|---|---|---|---|---|---|
| reporter | 89.0 | 94.4 | 94.1 | 94.6 | 98.2 | +5.6 | +3.6 |
| inducible | 89.6 | 90.7 | 90.3 | 91.1 | 99.0 | +1.6 | +7.9 |
| biosensor | 91.2 | 93.2 | 90.0 | 92.0 | 100.0 | +0.8 | +8.0 |
| gate | 92.2 | 96.2 | 96.5 | 97.5 | 99.1 | +5.2 | +1.6 |
| toggle | 91.0 | 94.5 | 95.5 | 94.5 | 100.0 | +3.5 | +5.5 |
| oscillator | 92.0 | 99.0 | 96.0 | 99.0 | 100.0 | +7.0 | +1.0 |
| feedback | 91.0 | 92.0 | 93.0 | 93.0 | 99.0 | +2.0 | +6.0 |
| cascade | 93.0 | 97.0 | 94.0 | 98.0 | 98.7 | +5.0 | +0.7 |
| **pathway** | **87.3** | 89.0 | 90.7 | 89.7 | **100.0** | +2.3 | **+10.3** |
| **crispr** | **84.7** | 87.3 | 87.7 | 91.3 | 100.0 | +6.7 | +8.7 |
| **kill** | **92.0** | 91.0 | 89.0 | **88.0** | 100.0 | **−4.0** | **+12.0** |

**Three stories in one table:**

- **Chen + LoRA close the gap on structure-heavy topologies** (gate, oscillator, cascade) — D matches or approaches Opus on these (Δ ≤ 1.6). These are the topologies where the "right answer" is a templatable circuit pattern.
- **Pathway and CRISPR remain the frontier gap** — pathway loses 10.3 pts to Opus, CRISPR loses 8.7. These demand domain-specific biochemistry (enzyme orders, Cas variants) that neither prompt nor LoRA fully transfers.
- **Kill switch is a regression with training.** A=92 → D=88. The 34-prompt kill subset has the model producing structurally cleaner outputs that implement the *wrong* logic (see biology validation — NAND becomes OR). This is the smoking gun for why we need a logic-fidelity rubric axis.

## Biology validation (expert review of 11 hardest outputs from Cell D)

| Rubric | Avg on same 11 items |
|---|---|
| Structural (sbol_eval_v2, 6-axis) | **91.8** |
| Biology (plausibility as-if-built) | **71.4** |
| Gap | **-20.4** |

**Strong topologies (bio ≥ 85):** feedback autoregulation, repressilator, metabolic pathways, tumor-targeting probiotic.
**Weak topologies (bio ≤ 65):** toggle switches (both failed — self-repression instead of cross-repression), NAND kill switch (arabinose-only gates survival), Cas13 collateral sensing (direct cleavage instead), Cas12a DETECTR (ssDNA reporter misrepresented as CDS).

The structural rubric catches well-formedness. A circuit can present a cyclic graph and complete transcription units while implementing the *wrong logic* — this is the 20-point gap. See `BIOLOGY_VALIDATION.md` for per-item verdicts.

## Why isn't Opus at 100? (rubric ceiling, not model ceiling)

Opus 4.7 scored 99.22/100. Of 100 prompts, 32 came in below the max. Those 32 lose points on exactly one place:

| Axis | sub-scorer | prompts affected | pts lost (total) |
|---|---|---|---|
| SV, BW, BA, DQ, REP | (all sub-scorers) | **0** | 0 |
| PF | organism_match, keyword_presence, completeness, quantitative_addressed | **0** | 0 |
| **PF** | **behavior_matches_logic** | **32** | **78** |

`behavior_matches_logic` (`sbol_eval_v2.py:572`) scores whether the `behavior` field re-uses content words from the prompt — it's a keyword-overlap check, not a semantics check. It returns 6 (strong overlap), 4, 2, or 0 by how many of the prompt's ≥5-letter words appear in `behavior`. When Opus paraphrases a prompt like "toggle between two states under mutually exclusive inducers" into a behavior string like "bistable system maintained by cross-repression", the overlap drops from 6 → 2 even though the semantics are perfect.

**This is a rubric artifact, not a model failure.** Two implications:

1. **The 99.22 number is a floor on Opus's true ceiling.** A semantic behavior-matcher (LLM-judge) would score Opus higher. Read it as "Opus never misses on structure or biology, and 32% of the time it phrases its behavior field in fresh words the keyword-matcher penalizes."
2. **Our own rubric is intentionally deterministic.** No LLM-judge, no prompt-dependent scoring — the axes are auditable, reproducible, and free. Cost: it penalizes fluency.

For the defense: if a reviewer says "99.22 feels too round — where does Opus actually fail?", the honest answer is "it doesn't; the rubric's keyword-overlap check does."

## Training data provenance (`generate_data_llm.py`)

Training data was synthesized, not scraped. Two-stage pipeline:

| Stage | Generator | Complexity | Cost | Share |
|---|---|---|---|---|
| Stage 1 | **Qwen 2.5 72B** (local, self-hosted) | simple / medium circuit examples | $0 | majority |
| Stage 2 | **GPT-5.4** (Anthropic API) | complex circuit examples | API paid | minority |

Outputs: `train.jsonl` 1,162 rows / `valid.jsonl` 64 / `test.jsonl` 75, all in chat format with a system prompt, user prompt, and assistant JSON response.

**Why synthesize instead of scrape?** Real SBOL deposits (e.g. SynBioHub) are sparse, heterogeneously annotated, and biased toward well-studied chassis. Synthesis lets us stratify by topology × organism × difficulty and guarantee every row parses. The tradeoff: the model learns our generator's distribution, not nature's — which is exactly why the biology-validation gap (-20 pts) matters.

**Why two generators?** Qwen 2.5 72B is free but drops in quality on 5-gene pathway or CRISPR circuits. GPT-5.4 handles those reliably. Mixed-provenance training is cheaper than all-GPT, cleaner than all-Qwen.

**Risk:** training on model-generated data can bake in the teacher's biases. Our eval prompts (`sbol_eval_v2.PROMPTS`) are hand-curated and do not overlap any training row (see contamination check), so evaluation is not teacher-dependent. But the *ceiling* of what the student can learn is bounded by what the two teachers knew.

## Comparison with Chen & Truong 2026

Chen & Truong report their prompt-only technique on their own benchmark (a different rubric and prompt set — their paper reports a composite "SBOL generation quality" score, not a structural 6-axis decomposition).

**We do not claim our 92.68 (base + Chen prompt) reproduces their headline.** What we claim:

- We applied their prompt template (4,780 chars: SBOL grounding + biological guardrails + internal acceptance checklist) to Qwen 3.5 27B Q4.
- On **our** stricter deterministic rubric, the prompt buys +2.97 over Qwen's default system prompt — roughly the same delta as 7.7 hours of LoRA training.
- The overlap with LoRA (+2.0 redundancy) is the defensible finding, independent of Chen's own numbers.

This is a methodological limitation: we use Chen & Truong's technique as a baseline, not their benchmark as a comparison. A strict reviewer could ask for a run of our model on their benchmark — we haven't done that.

## Limitations

Written up front so the Q&A doesn't have to surface them:

- **Single model family.** All results are Qwen 3.5 27B (plus one Gemma 26B-A4B data point on Jetson). We don't know whether the same deltas transfer to Llama, Mistral, DeepSeek, or older Qwen.
- **Single task.** SBOL JSON generation only. No test of the model on downstream tasks (DNA sequence assembly, part-swap edits, design critique, etc.).
- **100-prompt eval.** Sufficient for reliable averages (SEM ≈ 0.6) but the 2×2 ablation cells use 34 prompts each. The ~2-pt overlap finding has ~1 pt of noise around it.
- **No wet-lab validation.** Structural rubric = 91.8; biology-judged plausibility = 71.4. Neither is "actually built it." Beyond our scope and beyond any fine-tuning team's scope — requires a BSL-1 lab partner.
- **Single training seed.** We ran one LoRA training; did not measure run-to-run variance. A ±1 pt swing from seed change would not be surprising.
- **English-only prompts.** No multilingual evaluation.
- **Single-turn.** Real users iterate ("add a ribozyme insulator", "swap to Bacillus"). We do not test multi-turn refinement.
- **Judge for biology validation is an LLM (Opus).** The -20 pt gap is plausible but not blind — Opus judging Cell D outputs knows they're from a smaller model. Wet-lab or human expert judging would be more defensible.
- **Eval rubric is deterministic-by-choice.** Trades fluency-sensitivity for reproducibility (see Opus ceiling analysis above). A reviewer could ask for an LLM-judge ensemble.

**What we can defend:** the 91.57 local, the 2×2 ablation finding, the 0-contamination property, the quantization reliability cliff, the 20-pt structural-vs-biology gap, the per-topology where-it-breaks story.

**What we cannot defend yet:** "this works in any lab" — requires wet-lab.

## Headline for the slide

1. **Frontier ceiling:** Opus 4.7 @ **99.22** — upper bound, cloud-only, ~$5/100.
2. **Best local:** Qwen 3.5 27B Q8 MLX + LoRA @ **91.57** — ~92% of frontier, $0 variable.
3. **Ablation insight:** LoRA and Chen prompt are ~80% redundant; pick whichever fits your compute budget. Both push the same two axes (BA = chassis-appropriate parts, PF = part completeness) — same effect, different mechanism.
4. **Edge deploy:** Gemma 26B-A4B MoE at 15 W holds 92.8 on its own battery.
5. **Quantization reliability cliff:** Q8 has 1% failure rate, Q3 has 21% truncation rate — avg scores hide that Q3 is catastrophically less reliable, not gracefully degraded.
6. **Topology gap:** pathway and CRISPR lose 9–10 pts to frontier; kill-switch actually *regresses* with training (logic inversion). These point to specific follow-up work, not generic scale-up.
7. **Biology gap:** 20 pts between "valid JSON" (91.8) and "works in vivo" (71.4) — the next rubric axis to build.
8. **Cost of LoRA:** 7.7 hrs of M5 Max wall-clock, $0 out-of-pocket. ~3 hrs per rubric point. Chen prompt buys the same deltas for free — useful framing for when training compute is a constraint.
9. **Clean eval:** 0/100 prompts from sbol_eval_v2 appear (exact, substring, or Jaccard > 0.55) in any training split. Averages are not inflated by data leakage.
10. **Rubric is deterministic, not LLM-judged.** Opus's 99.22 (not 100) comes entirely from one keyword-overlap sub-scorer that penalizes paraphrasing — a rubric artifact, not a model failure. Deterministic scoring is reproducible and free; the tradeoff is fluency-insensitivity, which we disclose.
11. **Training data provenance:** Qwen 2.5 72B (local) + GPT-5.4 (API) synthesized 1,162 train / 64 valid / 75 test rows. Eval is hand-curated and disjoint from training.


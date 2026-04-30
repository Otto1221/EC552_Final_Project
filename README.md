# Newgenes — Fine-Tuning SBOL Generators on Constrained Hardware

Fine-tuning open LLMs to generate synthetic biology circuit designs (SBOL JSON) from natural-language prompts, benchmarked against a frontier cloud model on a deterministic 6-axis rubric.

**Question:** Can a $1,200 edge device running a fine-tuned 4B-active-param model produce SBOL designs competitive with Claude Opus 4.7?

**Answer:** Yes — **89.6 / 100** on the Jetson vs **99.2 / 100** on Opus, with a break-even against cloud API cost at ~5,900 circuits/year and ~21× the points-per-watt.

---

## Headline results

| Setup | Platform | Score (n=100) | $/100 circuits | Throughput |
|---|---|---:|---:|---:|
| **Opus 4.7 API** (ceiling) | Anthropic cloud | **99.22** | ~$6.80 | ~50 tok/s |
| Qwen 3.5 27B + LoRA, Q8 | Mac M5 Max (MLX) | 92.2 | $0 var. | 17.6 tok/s |
| Gemma 4 26B-A4B + LoRA, UD-Q3_K_M | **Jetson Orin NX (15 W)** | **89.60** | $0 var. | 7.0 tok/s |
| Gemma 4 26B-A4B bare, UD-Q3_K_M | Jetson Orin NX | 87.17 | $0 var. | 7.1 tok/s |

**Three things worth knowing beyond the averages:**

1. **LoRA and a frontier-style prompt are ~80% redundant on easy circuits.** A 2×2 ablation (base/LoRA × default/Chen-Truong prompt) shows +2.97 from prompt alone, +2.50 from LoRA alone, +3.47 combined. Pick whichever fits your compute budget.
2. **LoRA's real value is tail risk.** Bare Jetson has 2/100 catastrophic zeros (single-token JSON bugs); LoRA has 0/100 with a floor of 78. Average hides this.
3. **Break-even math favors local once you're past ~5,900 circuits/yr** — below that, use the API.

Full detail: [`REPORT.md`](./REPORT.md).

---

## Repo layout

```
.
├── README.md              ← you are here
├── REPORT.md              ← master writeup (benchmarks, methodology, limitations)
├── .gitignore
│
├── Code/assets/                ← charts and circuit renders used in the writeup (PNG)
├── Code/docs/                  ← BIOLOGY_VALIDATION.md, PRESENTATION_TABLE.md
├── Code/slides/                ← 7 presentation-ready slide_*.md scripts
├── Code/configs/               ← LoRA training configs (lora_config_*.yaml)
├── Code/data/                  ← train.jsonl, valid.jsonl, test.jsonl, demo_prompts.json, exemplar_bank.json
├── Code/results/               ← all eval results: sbol_eval_v2_*.json + .summary.json, biology_review_*.json
└── Code/src/                   ← all Python scripts (39 files)
    ├── sbol_eval_v2.py               ← deterministic 6-axis rubric scorer (the core artifact)
    ├── train.py                      ← MLX QLoRA training entry (bf16 patch + mlx_lm.lora)
    ├── infer.py, prepare_mlx.py      ← inference + MLX data prep
    ├── plot_tco.py, plot_efficiency.py      ← cost / efficiency charts
    ├── render_sbol_circuit.py        ← renders model JSON as a directed circuit graph
    ├── run_demo_prompts.py           ← runs the 10 novel demo prompts against a live llama-server
    ├── scrape_circuits.py            ← pulls real SBOL from SynBioHub / iGEM for training
    ├── chen_truong_system_prompt.py  ← reference implementation of Chen & Truong 2026 prompt
    ├── analyze_ablation.py           ← 2×2 ablation analysis over ../results/
    ├── check_contamination.py        ← eval-vs-train overlap audit
    ├── jetson_*.py, deploy_jetson.py ← edge deployment + Jetson eval harness
    ├── generate_*.py                 ← synthetic training data generators
    └── <20 other helper scripts>

Scripts in `Code/src/` read from `../Code/data/` and write to `../results/` (or `../assets/` for charts).
```

**Not in the repo** (too large or licensed elsewhere):
- Base model weights — download from HuggingFace (`Qwen/Qwen3.5-27B-Instruct`, `google/gemma-4-26b-a4b-it`).
- Quantized GGUFs — rebuilt via `llama.cpp` convert scripts; see `REPORT.md` §8.
- Trained LoRA adapter — ~384 MB, available on request.
- Opus API keys — set `ANTHROPIC_API_KEY` in your environment.

---

## Reproducing the results

The evaluation framework runs without model weights — point it at any OpenAI-compatible endpoint.

Run every command from the repo root. The scripts resolve `../Code/data/` and `../results/` relative to their own location in `Code/src/`.

### Score an existing response set
```bash
python3 Code/src/sbol_eval_v2.py --input Code/results/sbol_eval_v2_gemma_udq3km_lora.json --summary
```

### Run the 100-prompt eval against a local llama-server
```bash
# Start llama-server with your model + optional LoRA adapter
./llama-server --model <model.gguf> --lora <adapter.gguf> --port 8080

# Run eval (writes Code/results/sbol_eval_v2_<tag>.json + .summary.json)
LLAMA_URL=http://localhost:8080/v1/chat/completions python3 Code/src/jetson_sbol_eval_v2_http.py <tag>
```

### Run the 2×2 ablation (base/LoRA × default/Chen-prompt)
```bash
# Cells A, B, C, D on stride-3 subset (n=34 each)
SAMPLE_EVERY=3 python3 Code/src/jetson_sbol_eval_v2_http.py cell_a_base_default_s3
SAMPLE_EVERY=3 CHEN_PROMPT=1 python3 Code/src/jetson_sbol_eval_v2_http.py cell_b_base_chen_s3
python3 Code/src/extract_stride_subset.py mac_mlx_q8_lora_fix 3   # cell C from full run
SAMPLE_EVERY=3 CHEN_PROMPT=1 python3 Code/src/jetson_sbol_eval_v2_http.py cell_d_lora_chen_s3
python3 Code/src/analyze_ablation.py
```

### Generate the charts
```bash
python3 Code/src/plot_tco.py          # writes Code/assets/chart_tco.png
python3 Code/src/plot_efficiency.py   # writes Code/assets/chart_efficiency.png
```

### Train a LoRA adapter (MLX, M-series Mac, ~7.5 h)
```bash
mlx_lm.lora --config Code/configs/lora_config_qwen35_27b.yaml
# outputs adapters/<run-name>/adapter.safetensors
```

---

## Methodology, in one paragraph

Three training-data tiers (Qwen-synthesized simple, GPT-synthesized complex, SynBioHub-scraped real) combined into 1,162 prompt→JSON pairs, chat-formatted. QLoRA (rank 8, α 16, last 16 of 64 layers) trained on a 64 GB M5 Max under MLX for 2,500 iterations (≈1.76 epochs). Deployed as `.safetensors` on Mac or converted to GGUF LoRA for Jetson runtime. Evaluated on a hand-curated 100-prompt set (zero substring overlap with training; max Jaccard 0.47), scored by a deterministic 6-axis rubric (Structural Validity 20, Behavioral Wiring 20, Biology Appropriate 20, Part Fidelity 20, Description Quality 10, Response Format 10 — total 100). Full spec + provenance + contamination check: [`REPORT.md`](./REPORT.md) §3.

---

## Limitations (honest)

- **One model family** evaluated end-to-end (Qwen 3.5 27B), plus one edge datapoint (Gemma 4 26B-A4B). No transfer to Llama / Mistral / DeepSeek.
- **One task** — SBOL JSON generation only. No downstream assembly, part-swapping, or wet-lab validation.
- **Structural rubric, not biological.** 80 / 100 points are schema + wiring; only 20 points touch biology. A separate Opus-as-judge biology review scored the LoRA 71.4 / 100 — a 20-point gap that a purely structural rubric cannot see. See [`Code/docs/BIOLOGY_VALIDATION.md`](./Code/docs/BIOLOGY_VALIDATION.md).
- **Single training seed**, ±1 pt run-to-run variance plausible.
- **No in-vivo validation.** Rubric measures "is this a plausible circuit specification," not "does it work in E. coli."

---

## Citation / credits

- Chen & Truong 2026, *"Prompting LLMs for Synthetic Biology Design"* — baseline prompt (see `Code/src/chen_truong_system_prompt.py`).
- Qwen 3.5, Google Gemma 4 — base models.
- SynBioHub, iGEM Parts Registry, DataCurationProject — scraped training data.
- MLX, llama.cpp, Unsloth UD K-quants — training and deployment infrastructure.

Class project, spring 2026. See `REPORT.md` for the full writeup.

# NewGenes — The Whole Project, Explained for Beginners

A walkthrough that assumes no background in machine learning, synthetic biology, or edge computing. By the end, you should understand what we built, why, what we found, and how to reproduce it on your own machine.

---

## 1. What is this project, in one paragraph?

We taught small open-source AI models to design genetic circuits — the synthetic-biology equivalent of an electrical schematic — from plain-English descriptions. Then we showed that a $1,200 NVIDIA Jetson edge device, drawing 15 watts (less than a lightbulb), can produce designs **89.6 % as good** as Claude Opus 4.7 (the frontier model), at **0 cost per circuit** and with **21× the score-per-watt efficiency**.

The point isn't to beat Opus. It's to show that a small fine-tuned open model on cheap hardware is "good enough" for most labs — and lets you keep your IP off third-party servers.

---

## 2. What is a "genetic circuit," and why does an LLM care?

In synthetic biology, a **genetic circuit** is a designed piece of DNA that does something — express a protein when sugar appears, kill a cell when the wrong gene activates, oscillate like a clock. It's built from standard parts:

- **Promoter** — the "on switch" that initiates transcription
- **RBS** (ribosome binding site) — controls how often the gene gets translated
- **CDS** (coding sequence) — the gene itself
- **Terminator** — the "stop sign"

A circuit is a JSON object listing these parts and how they wire together. Example: *"Express GFP from a constitutive promoter in E. coli"* should produce a JSON spec with one promoter, one RBS, one CDS (the GFP gene), one terminator, and an interaction list saying *promoter→cds (transcription)*, *rbs→cds (translation)*.

This task is annoying for biologists (lots of bookkeeping) and a great fit for LLMs (structured output from natural language). The standard format we target is called **SBOL** — Synthetic Biology Open Language.

---

## 3. Quick glossary

| Term | What it means |
|---|---|
| **LLM** | Large Language Model — like ChatGPT, but ours is open-source so it runs on your hardware |
| **Fine-tuning** | Taking a pre-trained LLM and teaching it your specific task with a small dataset |
| **LoRA** | Low-Rank Adaptation. A cheap way to fine-tune by training only small "adapter" matrices instead of all 27 billion weights. Output is a ~400 MB file that snaps onto the original model |
| **Quantization** | Squashing the model's numbers from 16-bit to 4 or 3 bits so it fits in less RAM. Trades a little accuracy for fitting on small devices |
| **MLX** | Apple's ML framework that uses M-series GPUs efficiently |
| **GGUF** | A quantized model file format used by `llama.cpp` (the runtime we use on Jetson) |
| **Jetson Orin NX** | A small NVIDIA edge computer the size of a deck of cards. 16 GB RAM, 15 W power draw, $1,200 |
| **Rubric** | Our scoring system — 100 points across 6 categories, calculated by code (not human judges) |
| **Ablation** | An experiment that turns features off one at a time to see which ones actually matter |

---

## 4. The big question we asked

> Can a fine-tuned 4-billion-parameter model on a $1,200 edge device produce circuits competitive with a frontier cloud model that costs ~$0.07 per circuit?

**Answer: Yes.** Headline numbers:

| Setup | Hardware | Score (out of 100) | Cost per 100 circuits |
|---|---|---|---|
| Claude Opus 4.7 | Anthropic cloud | **99.22** | ~$6.80 |
| Qwen 3.5 27B + our LoRA | MacBook M5 Max | 92.2 | $0 |
| Gemma 4 26B-A4B + our LoRA | **Jetson Orin NX (15 W)** | **89.60** | $0 |
| Gemma 4 26B-A4B (no LoRA) | Jetson Orin NX | 87.17 | $0 |

You give up 9.6 points to save $6.80 per 100 circuits, run offline, and keep your data private. For most labs, that's a great trade.

---

## 5. The three things we discovered (beyond the averages)

Averages hide interesting structure. Three findings worth knowing:

### 5.1 LoRA and a "frontier-style prompt" are ~80% redundant
We ran a 2×2 ablation on the Jetson Orin NX (Gemma 4 26B-A4B, UD-Q3_K_M):

| | Default short prompt | Long Chen & Truong prompt |
|---|---|---|
| **Base model** | 86.94 (Cell A) | 89.91 (Cell B, +2.97) |
| **+ our LoRA** | 89.44 (Cell C, +2.50) | 90.41 (Cell D, +3.47) |

Solo gains: **+2.97** from prompt alone, **+2.50** from LoRA alone. Combined: only **+3.47** — much less than the **+5.47** you'd expect if they stacked. They're teaching the model the *same* things via different channels. Pick whichever fits your compute budget.

### 5.2 LoRA's real value is tail risk, not average
- **Bare model** on Jetson: 2 catastrophic failures out of 100 (single-token JSON bugs that score 0)
- **Same model with LoRA**: 0 failures, score floor of 78

The average hides this. **LoRA isn't making good outputs better — it's preventing bad outputs entirely.** That matters for production.

### 5.3 Break-even math
- For < 5,900 circuits/year → use the Opus API (you'll never recoup the $1,200 hardware cost)
- For > 5,900 circuits/year → buy a Jetson and run locally
- A 10-scientist lab generating 100k circuits/year saves ~$5,500/year vs cloud

---

## 6. How does the scoring rubric work?

Six axes, totaling 100 points, calculated by **code** (no human judges, no other LLM as judge — fully reproducible):

| Axis | Pts | What it checks |
|---|---|---|
| Structural Validity | 20 | Valid JSON, required fields, references to existing parts |
| Behavioral Wiring | 20 | Does the interaction graph match the prompt? (e.g., if prompt says "feedback loop," is there a cycle?) |
| Biology Appropriate | 20 | Right organism? Sensible parts? Promoter type matches host? |
| Part Fidelity | 20 | Are the named parts plausible? (no "GFP terminator" labeled as a CDS) |
| Description Quality | 10 | Does the JSON `behavior` field actually describe what the prompt asked for? |
| Response Format | 10 | Clean JSON output, no markdown fences, no extra prose |

A circuit scoring 95+ is publication-quality. 80–95 is "needs minor cleanup." Below 70 means real semantic errors.

**Honest caveat:** 80 of the 100 points are structural (schema + wiring). Only 20 touch *biology*. We ran a separate Opus-as-judge biology review on our LoRA outputs and scored 71.4 / 100 — meaning a structurally perfect circuit isn't always biologically sensible. See `docs/BIOLOGY_VALIDATION.md`.

---

## 7. Replicating the project

Three difficulty tiers depending on how deep you want to go.

### Tier 1: Just run the evaluation (15 minutes, no GPU needed)

You can score any LLM's outputs against our rubric. You don't need to train anything.

```bash
git clone https://github.com/Otto1221/EC552_Final_Project.git
cd EC552_Final_Project
pip install -r requirements.txt   # or: pip install matplotlib networkx numpy scipy

# Look at existing benchmark results
python3 src/score_distributions.py

# Inspect a single response file
python3 -c "import json; r=json.load(open('results/sbol_eval_v2_gemma_udq3km_lora.json')); print(r[0]['response'][:500])"
```

You'll see the score distribution table — exactly the numbers from our slides. The data and scoring code are public; nothing is hidden.

### Tier 2: Run the eval against your own LLM (1–4 hours)

You need an OpenAI-compatible endpoint. Easiest options:
- **OpenAI / Anthropic API** (paid, fast)
- **Ollama** (free, runs locally, supports many open models)
- **`llama.cpp` server** (free, runs anywhere, supports GGUF)
- **MLX server** on a Mac (free, fast on M-series)

Start your endpoint on `localhost:8080`, then:

```bash
# This sends all 100 evaluation prompts and writes results/sbol_eval_v2_<tag>.json
LLAMA_URL=http://localhost:8080/v1/chat/completions \
    python3 src/jetson_sbol_eval_v2_http.py mymodel_v1
```

Time depends on your model speed. On a MacBook with a 27B Q8 model: ~50s × 100 prompts = ~80 min. On a Jetson: ~2.5 hours. On the OpenAI API: ~10 min.

The script auto-resumes if you Ctrl+C and re-run.

### Tier 3: Train your own LoRA from scratch (~7.5 hours on M5 Max)

Requires:
- 64+ GB RAM Mac with an M-series GPU (M2 Max or better)
- Or any GPU with 24+ GB VRAM (Linux/Windows with PyTorch)
- The base model weights (download from HuggingFace, ~50 GB)
- Patience

```bash
# 1. Download the base model from HuggingFace
huggingface-cli download Qwen/Qwen3.5-27B-Instruct --local-dir ./Qwen3.5-27B

# 2. Convert to MLX format
mlx_lm.convert --hf-path ./Qwen3.5-27B --mlx-path ./Qwen3.5-27B-mlx -q  # -q = quantize to 4-bit

# 3. Train the LoRA (uses our config + our 1,162-row dataset)
mlx_lm.lora --config configs/lora_config_qwen35_27b.yaml

# 4. Run inference with your fresh LoRA
mlx_lm.generate \
    --model ./Qwen3.5-27B-mlx \
    --adapter-path ./adapters/qwen35-27b-newgenes \
    --prompt "Express GFP from a constitutive promoter in E. coli" \
    --temp 0.1 --max-tokens 1500
```

The training data is in `data/train.jsonl` (1,162 rows). You can edit it, add your own examples, retrain.

### Tier 4: Deploy to a Jetson edge device

This is the part that's specific to our hardware. If you have a Jetson Orin NX:

```bash
# 1. Convert your fused model to GGUF format using llama.cpp
python3 llama.cpp/convert_hf_to_gguf.py ./Qwen3.5-27B-merged --outtype f16

# 2. Quantize to UD-Q3_K_M (Unsloth's improved 3-bit format — fits in 16 GB RAM)
./llama.cpp/build/bin/llama-quantize Qwen3.5-27B.gguf Qwen3.5-27B-Q3_K_M.gguf Q3_K_M

# 3. SCP to Jetson
scp Qwen3.5-27B-Q3_K_M.gguf visionx@192.168.55.1:~/models/

# 4. On Jetson, start llama-server
./llama.cpp/build/bin/llama-server \
    --model ~/models/Qwen3.5-27B-Q3_K_M.gguf \
    --port 8080 -ngl 99
```

**Warning:** UD-Q3 has a ~21% truncation rate at the standard 1,800-token limit. Use UD-Q4 or higher for production unless you've stress-tested truncation.

---

## 8. Repo map (what's where)

```
.
├── README.md            ← high-level summary, headline numbers
├── REPORT.md            ← the master writeup, every detail
├── WALKTHROUGH.md       ← this file (beginner-friendly)
├── assets/              ← charts and circuit graph PNGs
├── data/                ← train/valid/test datasets, demo prompts
├── configs/             ← LoRA training YAML files
├── results/             ← every model's eval output (~25 JSON files)
├── docs/                ← biology validation notes, presentation tables
├── slides/              ← presentation-ready slide_*.md files
└── src/                 ← all 39 Python scripts
    ├── sbol_eval_v2.py       ← THE rubric. Read this first if you want to understand scoring
    ├── jetson_sbol_eval_v2_http.py  ← runs the 100-prompt eval against any HTTP endpoint
    ├── chen_truong_system_prompt.py ← the long "essay" prompt
    ├── analyze_ablation.py   ← runs the 2×2 cell comparison
    ├── plot_tco.py           ← cost-vs-volume break-even chart
    ├── plot_efficiency.py    ← points-per-watt chart
    ├── render_sbol_circuit.py ← turns model JSON into a circuit diagram PNG
    ├── demo_stream.py        ← live-streaming demo (used in presentation)
    ├── train.py              ← MLX LoRA training entry point
    ├── generate_*.py         ← scripts that synthesized our training data
    └── (helper utilities)
```

---

## 9. Limitations (the honest version)

We made claims; here's where they break:

- **One model family tested end-to-end.** Qwen 3.5 27B (Mac) + Gemma 4 26B-A4B (Jetson). No data on Llama, Mistral, DeepSeek.
- **One task.** SBOL JSON generation only. We don't generate the actual DNA sequences, do part swapping, or validate in a wet lab.
- **The rubric is structural, not biological.** A circuit can score 95/100 and still propose using a yeast promoter in *E. coli* — the rubric only catches it if the wiring is wrong, not if the biology is wrong.
- **Single training seed.** Probably ±1 point variance run-to-run.
- **No in-vivo validation.** No biologist has built any of these circuits in a lab. We measure "is this a plausible circuit specification," not "does it work."

---

## 10. Where to go next

Things you could do with this work:

- **Try a different base model.** All you need is the same 1,162-row training set and a fine-tuning recipe.
- **Add domain-specific training data.** We're weakest on metabolic pathways and CRISPR systems — adding 200 hand-curated examples in those areas would likely close most of the gap to Opus.
- **Build a wet-lab validation pipeline.** Take our top-scoring circuits, build them in *E. coli*, measure if they behave as predicted. This is the only way to test whether structural correctness translates to biological correctness.
- **Try an even smaller model.** The Jetson result used a 26B-parameter model. What about an 8B model? A 3B model? At what size does quality cliff off?
- **Different output formats.** Generate SBOL3 XML directly instead of JSON. We have an SBOL3 converter in `src/json_to_sbol3.py` but didn't fine-tune on it.

---

## Citations

If you use this code, please cite:

- **Chen & Truong 2026**, *"Prompting LLMs for Synthetic Biology Design"* — baseline prompt techniques.
- **Qwen 3.5** (Alibaba) and **Gemma 4** (Google) — base models.
- **MLX** (Apple), **llama.cpp** (Gerganov), **Unsloth UD K-quants** — training and deployment tooling.
- **SynBioHub**, **iGEM Parts Registry**, **DataCurationProject** — sources for our scraped training tier.

Class project, EC552 — Boston University, Spring 2026.

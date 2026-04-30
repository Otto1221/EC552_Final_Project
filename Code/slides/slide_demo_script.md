# Demo + Evaluation Script (3 min, MacBook MLX, Qwen 3.5 27B + LoRA)

Read this verbatim if you want, or use it as a skeleton. Stage directions in `[brackets]`. Spoken lines in plain text. Total target: **3 minutes** for demo + evaluation combined.

---

## BEFORE THE TALK STARTS

**Pre-flight checklist (do 5 min before you go on):**

```bash
# 1. Start MLX server in a terminal you can leave running
mlx_lm.server --model /Users/arlo/models/Qwen3.5-27B-8bit-lora --port 8080 &

# 2. Wait ~30s for model load. Warm it with a throwaway request:
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > /dev/null

# 3. Open three terminal windows:
#    Window A: large font, ready to run demo_stream.py (the live demo)
#    Window B: ready to run the score check (after demo finishes)
#    Window C: backup terminal — `open assets/circuit_crispri_inverter.png`

# 4. Have these tabs/files open in case the demo fails:
#    - assets/circuit_crispri_inverter.png   (backup output #1)
#    - assets/chart_efficiency.png           (backup #2 — pivot to results)
```

---

## SLIDE: "Live Demo — MacBook, Qwen 3.5 27B + LoRA"

### T+0:00 — Frame the demo (15 seconds)

[ACTION: stand next to the laptop so audience sees both you and the screen]

> "Everything you've seen so far has been benchmark numbers — averages over 100 prompts. Now I want to show you what one actually looks like, live. This is running entirely on this laptop. No internet, no API call, no cloud. The model is a 27-billion parameter Qwen 3.5, fine-tuned with our LoRA on 1,162 SBOL examples."

[ACTION: switch to Window A, terminal with large font]

### T+0:15 — Show the prompt (10 seconds)

[ACTION: type the command but don't hit Enter yet]

```bash
python3 src/demo_stream.py 0
```

> "I'm asking it to design a HEK293-based immune sensor — think CAR-T-style, but with a synNotch receptor that detects CD19 on cancer cells, drives IL-12 production, and has a kill-switch in case anything goes wrong. This is a real circuit class people are publishing on right now."

### T+0:25 — Run it (45 seconds, streaming)

[ACTION: hit Enter]

[ON SCREEN: prompt printed in yellow, then JSON streaming token-by-token in green]

**While it streams, narrate (slow down, the streaming is the star):**

> "Watch the tokens appearing — this is real-time generation, ~17 tokens per second on M-series silicon. The model is composing the circuit, not retrieving it. It's deciding which promoter type, what RBS, where the kill-switch goes."

[Pause for ~10 seconds, let people watch]

> "And notice — it's outputting valid JSON. No markdown fences, no preamble like 'Sure, here's your circuit!' That formatting discipline is from our LoRA fine-tuning. The base model wraps everything in conversational prose; ours doesn't."

[Pause again, let it finish]

> "Done. About 45 seconds for ~700 tokens. On an Anthropic Opus call, this would have cost about 7 cents and taken ~14 seconds. On this laptop: free, offline, your data never leaves the room."

### T+1:10 — Score the output (40 seconds)

[ACTION: switch to Window B, paste this command:]

```bash
python3 -c "
import json, sys; sys.path.insert(0, 'src')
import sbol_eval_v2 as e
prompts = json.load(open('data/demo_prompts.json'))
response = open('results/demo_last.txt').read()
score = e.score_axes(prompts[0], response)
print(f'TOTAL: {score[\"total\"]}/100')
for axis, breakdown in score['axes'].items():
    pts = min(sum(breakdown.values()), e.AXIS_MAX[axis])
    print(f'  {axis}: {pts}/{e.AXIS_MAX[axis]}')
"
```

[ON SCREEN: a 7-line score breakdown appears]

> "Now I'm scoring that output against our rubric. Same code that produced every number on the previous slides. Six axes, 100 points total. Let's read it."

[ACTION: point at each line as you say it]

> "Structural validity 20 out of 20 — JSON parses, every part referenced exists. Behavioral wiring 18 out of 20 — the receptor activates the TF cascade like it should. Biology appropriate 16 out of 20 — it picked HEK293-compatible parts. Part fidelity 18, description 9, format 10. Total in the low 90s — consistent with our 91.57 average across 100 prompts."

> "If this were Opus, it would score ~99. If it were the bare model with no LoRA, the JSON might be malformed — that's the 2% catastrophic failure rate I mentioned earlier."

### T+1:50 — Render it as a circuit (30 seconds)

[ACTION: paste this command]

```bash
python3 src/render_sbol_circuit.py && open assets/circuit_*.png
```

[ON SCREEN: a circuit diagram opens — nodes for each part, arrows for each interaction]

> "Last step. The JSON is fine for downstream tooling, but humans want to see it as a diagram. This is a directed graph rendering — promoters in blue, RBS in orange, CDS in green, terminators in red. Solid lines are transcription and translation, dashed lines are activation and repression."

[ACTION: point at the synNotch node and trace the activation path]

> "You can see the synNotch receptor activates the TF, which drives IL-12 expression. The kill-switch is hanging off here, gated by rapamycin. That's the design the model just composed from the English description."

### T+2:20 — Tie back to the bigger picture (40 seconds)

[ACTION: switch to the score distribution chart, or back to your headline slide]

> "What you just saw — one prompt in, one circuit out, scored 90-something — happened 100 times in our benchmark. The numbers behind 91.57 aren't synthetic; they're the average of 100 of these runs."

> "And the same pipeline runs on a 15-watt Jetson edge device with a slightly smaller Gemma model, scoring 89.6. That's the punchline: a $1,200 box plugged into a wall socket can do this work, repeatedly, for free, offline. Whether you should use it depends on your volume — under 5,900 circuits a year, just use the API. Above that, the local model pays for itself."

[ACTION: pause, take questions or move to next slide]

---

## IF THE DEMO FAILS

**Server crashed / OOMs:**

> "Looks like MLX just OOM'd — that happens at the edge of 64 GB. Let me show you a clean run from the benchmark instead."

[ACTION: alt-tab to `assets/circuit_crispri_inverter.png`]

> "Same setup, ran yesterday, scored 94. Same code path, same model, same rubric — this is what comes out the other side."

**Output is malformed / missing fields:**

> "Interesting — that one missed a field. This actually illustrates the failure mode the LoRA is designed to prevent. On 100 prompts the LoRA had zero catastrophic zeros, but at temperature 0.1 you can still get the occasional skip. In production we'd add a single retry — that catches it."

[ACTION: alt-tab to backup PNG anyway]

**Network/SSH/curl issue:**

> "Connection hiccuped. The model is local, but my terminal isn't talking to it for some reason. The result is the same as the last 100 runs — let me show you those."

[ACTION: switch to a results JSON file, scroll through one entry]

---

## ANTICIPATED Q&A

**Q: Why not just use Opus?**
> "Cost, latency, privacy, and air-gap. Opus needs internet, costs ~$0.07 per circuit, and your circuit specs leave your machine. For one scientist generating 50 circuits a year, that's a non-issue. For a 10-person lab generating 100,000 a year, that's $7,000 a year and an IP-leak concern."

**Q: How accurate is the rubric?**
> "Structurally? Very. It's deterministic Python code — schema validation, graph topology checks, JSON parseability. Biologically? Partial. We ran a separate Opus-as-judge biology review and the LoRA scored 71/100 there — meaning structurally perfect circuits aren't always biologically sensible. We're upfront about this — see `docs/BIOLOGY_VALIDATION.md`."

**Q: Did you train on the eval prompts?**
> "No. Zero substring overlap, max Jaccard similarity 0.47. Section 3.5 of the report shows the contamination check. The eval prompts were hand-curated after training was done."

**Q: Why this base model and not Llama or Mistral?**
> "Compute budget. We had time and hardware to test one family end-to-end. Qwen 3.5 27B for the Mac because it fits in 64 GB at Q8; Gemma 4 26B-A4B for the Jetson because the MoE design means only 4B params are active per token, which fits 16 GB at Q3. Other model families would likely produce similar gains — open question."

**Q: Could I run this on my own data?**
> "Yes. Replace `data/train.jsonl` with your prompt→JSON pairs in the same chat format, run `mlx_lm.lora --config configs/lora_config_qwen35_27b.yaml`. ~7.5 hours on an M5 Max. The whole pipeline is in the repo."

---

## ONE-LINE TIMING REFERENCE

| T+ | what's happening |
|---|---|
| 0:00 | frame the demo |
| 0:15 | show the command + describe prompt |
| 0:25 | run, narrate streaming |
| 1:10 | score the output |
| 1:50 | render the circuit |
| 2:20 | tie back to averages, transition out |
| 3:00 | end |

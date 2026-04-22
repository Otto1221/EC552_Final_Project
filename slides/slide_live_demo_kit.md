# Live Demo Kit — 10 Novel SBOL Prompts

Ready-to-run demo set for live presentation. None of these prompts appear in the 100-prompt eval set. Each is chosen to showcase a specific capability of the LoRA-tuned Jetson stack on circuits it has never seen.

## The 10 prompts

| ID | Diff | Topo | Organism | Hook |
|---|:---:|---|---|---|
| demo_01_cancer_sensor | 5 | biosensor | mammalian | synNotch CAR-T sensor with kill-switch |
| demo_02_carbon_capture | 5 | pathway | ecoli | Synthetic CO₂ fixation cascade |
| demo_03_light_switch | 4 | inducible | ecoli | Blue-light (EL222) optogenetic |
| demo_04_bacterial_oscillator | 5 | oscillator | ecoli | Dual-feedback delayed oscillator + QS reporter |
| demo_05_heavy_metal_sensor | 4 | biosensor | ecoli | Naked-eye arsenic water test |
| demo_06_pathogen_logic | 5 | gate | ecoli | 3-input AND gate via split-T7 + intein |
| demo_07_gut_therapeutic | 5 | pathway | ecoli | IBD probiotic: senses butyrate → IL-10 + nanobody |
| demo_08_recombinase_memory | 4 | toggle | ecoli | Bxb1 integrase 1-bit genetic memory |
| demo_09_plant_stress | 4 | inducible | plant | Drought reporter (DREB2A/rd29A) |
| demo_10_cell_free_diagnostic | 4 | biosensor | cellfree | Paper-based Zika toehold switch |

Full prompt text in `demo_prompts.json`.

## How to run the demo (when Jetson is back up)

```bash
# 1. On Jetson — start llama-server with LoRA hot-loaded
cd /home/visionx/newgenes
./bin/llama-server \
  --model models/gemma-4-4b-it-UD-Q3_K_M.gguf \
  --lora models/gemma4-lora-fp16.gguf \
  --ctx-size 3072 --cache-ram 0 --ctx-checkpoints 0 \
  --host 0.0.0.0 --port 8080 &

# 2. From Mac — tunnel + run the demo set
ssh -L 18080:localhost:8080 visionx@192.168.55.1 -N &
cd /Users/arlo/Newgenes/finetune
python3 run_demo_prompts.py --url http://localhost:18080 --out demo_results.json

# 3. Score + render the top-3
python3 -c "
import json, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('e', Path('sbol_eval_v2.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r = json.load(open('demo_results.json'))
scored = []
for x in r:
    if 'error' in x: continue
    obj = m.extract_json(x['response'])
    if obj is None: s = {'total': 0}
    else: s = m.score_entry(x['entry'], obj)
    scored.append((x['id'], s['total'], len(x['response']), x.get('time',0)))
for id_, t, cl, dt in sorted(scored, key=lambda t: -t[1]):
    print(f'{id_:<30} {t:>3}/100  {cl:>5}ch  {dt:>5.1f}s')
" > demo_scores.txt
```

## What to show on the slide during the live demo

1. **Show the prompt on screen** — let the audience read the complex bio description (e.g., the synNotch cancer sensor) before the model runs.

2. **Type the curl one-liner live** — `curl http://jetson.local:8080/v1/chat/completions ...` — drives home that this is a local HTTP endpoint, not an API call.

3. **Streamed output** — with streaming enabled, the audience watches JSON generate token-by-token at 7 tok/s. A 600-token response takes ~90 seconds — enough to narrate the rubric live.

4. **Render in real-time** — pipe the JSON into `render_sbol_circuit.py` and display the generated PNG. The circuit graph appears as the last token lands.

5. **Read the `behavior` field aloud** — this is where the model explains the circuit in plain English. For demo_07 (gut therapeutic), it will describe a self-limiting IBD probiotic in biologist-grade prose.

## Demo prompt selection criteria (why these 10)

- **d4–d5 bias**: LoRA's advantage over bare is biggest here (+6.15 on d5).
- **Visual topology diversity**: oscillator (loops), gate (branching), biosensor (simple), pathway (linear cascade) — each renders differently.
- **Real-world payload**: cancer immunotherapy, CO₂ capture, arsenic water test, Zika diagnostic, IBD therapeutic, plant stress — every prompt maps to a recognizable application.
- **Mechanism diversity**: optogenetic, recombinase, toehold switch, synNotch, split-T7, intein — every prompt uses a different non-trivial bio mechanism.

## Top 3 expected live-demo picks (pre-screening based on similar eval prompts)

Based on per-topology scores on the held-out eval (LoRA + Jetson):

1. **demo_04 bacterial_oscillator** — oscillator topology scored 99.0 on the 2×2 eval; bright loop graph renders dramatically.
2. **demo_06 pathogen_logic** — gate topology scored 97.5; three-input AND is a crowd-pleaser.
3. **demo_07 gut_therapeutic** — pathway topology is the hardest (89.7), and a living-therapeutic narrative is the presentation-worthy framing.

Run the full set and pick the final 3 based on *both* rubric score and visual rendering quality.

# QS Money Shot — bare+Q3_K_M vs LoRA+Q3_K_M

## Prompt (d5, E. coli, topology = qs)

> Population-level consensus circuit using AHL-mediated quorum sensing to synchronize GFP pulses across cells
> Keywords: ahl, gfp, synchrony

## Result

| | Score | Finish | Chars | Notes |
|---|---:|---|---:|---|
| **UD-Q3_K_M bare** | **0** | stop | 2627 | JSON parse failure |
| **UD-Q3_K_M + LoRA** | **93** | stop | 2350 | Valid SBOL circuit |

**Δ = +93 points.**

## Why bare scored 0 — a single bad token

Bare model emitted a C-style comment mid-JSON:

```json
    ...
    { "name": "ahl_molecule", "type": "other", ...  }
  ],
  // Note: The circuit logic implies a positive feedback loop via AHL
  "interactions": [
    ...
```

JSON does not permit `//` comments. Parser fails → every structural axis scores 0 → total 0.

The rest of the bare design is actually reasonable — components are present, topology is roughly correct. The LoRA's uplift on QS is not fundamentally about biology: **it's about the LoRA learning to emit structurally-parseable output under the SBOL schema**. One bad habit (adding explanatory comments) costs the bare model the entire d5 QS class.

## LoRA output (abbreviated)

```json
{
  "name": "population_consensus_gfp_pulse",
  "components": [
    {"name":"p_lux","type":"promoter",...},
    {"name":"luxr_cds","type":"cds",...},
    {"name":"luxi_cds","type":"cds",...},
    {"name":"gfp_cds","type":"cds",...},
    /* 7 more components */
  ],
  "interactions": [
    {"from":"p_constitutive","to":"luxr_cds","type":"transcription"},
    {"from":"luxr_cds","to":"p_lux","type":"activation"},
    {"from":"p_lux","to":"luxi_cds","type":"transcription"},
    {"from":"luxi_cds","to":"p_lux","type":"activation"},
    {"from":"p_lux","to":"gfp_cds","type":"transcription"}
  ],
  "behavior": "Positive feedback loop via AHL production and LuxR
   activation creates a quorum-sensing threshold that triggers
   synchronized GFP expression across the population.",
  "organism": "Escherichia coli"
}
```

Valid JSON. Complete positive-feedback loop. Correct QS semantics.

## Takeaway

LoRA's +24-point average uplift on quorum-sensing circuits (90.5 vs 66.5)
is partly *structural discipline*, not biology knowledge. Fine-tuning
teaches the model to stay inside the schema even when the response
genuinely requires an explanation — the LoRA puts that explanation in a
`behavior` field rather than a comment.

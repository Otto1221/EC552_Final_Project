#!/usr/bin/env python3
"""Merge separate expert gate_proj/up_proj LoRA weights into gate_up_proj.

The GGUF LoRA converter expects experts.gate_up_proj (merged), but MLX
trains separate gate_proj and up_proj LoRA weights. This script merges
them using a block-diagonal B matrix and concatenated A matrix.

Scaling compensation: llama.cpp computes scale = alpha / rank per-tensor.
Original: alpha=16, rank=8 → scale=2. Merged: alpha=16, rank=16 → scale=1.
To preserve the same effective delta, B values are scaled by 2x.
"""
import json, re, torch
from safetensors.torch import load_file, save_file
from pathlib import Path

SRC = Path("./peft_adapter_hf")
DST = Path("./peft_adapter_merged")


def process():
    DST.mkdir(exist_ok=True)

    tensors = load_file(str(SRC / "adapter_model.safetensors"))
    with open(SRC / "adapter_config.json") as f:
        config = json.load(f)

    original_rank = config["r"]
    alpha = config["lora_alpha"]
    scale_factor = 2.0  # rank doubles from 8→16, so B needs 2x to compensate

    print(f"Original rank: {original_rank}, alpha: {alpha}")
    print(f"Scale factor for merged B: {scale_factor}")

    gate_a = {}  # layer → tensor
    gate_b = {}
    up_a = {}
    up_b = {}
    out_tensors = {}

    for name, tensor in tensors.items():
        m = re.search(r'layers\.(\d+)\.experts\.gate_proj\.lora_([AB])\.weight', name)
        if m:
            layer = int(m.group(1))
            if m.group(2) == 'A':
                gate_a[layer] = tensor
            else:
                gate_b[layer] = tensor
            continue

        m = re.search(r'layers\.(\d+)\.experts\.up_proj\.lora_([AB])\.weight', name)
        if m:
            layer = int(m.group(1))
            if m.group(2) == 'A':
                up_a[layer] = tensor
            else:
                up_b[layer] = tensor
            continue

        out_tensors[name] = tensor

    layers_to_merge = sorted(set(gate_a.keys()) & set(gate_b.keys()) & set(up_a.keys()) & set(up_b.keys()))
    print(f"\nMerging gate+up LoRA for {len(layers_to_merge)} layers: {layers_to_merge}")

    for layer in layers_to_merge:
        ga = gate_a[layer]  # [128, 8, 2816]
        gb = gate_b[layer]  # [128, 704, 8]
        ua = up_a[layer]    # [128, 8, 2816]
        ub = up_b[layer]    # [128, 704, 8]

        n_experts = ga.shape[0]
        rank = ga.shape[1]
        in_features = ga.shape[2]
        out_gate = gb.shape[1]
        out_up = ub.shape[1]

        merged_rank = rank * 2

        # A: concat along rank dim → [128, 16, 2816]
        merged_a = torch.cat([ga, ua], dim=1)

        # B: block-diagonal with 2x scaling → [128, 1408, 16]
        merged_b = torch.zeros(n_experts, out_gate + out_up, merged_rank, dtype=gb.dtype)
        merged_b[:, :out_gate, :rank] = scale_factor * gb
        merged_b[:, out_gate:, rank:] = scale_factor * ub

        prefix = f"base_model.model.model.language_model.layers.{layer}.experts"
        out_tensors[f"{prefix}.gate_up_proj.lora_A.weight"] = merged_a
        out_tensors[f"{prefix}.gate_up_proj.lora_B.weight"] = merged_b

        print(f"  Layer {layer}: gate_A{list(ga.shape)} + up_A{list(ua.shape)} → A{list(merged_a.shape)}")
        print(f"  Layer {layer}: gate_B{list(gb.shape)} + up_B{list(ub.shape)} → B{list(merged_b.shape)} (scaled {scale_factor}x)")

    save_file(out_tensors, str(DST / "adapter_model.safetensors"))

    # Update target_modules in config
    new_targets = [t for t in config["target_modules"] if t not in ("experts.gate_proj", "experts.up_proj")]
    new_targets.append("experts.gate_up_proj")
    config["target_modules"] = sorted(new_targets)

    with open(DST / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nOutput: {DST}/")
    print(f"Tensors: {len(tensors)} → {len(out_tensors)} (merged {len(layers_to_merge)} gate+up pairs)")
    print(f"Target modules: {config['target_modules']}")


if __name__ == "__main__":
    process()

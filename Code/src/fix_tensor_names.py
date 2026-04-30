#!/usr/bin/env python3
"""Rename MLX-fused tensor names to HuggingFace format for GGUF conversion.

MLX fuse produces tensor names like:
  language_model.model.layers.N.experts.switch_glu.{gate,up,down}_proj.weight

HuggingFace original uses:
  model.language_model.layers.N.experts.{gate_up_proj, down_proj}

This script transforms the names and merges gate+up projections.
"""
import json, os, glob, re
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

SRC = Path("./merged_model")
DST = Path("./merged_model_hf")


def rename_tensor(name):
    """Map MLX tensor name to HuggingFace format."""
    # Prefix: language_model.model. → model.language_model.
    if name.startswith("language_model.model."):
        name = "model.language_model." + name[len("language_model.model."):]

    # Expert tensors: remove switch_glu, remove .weight
    if ".experts.switch_glu." in name:
        name = name.replace(".experts.switch_glu.", ".experts.")
        if name.endswith(".weight"):
            name = name[:-len(".weight")]

    return name


def process():
    DST.mkdir(exist_ok=True)

    shard_files = sorted(glob.glob(str(SRC / "*.safetensors")))
    print(f"Processing {len(shard_files)} shards...")

    new_weight_map = {}
    all_pending_gates = {}  # key: (shard_idx, layer) → tensor
    all_pending_ups = {}

    # First pass: load all tensors, identify gate/up pairs to merge
    shard_tensors = []
    for shard_file in shard_files:
        tensors = load_file(shard_file)
        shard_tensors.append((shard_file, tensors))

    # Build output shards
    for shard_idx, (shard_file, tensors) in enumerate(shard_tensors):
        out_tensors = {}
        shard_name = os.path.basename(shard_file)

        gate_proj_by_layer = {}
        up_proj_by_layer = {}

        for old_name, tensor in tensors.items():
            new_name = rename_tensor(old_name)

            # Collect gate and up projections for merging
            m = re.search(r'layers\.(\d+)\.experts\.gate_proj$', new_name)
            if m:
                layer = int(m.group(1))
                gate_proj_by_layer[layer] = tensor
                continue

            m = re.search(r'layers\.(\d+)\.experts\.up_proj$', new_name)
            if m:
                layer = int(m.group(1))
                up_proj_by_layer[layer] = tensor
                continue

            out_tensors[new_name] = tensor

        # Merge gate+up pairs in this shard
        for layer in sorted(gate_proj_by_layer.keys()):
            if layer in up_proj_by_layer:
                gate = gate_proj_by_layer[layer]
                up = up_proj_by_layer[layer]
                # gate: [num_experts, moe_intermediate, hidden]
                # up:   [num_experts, moe_intermediate, hidden]
                # merged: [num_experts, 2*moe_intermediate, hidden]
                merged = torch.cat([gate, up], dim=1)
                merged_name = f"model.language_model.layers.{layer}.experts.gate_up_proj"
                out_tensors[merged_name] = merged
                print(f"  Merged layer {layer}: gate{list(gate.shape)} + up{list(up.shape)} → gate_up_proj{list(merged.shape)}")
            else:
                # gate without matching up in this shard — store for cross-shard merge
                all_pending_gates[(shard_idx, layer)] = gate_proj_by_layer[layer]

        for layer in up_proj_by_layer:
            if layer not in gate_proj_by_layer:
                all_pending_ups[(shard_idx, layer)] = up_proj_by_layer[layer]

        # Save output shard
        save_file(out_tensors, str(DST / shard_name))
        for name in out_tensors:
            new_weight_map[name] = shard_name
        print(f"  Saved {shard_name}: {len(out_tensors)} tensors")

    if all_pending_gates or all_pending_ups:
        print(f"\n  WARNING: {len(all_pending_gates)} unmerged gate projections, {len(all_pending_ups)} unmerged up projections")
        print("  Gate/up pairs split across shards — merging cross-shard...")
        # Find matching pairs
        gate_layers = {layer for (_, layer) in all_pending_gates}
        up_layers = {layer for (_, layer) in all_pending_ups}
        for layer in gate_layers & up_layers:
            gate_key = [k for k in all_pending_gates if k[1] == layer][0]
            up_key = [k for k in all_pending_ups if k[1] == layer][0]
            gate = all_pending_gates[gate_key]
            up = all_pending_ups[up_key]
            merged = torch.cat([gate, up], dim=1)
            merged_name = f"model.language_model.layers.{layer}.experts.gate_up_proj"
            # Add to the first shard that had the gate tensor
            target_shard = os.path.basename(shard_files[gate_key[0]])
            target_path = DST / target_shard
            existing = load_file(str(target_path))
            existing[merged_name] = merged
            save_file(existing, str(target_path))
            new_weight_map[merged_name] = target_shard
            print(f"  Cross-shard merged layer {layer} → {target_shard}")

    # Write index file
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in
                      [v for sf in [load_file(str(DST / f)) for f in set(new_weight_map.values())] for v in sf.values()])},
        "weight_map": dict(sorted(new_weight_map.items()))
    }
    with open(DST / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy config files
    import shutil
    for cfg in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
        src_path = SRC / cfg
        if src_path.exists():
            shutil.copy2(src_path, DST / cfg)

    # Fix config.json: ensure it has text_config at top level for the converter
    with open(DST / "config.json") as f:
        config = json.load(f)

    # The converter reads hparams from text_config, make sure it's accessible
    if "text_config" in config:
        # Flatten text_config keys into top level for converter compatibility
        for k, v in config["text_config"].items():
            if k not in config:
                config[k] = v

    with open(DST / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Output: {DST}/")
    print(f"Total tensors: {len(new_weight_map)}")


if __name__ == "__main__":
    process()

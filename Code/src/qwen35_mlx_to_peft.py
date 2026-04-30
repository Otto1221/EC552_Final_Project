#!/usr/bin/env python3
"""Convert Qwen 3.5 27B MLX LoRA adapter to HuggingFace PEFT format.

Qwen 3.5 is dense (no MoE merge needed) and has standard module names.
LoRA is applied to self_attn (q/k/v/o_proj), MLP (gate/up/down_proj),
and linear_attn (in_proj_a/b/qkv/z, out_proj — GatedDeltaNet projections).

MLX name:  language_model.model.layers.N.MODULE.lora_a (or .lora_b)
PEFT name: base_model.model.language_model.model.layers.N.MODULE.lora_A.weight
"""
import json, os, sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

ADAPTER_PATH = Path("./adapters/qwen35-27b-newgenes")
PEFT_DIR = Path("./qwen35_peft_adapter")
BASE_MODEL_HF = "Qwen/Qwen3.5-27B"


def main():
    PEFT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ADAPTER_PATH / "adapter_config.json") as f:
        mlx_config = json.load(f)

    weights = dict(mx.load(str(ADAPTER_PATH / "adapters.safetensors")))
    print(f"Loaded {len(weights)} MLX LoRA tensors")

    peft_weights = {}
    target_modules = set()
    for name, tensor in weights.items():
        arr = np.array(tensor.astype(mx.float32))

        if name.endswith(".lora_a"):
            suffix = ".lora_A.weight"
            peft_base = name[:-len(".lora_a")]
        elif name.endswith(".lora_b"):
            suffix = ".lora_B.weight"
            peft_base = name[:-len(".lora_b")]
        else:
            raise ValueError(f"unexpected tensor name: {name}")

        arr = arr.T if arr.ndim == 2 else arr
        peft_name = f"base_model.model.{peft_base}{suffix}"
        peft_weights[peft_name] = arr

        parts = peft_base.split(".")
        target_modules.add(parts[-1])

    save_file(peft_weights, str(PEFT_DIR / "adapter_model.safetensors"))
    total_mb = sum(v.nbytes for v in peft_weights.values()) / 1e6
    print(f"Saved PEFT adapter_model.safetensors ({total_mb:.1f} MB, {len(peft_weights)} tensors)")

    peft_config = {
        "auto_mapping": None,
        "base_model_name_or_path": BASE_MODEL_HF,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "lora_alpha": mlx_config["lora_parameters"]["alpha"],
        "lora_dropout": mlx_config["lora_parameters"]["dropout"],
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": mlx_config["lora_parameters"]["rank"],
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }
    with open(PEFT_DIR / "adapter_config.json", "w") as f:
        json.dump(peft_config, f, indent=2)

    print(f"Target modules: {sorted(target_modules)}")
    print(f"PEFT adapter written to {PEFT_DIR}/")


if __name__ == "__main__":
    main()

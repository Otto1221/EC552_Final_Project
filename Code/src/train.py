#!/usr/bin/env python3
"""Training wrapper that patches LoRA MoE weights to bfloat16 before training.

MLX 0.31.1's metallib lacks float32 steel_gather_mm kernels. The LoRA adapter's
SwitchLinear weights (lora_a, lora_b) default to float32, which triggers the
missing kernel during the MoE gather_mm. This script patches them to bfloat16
before launching the standard mlx_lm.lora training.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner import lora

_orig_init = lora.LoRASwitchLinear.__init__

def _patched_init(self, input_dims, output_dims, num_experts, r=8, dropout=0.0, scale=20.0, bias=False):
    _orig_init(self, input_dims, output_dims, num_experts, r=r, dropout=dropout, scale=scale, bias=bias)
    self.lora_a = self.lora_a.astype(mx.bfloat16)
    self.lora_b = self.lora_b.astype(mx.bfloat16)

lora.LoRASwitchLinear.__init__ = _patched_init

from mlx_lm.lora import main
main()

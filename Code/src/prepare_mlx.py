#!/usr/bin/env python3
"""Convert OpenAI JSONL training data to MLX-LM chat fine-tuning format."""
import json
import os

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
OUT_DIR = os.path.join(SRC_DIR, 'mlx_data')
os.makedirs(OUT_DIR, exist_ok=True)

for split in ['train', 'valid', 'test']:
    src = os.path.join(SRC_DIR, f'{split}.jsonl')
    dst = os.path.join(OUT_DIR, f'{split}.jsonl')
    count = 0
    with open(src) as f_in, open(dst, 'w') as f_out:
        for line in f_in:
            d = json.loads(line)
            # MLX-LM expects {"messages": [...]} format — same as OpenAI
            f_out.write(json.dumps({"messages": d["messages"]}, ensure_ascii=False) + '\n')
            count += 1
    print(f'{split}: {count} examples → {dst}')

print(f'\nData ready in {OUT_DIR}/')

#!/usr/bin/env python3
"""Efficiency bar chart: rubric points per watt."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets" / "chart_efficiency.png"
OUT.parent.mkdir(exist_ok=True)

# [label, score, active_watts, color, note]
# Active watts = power during inference, not idle
ROWS = [
    ("Opus 4.7\n(cloud API)",         99.22,  350.0, "#c44e52", "estimated datacenter server W/request,\ntypical per-request allocation"),
    ("MacBook M5 Max\nQ8 LoRA (MLX)", 92.2,   50.0, "#4c72b0", "measured peak during inference"),
    ("Jetson Orin NX\nUD-Q3_K_M + LoRA", 89.60, 15.0, "#55a868", "~15W TDP, measured sustained"),
    ("Jetson Orin NX\nUD-Q3_K_M bare",   87.17, 15.0, "#9aab6a", "same hardware, no LoRA"),
]

labels = [r[0] for r in ROWS]
scores = [r[1] for r in ROWS]
watts  = [r[2] for r in ROWS]
colors = [r[3] for r in ROWS]
ratio  = [s/w for s, w in zip(scores, watts)]

fig, axes = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={"width_ratios":[1, 1, 1.4]})

# Panel 1: Score
ax = axes[0]
bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="black", linewidth=0.8)
for i, (b, s) in enumerate(zip(bars, scores)):
    ax.text(b.get_x()+b.get_width()/2, s+1, f"{s:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Rubric score (max 100)", fontsize=10)
ax.set_title("Quality", fontsize=12)
ax.set_ylim(0, 108)
ax.grid(axis="y", ls="--", alpha=0.3)

# Panel 2: Power
ax = axes[1]
bars = ax.bar(range(len(labels)), watts, color=colors, edgecolor="black", linewidth=0.8)
for i, (b, w) in enumerate(zip(bars, watts)):
    ax.text(b.get_x()+b.get_width()/2, w+8, f"{w:.0f} W", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Active power (watts)", fontsize=10)
ax.set_title("Power draw", fontsize=12)
ax.set_ylim(0, max(watts)*1.18)
ax.grid(axis="y", ls="--", alpha=0.3)

# Panel 3: Points per watt
ax = axes[2]
bars = ax.barh(range(len(labels)), ratio, color=colors, edgecolor="black", linewidth=0.8)
for i, (b, r) in enumerate(zip(bars, ratio)):
    ax.text(r+0.1, b.get_y()+b.get_height()/2, f"{r:.2f} pts/W",
            va="center", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Rubric points per watt (higher = better)", fontsize=10)
ax.set_title("Efficiency  (score / W)", fontsize=12)
ax.set_xlim(0, max(ratio)*1.20)
ax.grid(axis="x", ls="--", alpha=0.3)

# Ratio vs Opus annotation
jet_ratio = ratio[2] / ratio[0]
fig.text(0.5, 0.02,
    f"Jetson LoRA delivers {jet_ratio:.1f}× the points-per-watt of an Opus API call "
    f"— at 90% of the quality.",
    ha="center", fontsize=11, style="italic", color="#333")

fig.suptitle("SBOL Generation Efficiency — Cloud vs Local Platforms", fontsize=13, y=0.97)
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")

print("\nSummary:")
for r, label, s, w, rt in zip(ROWS, labels, scores, watts, ratio):
    print(f"  {label.replace(chr(10),' '):<40}  {s:>5.2f} pts  {w:>5.0f} W  → {rt:>5.2f} pts/W")

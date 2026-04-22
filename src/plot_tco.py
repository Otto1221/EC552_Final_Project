#!/usr/bin/env python3
"""TCO (total cost of ownership) break-even chart.

Compares Opus API vs MacBook Q8 vs Jetson Orin NX across circuit volumes.
All numbers amortized over a 3-year horizon for capex.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets" / "chart_tco.png"
OUT.parent.mkdir(exist_ok=True)

HORIZON_YEARS = 3

# Opus 4.7 API pricing (measured from sbol_eval_v2_opus_47.json):
# ~125 in-tokens + ~880 out-tokens per SBOL ≈ $0.0680 per circuit
OPUS_PER_CIRCUIT = 0.0680

# MacBook M5 Max 64GB
MAC_CAPEX = 4000.0       # one-time hardware
MAC_POWER_W = 50.0       # sustained during inference
MAC_TOK_PER_S = 17.85    # from our measurements
MAC_TOK_PER_CIRCUIT = 880
MAC_SECONDS_PER_CIRCUIT = MAC_TOK_PER_CIRCUIT / MAC_TOK_PER_S  # ~49s

# Jetson Orin NX 16GB
JETSON_CAPEX = 1200.0
JETSON_POWER_W = 15.0
JETSON_TOK_PER_S = 7.04
JETSON_TOK_PER_CIRCUIT = 870
JETSON_SECONDS_PER_CIRCUIT = JETSON_TOK_PER_CIRCUIT / JETSON_TOK_PER_S  # ~124s

KWH_PRICE = 0.15  # $/kWh, typical US commercial rate

def local_annual_cost(capex, power_w, seconds_per_circuit, n_per_year):
    amortized = capex / HORIZON_YEARS
    active_hours = n_per_year * seconds_per_circuit / 3600
    power_cost = active_hours * (power_w / 1000) * KWH_PRICE
    return amortized + power_cost

def opus_annual_cost(n_per_year):
    return n_per_year * OPUS_PER_CIRCUIT

# ----- plot ------------------------------------------------------------------
xs = np.logspace(2, 5.5, 200)  # 100 → 316k circuits/yr

opus  = np.array([opus_annual_cost(x) for x in xs])
mac   = np.array([local_annual_cost(MAC_CAPEX,    MAC_POWER_W,    MAC_SECONDS_PER_CIRCUIT,    x) for x in xs])
jet   = np.array([local_annual_cost(JETSON_CAPEX, JETSON_POWER_W, JETSON_SECONDS_PER_CIRCUIT, x) for x in xs])

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.plot(xs, opus, color="#c44e52", lw=2.5, label=f"Opus 4.7 API  (${OPUS_PER_CIRCUIT:.3f}/circuit)")
ax.plot(xs, mac,  color="#4c72b0", lw=2.5, label=f"MacBook M5 Max  (${MAC_CAPEX:.0f} capex)")
ax.plot(xs, jet,  color="#55a868", lw=2.5, label=f"Jetson Orin NX  (${JETSON_CAPEX:.0f} capex)")

# break-even points
# Opus crosses Jetson when opus_cost == jetson_cost
def crossover(f, g, lo, hi):
    from scipy.optimize import brentq  # lazy
    try:
        return brentq(lambda x: f(x) - g(x), lo, hi)
    except Exception:
        return None

try:
    from scipy.optimize import brentq
    j_break = brentq(lambda x: opus_annual_cost(x) - local_annual_cost(JETSON_CAPEX, JETSON_POWER_W, JETSON_SECONDS_PER_CIRCUIT, x), 100, 100000)
    m_break = brentq(lambda x: opus_annual_cost(x) - local_annual_cost(MAC_CAPEX, MAC_POWER_W, MAC_SECONDS_PER_CIRCUIT, x), 100, 200000)
except ImportError:
    # fallback: manual bisection
    def bisect(f, lo, hi):
        for _ in range(60):
            m = (lo + hi) / 2
            if f(m) > 0: hi = m
            else: lo = m
        return (lo + hi) / 2
    j_break = bisect(lambda x: opus_annual_cost(x) - local_annual_cost(JETSON_CAPEX, JETSON_POWER_W, JETSON_SECONDS_PER_CIRCUIT, x), 100, 100000)
    m_break = bisect(lambda x: opus_annual_cost(x) - local_annual_cost(MAC_CAPEX, MAC_POWER_W, MAC_SECONDS_PER_CIRCUIT, x), 100, 200000)

for bp, label, color in [(j_break, f"Jetson breaks even\n{j_break:.0f} circuits/yr", "#55a868"),
                         (m_break, f"MacBook breaks even\n{m_break:.0f} circuits/yr", "#4c72b0")]:
    cost_at_bp = opus_annual_cost(bp)
    ax.scatter([bp], [cost_at_bp], s=90, color=color, zorder=5, edgecolor="black", linewidths=1.2)
    ax.annotate(label, xy=(bp, cost_at_bp),
                xytext=(bp*1.5, cost_at_bp*0.45),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

# Typical biotech lab shading (50-500 circuits/week = 2.5k-25k/yr)
ax.axvspan(2500, 25000, alpha=0.10, color="gray")
ax.text(np.sqrt(2500*25000), ax.get_ylim()[1]*0.80 if False else 30,
        "Typical biotech lab\n(50–500 circuits/wk)", ha="center", fontsize=9, color="gray")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Circuits generated per year", fontsize=11)
ax.set_ylabel(f"Total annual cost (USD, {HORIZON_YEARS}-yr amortization)", fontsize=11)
ax.set_title("Cloud vs Local SBOL generation — break-even analysis", fontsize=13)
ax.grid(which="both", ls="--", alpha=0.3)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

# Scenario dots — real-world reference points
SCENARIOS = [
    ("1 scientist casual\n(100/yr)", 100),
    ("1 active lab\n(10k/yr)", 10000),
    ("10-scientist org\n(100k/yr)", 100000),
]
for label, v in SCENARIOS:
    ax.axvline(v, color="#888", ls=":", alpha=0.3, lw=1)
    ax.text(v, 2.5, label, rotation=90, fontsize=8, color="#666", va="bottom", ha="right")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"\nJetson → Opus crossover: {j_break:>8.0f} circuits/yr  ({j_break/52:.0f}/wk)")
print(f"MacBook→ Opus crossover: {m_break:>8.0f} circuits/yr  ({m_break/52:.0f}/wk)")
print(f"\nAt 10,000 circuits/yr:")
print(f"  Opus:   ${opus_annual_cost(10000):>8.2f}/yr")
print(f"  Mac:    ${local_annual_cost(MAC_CAPEX, MAC_POWER_W, MAC_SECONDS_PER_CIRCUIT, 10000):>8.2f}/yr")
print(f"  Jetson: ${local_annual_cost(JETSON_CAPEX, JETSON_POWER_W, JETSON_SECONDS_PER_CIRCUIT, 10000):>8.2f}/yr")

# EVolvAI — Generative Counterfactual Framework

**Sujay's module** within the Spatio-Temporal EV Infrastructure Optimizer project.

Generates physics-grounded "what-if" EV charging demand scenarios for any intervention (severe winter storm, mass fleet electrification, etc.) and hands off the results as NumPy tensors to the rest of the team.

---

## What This Module Does

Traditional EV demand forecasting (LSTMs, RNNs) maps historical data to a future prediction — one deterministic answer. That's not enough for grid resilience planning, where you need to answer *"what if 100% of ICE vehicles converted to EV during a winter storm?"*

This module does that. It trains a **Variational Autoencoder with Temporal Convolutional Networks** conditioned on a 5-dimensional intervention vector. At inference, you change the vector and get a statistically realistic demand profile you've never seen in the data — a counterfactual.

---

## Team Handoff (What You Get)

Each generated scenario is a `.npy` file in `output/`:

| Spec | Value |
|---|---|
| Format | `.npy` (NumPy float32) |
| Shape | `[24, 50]` — 24 hours × 50 grid nodes |
| Units | **kW** — active power demand per node per hour |
| Hour 0 | Midnight → 01:00 (standard time) |

### Loading in your code

```python
import numpy as np

demand = np.load('output/extreme_winter_storm.npy')  # shape [24, 50]
# demand[hour, node_index] → kW at that hour for that node
```

Available scenarios out of the box:

| File | Description |
|---|---|
| `extreme_winter_storm.npy` | Severe cold + 2.5× fleet electrification |
| `summer_peak.npy` | Heat event + 1.5× electrification |
| `full_electrification.npy` | Normal weather + 3× fleet (full ICE→EV) |
| `extreme_winter_v2.npy` | Winter storm + 2.5× fleet + weekend traffic |
| `mock_demand_tensor.npy` | Synthetic placeholder — use this *right now* before training |

---

## Getting Started

**No GPU, no training required for the mock handoff:**

```bash
pip install numpy
python run.py mock
# → output/mock_demand_tensor.npy  [24, 50]
```

**Full pipeline (needs PyTorch):**

```bash
pip install -r requirements.txt
python run.py train     # trains the model
python run.py generate  # generates all scenario .npy files
```

---

## Async Handoff Protocol

1. **Now:** Use `output/mock_demand_tensor.npy`. Shape and units are identical to the real output — build your code against this today.
2. **After training:** Replace the mock path with `output/<scenario>.npy`. Zero code changes needed on your end.
3. **Later:** Optimization team feeds back critical-node bottlenecks → Sujay tightens the latent conditioning to stress-test those specific nodes.

### Data integrity rules

- Arrays will never contain `NaN` or `None`.
- Shape is always `[24, NUM_NODES]`. If `NUM_NODES` changes it will be announced — update one import and you're done.
- Hour 0 is always midnight standard time.

---

## What This Module Needs From You

| What | From | Format |
|---|---|---|
| Grid topology (bus index, node type, transformer kVA) | Lochan | `.json` |
| Historical charging logs (start, stop, energy kWh, peak kW) | External (Caltech ACN / NREL) | `.csv` or `.parquet` |
| Hourly weather (temp, precipitation, wind) | External (Open-Meteo) | `.csv` |

Place raw files in `data/raw/` and run `python data_pipeline/preprocess.py` once the merge logic is filled in.

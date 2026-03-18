# EVolvAI — Developer Roadmap

Internal reference for anyone building or extending this module. Covers project phases, architecture decisions, data pipeline, training, and integration.

---

## Project Structure

```
EVolvAI/
├── run.py                          # CLI: mock | train | generate | all
├── requirements.txt
├── generative_core/
│   ├── __init__.py                 # Package overview + architecture doc
│   ├── config.py                   # Single source of truth for all constants
│   ├── models.py                   # CausalConv1d, TCNBlock, TCN, GCD-VAE
│   ├── data_loader.py              # EVDemandDataset (real + synthetic fallback)
│   ├── train.py                    # Training loop + checkpoint save
│   ├── generate.py                 # Counterfactual generation from checkpoint
│   └── mock.py                     # Fast mock tensor generator (no torch needed)
├── data_pipeline/
│   └── preprocess.py               # [TODO] merge raw CSVs → train_data.parquet
├── environment_graph/              # Lochan: graph builder + AC power flow surrogate
├── frontend_dashboard/             # UI team: Streamlit/React map dashboard
├── data/
│   ├── raw/                        # Downloaded files — never modify these
│   └── processed/
│       └── train_data.parquet      # [TODO] output of preprocess.py
└── output/                         # Generated at runtime
    ├── mock_demand_tensor.npy
    ├── gcvae_model.pt
    └── <scenario_name>.npy
```

---

## Architecture

### Why TCN + VAE?

Standard LSTMs map one history → one future (deterministic). We need to generate *plausible distributions* of future demand under conditions that haven't occurred in the training data. The VAE gives us a structured latent space we can intervene on; the TCN handles the 24-hour temporal structure without vanishing gradients.

### Model pipeline

```
Input [B, NUM_FEATURES, 24]
  │
  ├─ Encoder TCN ─→ flatten ─→ fc_mu, fc_logvar  →  Z ~ N(μ, σ²)
  │                                                        │
  └─────────────────────────────────────────── cat([Z, C]) │
                                                           │
                                                      Decoder FC
                                                           │
                                                      Decoder TCN
                                                           │
                                               Output [B, NUM_FEATURES, 24]
```

`C` is the 5-dim condition vector. Changing `C` at inference time while keeping `Z` fixed produces a counterfactual of the *same latent demand state* under a different scenario.

### Condition vector layout

| Index | Meaning | Range |
|---|---|---|
| 0 | Temperature anomaly | float (deviation from seasonal avg) |
| 1 | EV electrification multiplier | float (1.0 = today's fleet) |
| 2 | Solar availability | 0.0 (cloudy) – 1.0 (clear) |
| 3 | Weekend flag | 0 or 1 |
| 4 | Holiday flag | 0 or 1 |

---

## Phase 1 — Data Contract & Setup ✅

- [ ] **Lochan + Sujay:** Agree on PyTorch tensor shapes (`.pt`) for the graph builder → GCD-VAE interface
- [ ] **Lochan:** Push `mock_graph_tensor.pt` with random noise in the agreed shape
- [ ] **Sujay:** Load `mock_graph_tensor.pt` to verify the pipeline connects

---

## Phase 2 — Parallel Development

### Track 3: Sujay (this repo)

**Data Pipeline**
- [ ] Download Caltech ACN-Data → `data/raw/acn_sessions.csv`
- [ ] Download Open-Meteo weather → `data/raw/weather.csv`
- [ ] Implement `data_pipeline/preprocess.py` to output:
  ```
  data/processed/train_data.parquet
  columns: date, hour, node_id, demand_kw
  ```
- [ ] Get grid topology JSON from Lochan → `data/grid_topology.json`

**Model**
- [x] GCD-VAE architecture (TCN encoder + conditioned decoder)
- [x] `COND_DIM=5` condition vector with 4 scenarios defined
- [x] Gradient clipping, mean-reduced loss, synthetic fallback in DataLoader
- [ ] Wire real parquet into `data_loader.py` (falls back to synthetic until then)
- [ ] Expand weather channels beyond synthetic random (tie to real CSV)

**Training**
- [ ] Upload project to Google Drive
- [ ] Open `EVolvAI_Training.ipynb` in Colab → Runtime: T4 GPU
- [ ] Set `EPOCHS = 50` minimum in config cell
- [ ] Monitor: loss should decrease steadily. If it plateaus, increase `LATENT_DIM`

**Tuning reference**

| Symptom | Fix |
|---|---|
| Loss explodes | Lower `LEARNING_RATE` to `1e-4` |
| Reconstructions are blurry | Raise `KLD_WEIGHT` gradually: 0.5 → 1.0 → 2.0 |
| Model memorises training data | Raise `DROPOUT` to 0.3, get more data |
| Loss stable but outputs flat | Raise `LATENT_DIM` (try 32 or 64) |

---

## Phase 3 — Integration

- [ ] Swap Lochan's mock tensors for real data pipeline output
- [ ] Run `python run.py generate` → hand `.npy` files to Lochan and UI team
- [ ] Run end-to-end latency tests on the full pipeline
- [ ] Generate attention-map visuals for the UI dashboard presentation

---

## Sector 3 — Literature Justification

Current literature on EV demand forecasting uses RNNs/LSTMs for deterministic baseline forecasting. These fail for two reasons relevant to this project:

1. **No extrapolation to unseen conditions.** A model trained on historical summers cannot generate a credible "100% fleet electrification winter storm" profile — it has never seen either condition.

2. **No causal intervention.** Standard conditional models (e.g., "condition on day of week") apply weak statistical conditioning. We need *intervention-based latent conditioning* — explicitly forcing the latent representation to reflect a physical change in the world.

Our architecture addresses both: the VAE learns a structured latent space of demand states; the causal TCN encoder captures temporal dependencies without information leakage; the decoder is conditioned on an explicit intervention vector that maps directly to physical and socioeconomic variables.

This transitions demand modelling from observation to **active scenario planning** — the essential input for Lochan's AC power-flow constraint engine and the optimization layer.

# EVolvAI: Generative Counterfactual Framework

## Overview
This repository contains the core deep learning architecture for the **Generative Counterfactual Framework** – a component of the EVolvAI project.

Traditional power systems forecasting relies on deterministic predictions from historical data (LSTMs, RNNs). These cannot extrapolate to extreme "what-if" scenarios necessary for grid resilience planning.

This module bridges that gap by combining **Temporal Convolutional Networks (TCNs)** with a **Variational Autoencoder (VAE)** and **intervention-based latent conditioning**. The system generates realistic 24-hour demand profiles mapped to physical grid topology under extreme or unseen conditions (e.g., severe winter storms + 100% EV fleet electrification).

---

## 📂 Project Structure

```
EVolvAI/
├── run.py                          # CLI entry point (mock / train / generate / all)
├── generative_model/
│   ├── __init__.py
│   ├── config.py                   # All hyperparameters and scenario definitions
│   ├── models.py                   # CausalConv1d, TCN, GenerativeCounterfactualVAE
│   ├── data_loader.py              # EVDemandDataset + DataLoader factory
│   ├── train.py                    # Training loop with model checkpoint saving
│   ├── generate.py                 # Counterfactual scenario generation
│   └── mock.py                     # Quick mock tensor generator for async handoff
├── EVolvAI_Training.ipynb          # Self-contained Colab notebook (GPU training)
├── sector_3_review.md              # Sector 3 literature review
├── SUJAY.md                        # Original task roadmap
└── output/                         # Generated at runtime
    ├── mock_demand_tensor.npy
    ├── gcvae_model.pt
    ├── extreme_winter_storm.npy
    ├── summer_peak.npy
    └── full_electrification.npy
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Training)
1. Upload `EVolvAI_Training.ipynb` to [Google Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The notebook is self-contained.

### Option 2: Local CLI (requires `pip install torch numpy`)
```bash
# Generate mock tensor for async handoff
python run.py mock

# Train the GCD-VAE model
python run.py train

# Generate all counterfactual scenarios from a trained model
python run.py generate

# Run the full pipeline (mock → train → generate)
python run.py all
```

---

## ⚙️ Configuration

All hyperparameters, scenario definitions, and output paths are centralized in [`generative_model/config.py`](generative_model/config.py). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `NUM_NODES` | 50 | Grid topology nodes |
| `SEQ_LEN` | 24 | Hours per profile |
| `LATENT_DIM` | 16 | VAE latent space size |
| `COND_DIM` | 2 | Condition vector size `[WeatherFlag, EV_Multiplier]` |
| `EPOCHS` | 10 | Training epochs |
| `SCENARIOS` | 3 defined | Dict of counterfactual triggers |

---

## 🔧 What Has Been Completed (Generative AI & Demand Modeler)

1. **Sector 3 Literature Review** – Justification for Causal ML and conditional VAEs over deterministic forecasting.
2. **Mock Handoff System** – Immediate `[24, N]` numpy tensor generator to unblock other teams.
3. **GCD-VAE Architecture** – TCN encoder/decoder with causal convolutions and latent conditioning.
4. **Remote Training Pipeline** – Google Colab notebook for cloud GPU training.

---

## ⏭️ Next Steps for the Pipeline

### For Grid Physics (Akshay) & Optimization (Teammate B)

**You do NOT need to wait for the AI models to finish training.**

1. **Use Mock Data Now:**
   ```bash
   python run.py mock
   ```
   This creates `output/mock_demand_tensor.npy` — a `[24, 50]` array in kW. Feed this directly into your `pandapower`/`MATPOWER` simulations and optimization algorithms.

2. **Swap to Real AI Output Later:**
   Once the AI Modeler provides trained scenario tensors (e.g., `output/extreme_winter_storm.npy`), simply replace the mock tensor path in your code. The output shape is identical — **no code changes needed on your end**.

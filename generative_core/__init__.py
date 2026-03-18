"""
EVolvAI – Generative Counterfactual Framework
==============================================

Package: generative_core
Author:  Sujay (ML – Generation & Optimisation)
Project: Spatio-Temporal EV Infrastructure Optimizer

Overview
--------
This package implements the core deep-learning pipeline that generates
synthetic-but-physically-realistic future EV charging demand scenarios
("counterfactuals") conditioned on intervention triggers such as extreme
weather events or fleet electrification surges.

Architecture
------------
The pipeline is built around a Graph Conditioned Diffusion VAE (GCD-VAE),
implemented with:

  * **Temporal Convolutional Networks (TCNs)** – causal convolutions capture
    long-range temporal dependencies across 24-hour demand windows without the
    vanishing-gradient problems of RNNs.

  * **Variational Autoencoder (VAE)** – encodes historical demand into a
    probabilistic latent space Z.  The decoder reconstructs node-level kW
    demand profiles from Z concatenated with a condition vector C.

  * **Intervention-based latent conditioning** – the condition vector C
    (length COND_DIM = 5) encodes physical and socioeconomic triggers:
        C[0] = temperature anomaly     (float, deviation from seasonal avg)
        C[1] = EV electrification mult (float, 1.0 = baseline fleet size)
        C[2] = solar availability      (float, 0.0 cloudy – 1.0 clear)
        C[3] = weekend flag            (0 or 1)
        C[4] = holiday flag            (0 or 1)

Data Flow
---------
    raw CSVs          parquet             DataLoader           Model
    (data/raw/)  →  (data/processed/)  →  (data_loader.py)  →  (models.py)
                                                                     ↓
                                                    Latent space Z + Condition C
                                                                     ↓
                                                        [24, NUM_NODES] kW tensor
                                                                     ↓
                                                    output/*.npy  (handoff to team)

Team Handoff
------------
The output tensors (shape [24, NUM_NODES]) are the async contract with:
  * Lochan  – feeds into pandapower AC-power-flow constraint engine.
  * UI team – parsed to render map overlays and grid-stress warnings.

Quick start (no GPU needed):
    python run.py mock      # instant [24, 50] tensor, no training required
    python run.py train     # trains GCD-VAE (needs `pip install torch`)
    python run.py generate  # generates all counterfactual scenarios
"""

from . import config  # noqa: F401 – re-exported for `from generative_core import config`

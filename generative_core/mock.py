"""
generative_core/mock.py
========================
Fast synthetic demand generator for async team handoff.

Purpose
-------
The real GCD-VAE model takes time to train.  This module provides a
lightweight [24, NUM_NODES] kW tensor that Lochan and the UI team can
use immediately to build and test their downstream systems.  The shape
and units are identical to the real model output – the only difference
is statistical: the mock uses a simple sinusoidal pattern whereas the
trained model learns the true demand distribution.

Swap path: replace `output/mock_demand_tensor.npy` with `output/<scenario>.npy`
once training is complete.  No interface changes are required on the consumers'
side – the file format and shape are guaranteed identical.

Requires only NumPy – no PyTorch dependency, so it runs on machines that
don't have a GPU or torch installed.
"""

import os

import numpy as np

from . import config


def generate_mock_demand(num_nodes: int = config.NUM_NODES,
                          num_hours: int = config.SEQ_LEN) -> np.ndarray:
    """Generate a single-day synthetic EV demand array.

    Shape and units are identical to the real model output so downstream
    code requires no changes when swapping in trained scenario tensors.

    The demand pattern has three components:
      1. Uniform random base (10–100 kW per node per hour).
      2. Diurnal sine-wave multiplier peaking around hour 18 (evening commute).
      3. Sparse fast-charging spikes (+150 kW, 5 % probability per cell).

    Args:
        num_nodes (int): Number of spatial grid nodes.
        num_hours (int): Hours in the profile window (default 24).

    Returns:
        float64 NumPy array of shape [num_hours, num_nodes] in kW.
    """
    base = np.random.uniform(10, 100, (num_hours, num_nodes))

    # Peak around hour 18 (6 pm return commute), trough around 06:00.
    diurnal = np.clip(
        [1 + np.sin((h - 12) * np.pi / 12) for h in range(num_hours)],
        0.5, 2.0,
    )
    demand = base * diurnal[:, np.newaxis]

    # Occasional fast-charging events (e.g. highway DC fast charger, ~150 kW).
    spikes = np.random.choice([0, 150], size=demand.shape, p=[0.95, 0.05])
    return demand + spikes


def save_mock(num_nodes: int = config.NUM_NODES) -> np.ndarray:
    """Generate and persist the mock demand tensor to config.MOCK_TENSOR_PATH.

    Creates the output directory if it doesn't exist.

    Args:
        num_nodes (int): Grid size (should match what the team expects).

    Returns:
        The generated array (so callers can inspect it without re-loading).
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tensor = generate_mock_demand(num_nodes=num_nodes)
    np.save(config.MOCK_TENSOR_PATH, tensor)
    print(f"[mock] shape={tensor.shape}  → {config.MOCK_TENSOR_PATH}")
    return tensor

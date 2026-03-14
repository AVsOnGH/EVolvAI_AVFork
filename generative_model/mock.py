"""
Quick mock output generator.
Produces a realistic-looking [24, N] demand tensor for async handoff to the
Grid Physics and Optimization teams while the real models train.
"""

import os
import numpy as np
from . import config


def generate_mock_demand(num_nodes=config.NUM_NODES, num_hours=config.SEQ_LEN):
    """
    Generate a mock EV demand array with a diurnal pattern and random spikes.

    Returns:
        np.ndarray of shape [num_hours, num_nodes] in kW.
    """
    base = np.random.uniform(10, 100, (num_hours, num_nodes))

    # Diurnal multiplier – peak around hour 18
    diurnal = np.clip(
        [1 + np.sin((h - 12) * np.pi / 12) for h in range(num_hours)],
        0.5, 2.0,
    )
    demand = base * diurnal[:, np.newaxis]

    # Random fast-charging spikes
    spikes = np.random.choice([0, 150], size=demand.shape, p=[0.95, 0.05])
    demand += spikes

    return demand


def save_mock(num_nodes=config.NUM_NODES):
    """Generate and save the mock tensor to config.MOCK_TENSOR_PATH."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tensor = generate_mock_demand(num_nodes=num_nodes)
    np.save(config.MOCK_TENSOR_PATH, tensor)
    print(f"[mock] shape={tensor.shape}  → {config.MOCK_TENSOR_PATH}")
    return tensor

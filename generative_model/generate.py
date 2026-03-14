"""
Counterfactual scenario generation.
Loads a trained model (or accepts one in-memory) and generates demand
tensors of shape [24, NUM_NODES] for each scenario defined in config.
"""

import os
import numpy as np
import torch

from . import config
from .models import GenerativeCounterfactualVAE


def load_model(device=None):
    """Load a trained model from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenerativeCounterfactualVAE().to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH,
                                     map_location=device))
    model.eval()
    return model, device


def generate_counterfactual(model, device, condition, num_nodes=config.NUM_NODES):
    """
    Generate a single counterfactual demand profile.

    Args:
        model: Trained GenerativeCounterfactualVAE.
        device: torch device.
        condition: list of floats, length == config.COND_DIM.
        num_nodes: number of grid nodes (used only for assertion).

    Returns:
        np.ndarray of shape [24, num_nodes] in kW.
    """
    with torch.no_grad():
        z = torch.randn(1, config.LATENT_DIM, device=device)
        cond = torch.tensor([condition], dtype=torch.float32, device=device)
        out = model.decode(z, cond)

        # Drop weather channel, permute to [seq_len, nodes]
        demand = out[:, :num_nodes, :].squeeze(0).permute(1, 0)
        assert demand.shape == (config.SEQ_LEN, num_nodes), \
            f"Shape mismatch: got {demand.shape}"
        return demand.cpu().numpy()


def generate_all_scenarios(model=None, device=None, save=True):
    """
    Run every scenario defined in config.SCENARIOS and optionally save as .npy.

    Returns:
        dict mapping scenario_name → np.ndarray [24, NUM_NODES].
    """
    if model is None:
        model, device = load_model(device)

    results = {}
    for name, spec in config.SCENARIOS.items():
        tensor = generate_counterfactual(model, device, spec["condition"])
        results[name] = tensor
        print(f"  [{name}] {spec['description']}  shape={tensor.shape}")

        if save:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            path = os.path.join(config.OUTPUT_DIR, f"{name}.npy")
            np.save(path, tensor)
            print(f"    → saved {path}")

    return results

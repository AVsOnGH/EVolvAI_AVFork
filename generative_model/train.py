"""
Training loop for the GenerativeCounterfactualVAE.
Saves the trained model checkpoint to config.MODEL_SAVE_PATH.
"""

import os
import torch
import torch.optim as optim

from . import config
from .models import GenerativeCounterfactualVAE, vae_loss_function
from .data_loader import get_dataloader


def train(epochs=config.EPOCHS, save=True):
    """Train the GCD-VAE and optionally save the checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    model = GenerativeCounterfactualVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loader = get_dataloader()

    baseline_cond = config.BASELINE_CONDITION  # [0.0, 1.0]

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            # [batch, seq_len, features] → [batch, features, seq_len]
            x = batch.permute(0, 2, 1).to(device)
            cond = torch.tensor([baseline_cond] * x.size(0),
                                dtype=torch.float32, device=device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, cond)
            loss = vae_loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader.dataset)
        print(f"  Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

    if save:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"[train] Model saved → {config.MODEL_SAVE_PATH}")

    return model, device

"""
Dataset and DataLoader for simulated EV charging demand + weather data.
Replace the simulation logic here with real CSV / Parquet loaders when
Caltech ACN-Data or NREL datasets are available.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import config


class EVDemandDataset(Dataset):
    """Simulated daily EV charging + weather profiles."""

    def __init__(self,
                 num_samples=config.NUM_SAMPLES,
                 num_nodes=config.NUM_NODES,
                 seq_len=config.SEQ_LEN):
        self.num_samples = num_samples

        # ── Simulated historical EV charging data ────────────────────────
        base = np.random.uniform(10, 100, (num_samples, seq_len, num_nodes))
        diurnal = np.clip(
            [1 + np.sin((h - 12) * np.pi / 12) for h in range(seq_len)],
            0.5, 2.0,
        ).reshape(1, seq_len, 1)
        charge = (base * diurnal).astype(np.float32)

        # ── Simulated weather (temperature °C) ──────────────────────────
        weather = np.random.uniform(-10, 40,
                                    (num_samples, seq_len, 1)).astype(np.float32)

        # ── Normalize ────────────────────────────────────────────────────
        charge  = (charge  - charge.mean())  / (charge.std()  + 1e-5)
        weather = (weather - weather.mean()) / (weather.std() + 1e-5)

        # Concatenate along feature axis → [samples, seq_len, num_nodes+1]
        self.data = np.concatenate([charge, weather], axis=-1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Returns [seq_len, features]; permuted to [features, seq_len] in train loop
        return torch.from_numpy(self.data[idx])


def get_dataloader(batch_size=config.BATCH_SIZE, num_nodes=config.NUM_NODES):
    """Convenience wrapper that returns a shuffled DataLoader."""
    return DataLoader(EVDemandDataset(num_nodes=num_nodes),
                      batch_size=batch_size, shuffle=True)

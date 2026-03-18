"""
generative_core/data_loader.py
===============================
Dataset and DataLoader for EV charging demand + weather time-series.

The loader has two operating modes that switch automatically:

  1. **Real data** (preferred): reads a parquet file at config.DATA_PATH.
     The file must contain the columns:
       date (str/date), hour (int, 0–23), node_id (str), demand_kw (float)
     It is pivoted so rows = (date, hour) and columns = node_id, then reshaped
     to [num_days, SEQ_LEN, NUM_NODES].

  2. **Synthetic fallback**: if the parquet doesn't exist or fails validation,
     generates a diurnal sinusoidal pattern with uniform random noise.
     This unblocks model development while waiting for real data.

In either case, three synthetic weather channels (temperature, precipitation,
wind) are appended.  Replace with real hourly weather CSVs via the
`data_pipeline/preprocess.py` script once the data is sourced.

All arrays are Z-score normalised before being concatenated, which ensures
that the weather and demand channels live on the same scale.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import config

logger = logging.getLogger(__name__)


class EVDemandDataset(Dataset):
    """PyTorch Dataset of 24-hour EV demand + weather profiles.

    Each sample is a single daily profile of shape [SEQ_LEN, NUM_FEATURES]
    where NUM_FEATURES = NUM_NODES + NUM_WEATHER_FEATURES.

    The time axis is last here to match NumPy convention; the DataLoader
    consumer (train.py) permutes to [features, seq_len] for Conv1d.

    Args:
        num_samples (int): How many days to synthesise when no parquet exists.
        num_nodes (int): Expected number of spatial grid nodes.
        seq_len (int): Expected hours per daily profile.
    """

    def __init__(self,
                 num_samples: int = config.NUM_SAMPLES,
                 num_nodes: int = config.NUM_NODES,
                 seq_len: int = config.SEQ_LEN):

        charge = self._try_load_real(num_nodes, seq_len)
        if charge is None:
            charge = self._generate_synthetic(num_samples, num_nodes, seq_len)

        actual_samples = charge.shape[0]

        # Append weather channels.  Once real weather data is available, replace
        # this block with a merger against the real parquet weather columns.
        weather = np.random.uniform(
            -10, 40,
            (actual_samples, seq_len, config.NUM_WEATHER_FEATURES),
        ).astype(np.float32)

        # Z-score normalise independently to put all channels on the same scale.
        charge  = self._normalize(charge)
        weather = self._normalize(weather)

        # Final shape: [samples, seq_len, num_nodes + NUM_WEATHER_FEATURES]
        self.data = np.concatenate([charge, weather], axis=-1)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Z-score normalise an array, safe against zero-variance inputs.

        Args:
            arr: Any float32 NumPy array.

        Returns:
            Normalised copy of arr.  If std < 1e-8, only the mean is subtracted.
        """
        std = arr.std()
        if std < 1e-8:
            return arr - arr.mean()
        return (arr - arr.mean()) / (std + 1e-8)

    @staticmethod
    def _try_load_real(num_nodes: int, seq_len: int) -> Optional[np.ndarray]:
        """Attempt to load and validate the real parquet dataset.

        Checks:
          * File exists at config.DATA_PATH.
          * Required columns are present.
          * Node count after pivoting matches num_nodes.

        Args:
            num_nodes: Expected column count after pivoting on node_id.
            seq_len:   Expected row-per-group count (hours in a day).

        Returns:
            float32 array [num_days, seq_len, num_nodes], or None on failure.
        """
        import os
        if not os.path.isfile(config.DATA_PATH):
            logger.info("Parquet not found at %s – using synthetic data.", config.DATA_PATH)
            return None

        try:
            import pandas as pd
            df = pd.read_parquet(config.DATA_PATH)

            required = {"date", "hour", "node_id", "demand_kw"}
            missing = required - set(df.columns)
            if missing:
                logger.warning("Parquet missing columns %s – falling back to synthetic.", missing)
                return None

            pivot = df.pivot_table(
                index=["date", "hour"], columns="node_id", values="demand_kw",
            )
            if pivot.shape[1] != num_nodes:
                logger.warning(
                    "Parquet has %d nodes but config.NUM_NODES=%d – falling back.",
                    pivot.shape[1], num_nodes,
                )
                return None

            charge = pivot.values.reshape(-1, seq_len, num_nodes).astype(np.float32)
            logger.info("Loaded real data: %d samples from %s.", charge.shape[0], config.DATA_PATH)
            return charge

        except Exception as exc:        # parquet read / reshape can raise many things
            logger.warning("Parquet load failed (%s) – falling back to synthetic.", exc)
            return None

    @staticmethod
    def _generate_synthetic(num_samples: int, num_nodes: int, seq_len: int) -> np.ndarray:
        """Generate a diurnal synthetic charging dataset.

        The base is uniform random kW per node, modulated by a sine wave that
        peaks around hour 18 (evening commute return).  This is realistic enough
        for architecture debugging and async team handoff.

        Args:
            num_samples: Number of daily profiles to produce.
            num_nodes:   Spatial grid size.
            seq_len:     Hours per profile (normally 24).

        Returns:
            float32 array [num_samples, seq_len, num_nodes] in kW.
        """
        base = np.random.uniform(10, 100, (num_samples, seq_len, num_nodes))
        # Diurnal multiplier: 0.5 at midnight, ~2.0 at hour 18 (evening peak)
        diurnal = np.clip(
            [1 + np.sin((h - 12) * np.pi / 12) for h in range(seq_len)],
            0.5, 2.0,
        ).reshape(1, seq_len, 1)
        return (base * diurnal).astype(np.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        # Use the actual array length, not the constructor argument, so that
        # real parquet datasets with fewer rows don't cause an index error.
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the profile at index idx as a float32 tensor.

        Returns:
            Tensor of shape [seq_len, num_features].
            train.py permutes this to [num_features, seq_len] for Conv1d.
        """
        return torch.from_numpy(self.data[idx])


def get_dataloader(batch_size: int = config.BATCH_SIZE,
                   num_nodes: int = config.NUM_NODES) -> DataLoader:
    """Build and return a shuffled training DataLoader.

    Args:
        batch_size: Mini-batch size (default from config).
        num_nodes:  Grid size used to validate real parquet node count.

    Returns:
        DataLoader yielding [batch, seq_len, num_features] float32 tensors.
    """
    dataset = EVDemandDataset(num_nodes=num_nodes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

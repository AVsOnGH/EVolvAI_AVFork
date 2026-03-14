import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EVDemandDataset(Dataset):
    """
    Simulated dataset for EV charging demand and weather data.
    """
    def __init__(self, num_samples=1000, num_nodes=50, sequence_length=24):
        """
        Args:
            num_samples (int): Total number of days/samples.
            num_nodes (int): Number of nodes in the grid.
            sequence_length (int): Length of the sequence (e.g., 24 hours).
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.seq_len = sequence_length
        
        # Simulate historical EV charging data: Shape [samples, seq_len, nodes]
        # We assume base load is between 10-100 kW, plus a diurnal multiplier.
        base_demand = np.random.uniform(10, 100, (num_samples, sequence_length, num_nodes))
        diurnal = np.array([1 + np.sin((h - 12) * np.pi / 12) for h in range(sequence_length)])
        diurnal = np.clip(diurnal, 0.5, 2.0).reshape(1, sequence_length, 1)
        self.charge_data = (base_demand * diurnal).astype(np.float32)
        
        # Simulate weather data (Temperature in Celsius): Shape [samples, seq_len, 1]
        self.weather_data = np.random.uniform(-10, 40, (num_samples, sequence_length, 1)).astype(np.float32)
        
        # Normalize Data (Mock Normalization)
        self.charge_data = (self.charge_data - np.mean(self.charge_data)) / (np.std(self.charge_data) + 1e-5)
        self.weather_data = (self.weather_data - np.mean(self.weather_data)) / (np.std(self.weather_data) + 1e-5)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Concatenate charging data and weather data. 
        # TCN expects input of shape [Channels/Features, Sequence_Length] when using PyTorch Conv1d
        # We prepare it as [Sequence_Length, Features] and permute it in the training loop
        
        charge = torch.tensor(self.charge_data[idx])    # [seq_len, nodes]
        weather = torch.tensor(self.weather_data[idx])  # [seq_len, 1]
        
        # Features = nodes + 1 (weather)
        combined = torch.cat([charge, weather], dim=-1) # [seq_len, nodes + 1]
        
        return combined

def get_dataloader(batch_size=32, num_nodes=50):
    """Returns a DataLoader for the simulated dataset."""
    dataset = EVDemandDataset(num_nodes=num_nodes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    loader = get_dataloader(batch_size=8)
    for batch in loader:
        print(f"Batch shape (Batch Size, Sequence Length, Features): {batch.shape}")
        print("Data loading successful.")
        break

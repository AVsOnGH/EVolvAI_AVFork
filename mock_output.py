import numpy as np

def generate_mock_demand(num_nodes=100, num_hours=24):
    """
    Generates a mock 2D tensor of EV charging demand.
    
    Args:
        num_nodes (int): The number of nodes in the grid topology.
        num_hours (int): The number of hours in the profile (typically 24).
        
    Returns:
        np.ndarray: A 2D array of shape [num_hours, num_nodes] representing kW demand.
    """
    # Generate random demand in kW, roughly between 10kW and 250kW per node
    # Adding a diurnal pattern for realism (higher demand in the evening)
    base_demand = np.random.uniform(low=10, high=100, size=(num_hours, num_nodes))
    
    # Simple diurnal multiplier (peak around hour 18)
    time_of_day_multiplier = np.array([1 + np.sin((h - 12) * np.pi / 12) for h in range(num_hours)])
    time_of_day_multiplier = np.clip(time_of_day_multiplier, 0.5, 2.0)
    
    # Broadcast and multiply
    demand_kw = base_demand * time_of_day_multiplier[:, np.newaxis]
    
    # Add some random spikes to represent fast charging
    spikes = np.random.choice([0, 150], size=(num_hours, num_nodes), p=[0.95, 0.05])
    demand_kw += spikes
    
    return demand_kw

if __name__ == "__main__":
    print("Generating Mock Output for Async Handoff...")
    N_NODES = 50
    mock_tensor = generate_mock_demand(num_nodes=N_NODES, num_hours=24)
    print(f"Generated mock tensor of shape: {mock_tensor.shape}")
    print(f"Sample data (first 5 hours for node 0):")
    print(mock_tensor[:5, 0])
    print("\nMock output is ready for integration manually. Save this as a numpy array or pass it directly.")
    
    # Example to save for teammates
    np.save('mock_demand_tensor.npy', mock_tensor)
    print("Saved to mock_demand_tensor.npy")

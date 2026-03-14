import torch
import torch.optim as optim
from models import GenerativeCounterfactualVAE, vae_loss_function
from data_loader import get_dataloader

def train_model():
    print("--- Starting Training Process ---")
    num_nodes = 50
    # +1 for the weather feature
    num_features = num_nodes + 1
    seq_len = 24
    cond_dim = 2 # e.g., [Extreme_Weather_Flag, Electrification_Multiplier]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GenerativeCounterfactualVAE(num_features=num_features, seq_len=seq_len, cond_dim=cond_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataloader = get_dataloader(batch_size=32, num_nodes=num_nodes)
    
    epochs = 2 # Just a small number for demonstration
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            # Data from loader is [batch, seq_len, features]. 
            # Conv1d expects [batch, channels(features), seq_len]
            data = data.permute(0, 2, 1).to(device)
            
            # Dummy condition vectors for training base behavior (e.g., standard conditions)
            # [0, 1] -> [No extreme weather, 1.0x electrification]
            condition = torch.tensor([[0.0, 1.0]] * data.size(0)).to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, condition)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch: {epoch+1} Average loss: {train_loss / len(dataloader.dataset):.4f}")
        
    print("Training Complete.\n")
    return model, device

def apply_interventions(model, device, num_nodes=50):
    print("--- Applying Interventions (Counterfactual Generation) ---")
    model.eval()
    
    # We don't need real inputs to generate, we can sample from the prior Z ~ N(0, I)
    # However, to condition on a *specific* historical day, we would encode it first.
    # Here, we generate a novel counterfactual day from scratch using random noise.
    
    batch_size = 1
    latent_dim = 16
    
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim).to(device)
        
        # Scenario 1: Extreme Winter Storm + 2.5x Fleet Electrification Surge
        condition_vector = torch.tensor([[1.0, 2.5]]).to(device) # [WeatherFlag, EV_Multiplier]
        
        counterfactual_output = model.decode(z, condition_vector)
        
        # Output is [batch, features, seq_len]
        # We drop the weather feature (the last one) to get just the node demand
        demand_output = counterfactual_output[:, :-1, :] # [batch, nodes, seq_len]
        
        # Squeeze batch dim and permute to [seq_len, nodes] as requested by team
        final_tensor = demand_output.squeeze(0).permute(1, 0)
        
        print(f"Condition Vector Applied: {condition_vector.cpu().numpy()}")
        print(f"Final Output Tensor Shape (24 hours, N_Nodes): {final_tensor.shape}")
        
        # Sanity check
        assert final_tensor.shape == (24, num_nodes), "Output shape mismatch!"
        print("Success: Generated counterfactual accurately matches required topological shape.")
        return final_tensor

if __name__ == "__main__":
    trained_model, compute_device = train_model()
    cf_tensor = apply_interventions(trained_model, compute_device)

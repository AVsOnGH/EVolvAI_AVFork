import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    """
    A causal 1D convolution to ensure temporal ordering (future doesn't leak into past).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, 
                                 self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GenerativeCounterfactualVAE(nn.Module):
    def __init__(self, num_features, seq_len=24, latent_dim=16, cond_dim=2):
        """
        TCN-VAE with continuous latent variables and conditional inputs.
        Args:
            num_features: Input features per timestep (nodes + exogenous)
            seq_len: Length of sequence (e.g., 24 hours)
            latent_dim: Size of the latent space Z
            cond_dim: Size of the condition vector C (intervention triggers)
        """
        super(GenerativeCounterfactualVAE, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Encoder: TCN to extract temporal features
        tcn_channels = [32, 64]
        self.encoder_tcn = TemporalConvNet(num_inputs=num_features, num_channels=tcn_channels)
        
        # Flattened sequence * final tcn channel size
        tcn_out_dim = seq_len * tcn_channels[-1] 
        
        self.fc_mu = nn.Linear(tcn_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(tcn_out_dim, latent_dim)

        # Decoder: Dense layers mapping from (Z + C) back to sequence
        # Z: Latent variable, C: Causal condition variable
        dec_in_dim = latent_dim + cond_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(dec_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, tcn_out_dim),
            nn.ReLU()
        )
        
        # TCN to smooth out the decoder reconstruction
        self.decoder_tcn = TemporalConvNet(num_inputs=tcn_channels[-1], num_channels=[32, num_features])

    def encode(self, x):
        # x is [batch, features, seq_len]
        h = self.encoder_tcn(x)
        # Flatten
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        # Concatenate latent variable Z with condition C
        # z: [batch, latent_dim]
        # condition: [batch, cond_dim]
        zc = torch.cat([z, condition], dim=-1)
        h = self.decoder_fc(zc)
        
        # Reshape to [batch, channels, seq_len]
        h = h.view(h.size(0), 64, self.seq_len) # 64 was the last channel size in encoder TCN
        
        # Final TCN to generate the output sequence and features
        recon_x = self.decoder_tcn(h)
        return recon_x

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Reconstruction + KL divergence losses summed over all elements and batch.
    """
    # MSE Loss for reconstruction
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

if __name__ == "__main__":
    # Smoke test
    batch_size = 4
    seq_len = 24
    nodes = 50
    features = nodes + 1 # 1 for weather
    
    # [batch, features, seq_len]
    dummy_x = torch.randn(batch_size, features, seq_len)
    # [batch, cond_dim] - e.g., (Extreme Weather Flag, Electrification Multiplier)
    dummy_cond = torch.randn(batch_size, 2)
    
    model = GenerativeCounterfactualVAE(num_features=features, seq_len=seq_len)
    
    recon, mu, logvar = model(dummy_x, dummy_cond)
    print(f"Input shape: {dummy_x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent Mu shape: {mu.shape}")
    print("Model forward pass successful. Output tensor shape meets [batch, features, 24] requirement.")

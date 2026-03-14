"""
PyTorch model definitions for the Generative Counterfactual Framework.
Contains: CausalConv1d, TCNBlock, TemporalConvNet, GenerativeCounterfactualVAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution – the future never leaks into the past."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        self._causal_padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=self._causal_padding,
                         dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out = super().forward(x)
        if self._causal_padding != 0:
            return out[:, :, :-self._causal_padding]
        return out


class TCNBlock(nn.Module):
    """Residual block with two causal convolutions and optional down-sample."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 dropout=config.DROPOUT):
        super().__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size,
                                  stride=stride, dilation=dilation)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size,
                                  stride=stride, dilation=dilation)
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(), nn.Dropout(dropout),
            self.conv2, nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TCN blocks with exponentially increasing dilation."""

    def __init__(self, num_inputs, num_channels,
                 kernel_size=config.KERNEL_SIZE, dropout=config.DROPOUT):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size,
                                   stride=1, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GenerativeCounterfactualVAE(nn.Module):
    """
    TCN-VAE with latent conditioning for counterfactual demand generation.

    Input shape  : [batch, num_features, seq_len]
    Output shape : [batch, num_features, seq_len]
    Condition    : [batch, cond_dim]
    """

    def __init__(self,
                 num_features=config.NUM_FEATURES,
                 seq_len=config.SEQ_LEN,
                 latent_dim=config.LATENT_DIM,
                 cond_dim=config.COND_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        tcn_ch = config.TCN_CHANNELS

        # Encoder
        self.encoder_tcn = TemporalConvNet(num_features, tcn_ch)
        tcn_flat = seq_len * tcn_ch[-1]
        self.fc_mu     = nn.Linear(tcn_flat, latent_dim)
        self.fc_logvar = nn.Linear(tcn_flat, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, config.DECODER_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.DECODER_HIDDEN, tcn_flat),
            nn.ReLU(),
        )
        self.decoder_tcn = TemporalConvNet(tcn_ch[-1], [32, num_features])

    # ── core methods ──────────────────────────────────────────────────────

    def encode(self, x):
        h = self.encoder_tcn(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, condition):
        h = self.decoder_fc(torch.cat([z, condition], dim=-1))
        h = h.view(h.size(0), config.TCN_CHANNELS[-1], self.seq_len)
        return self.decoder_tcn(h)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """Reconstruction (MSE) + KL divergence."""
    recon = F.mse_loss(recon_x, x, reduction="sum")
    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + config.KLD_WEIGHT * kld

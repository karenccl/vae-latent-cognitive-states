"""Windowed GRU-VAE for [correct, standardised log-RT] windows."""
import torch
import torch.nn as nn


class WindowVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=4):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, h = self.encoder(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, window_size):
        h0 = torch.tanh(self.decoder_input(z)).unsqueeze(0)
        repeated = z.new_zeros(z.size(0), window_size, h0.size(-1))
        output, _ = self.decoder(repeated, h0)
        return self.output_layer(output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.size(1)), mu, logvar, z


def vae_loss(x, x_hat, mu, logvar, beta=1.0, recon="mean"):
    """recon='mean': MSE averaged over all elements (the project spec).
    recon='sum': squared error summed over the window, averaged over the batch
    (a Gaussian log-likelihood with unit variance; sensitivity analysis)."""
    if recon == "mean":
        reconstruction = nn.functional.mse_loss(x_hat, x, reduction="mean")
    else:
        reconstruction = ((x_hat - x) ** 2).sum(dim=(1, 2)).mean()
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return reconstruction + beta * kl, reconstruction, kl

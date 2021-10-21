from gen import utils, ae
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Auto Encoder

    Largely based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    with some random alterations for the layers.
    """

    def __init__(
        self,
        n_in: int,
        n_h: int = 50,
        n_z: int = 10,
        p: float = 0.01,
        mean_encoder_model: nn.Module = None,
        logvar_encoder_model: nn.Module = None,
        decoder_model: nn.Module = None,
    ):
        super().__init__()
        utils.logger.info(f"n_in {n_in}, n_h {n_h}, n_z {n_z}")

        self.z_mean = (
            ae.get_encoder_model(n_in, n_h, p, n_z)
            if mean_encoder_model is None
            else mean_encoder_model
        )
        self.z_logvar = (
            ae.get_encoder_model(n_in, n_h, p, n_z)
            if logvar_encoder_model is None
            else logvar_encoder_model
        )
        self.decoder = (
            ae.get_decoder_model(n_in, n_h, p, n_z)
            if decoder_model is None
            else decoder_model
        )

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x_cat, x_cont):
        μ, logvar = self.get_distribution_statistics(x_cont)
        z = self.reparameterize(μ, logvar)
        return z, μ, logvar

    def forward(self, x_cat, x_cont):
        z, μ, logvar = self.encode(x_cat, x_cont)
        return self.decode(z), x_cont, μ, logvar

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.n_z)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate_reconstruction(
        self, x_cat: Tensor, x_cont: Tensor, *args, **kwargs
    ) -> Tensor:
        return self.forward(x_cat, x_cont)[0]

    def get_distribution_statistics(self, x):
        return self.z_mean(x), self.z_logvar(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class VAE_Loss:
    """
    https://arxiv.org/pdf/1312.6114.pdf -> eq. 10
    """

    def __init__(self, kld_weight: float = 1.0):
        self.kld_weight = kld_weight  # importance of the Kullback-Leibler Loss - may be the batchsize or something else

    def loss(self, reconstructions, originals, mu, logvar) -> dict:
        reconstruction_loss = F.mse_loss(reconstructions, originals)
        # 1/2 sum_j(1 + log(σ_j^2) - μ_j^2 - σ_j^2) + 1/L sum_l log p_θ(x|z_l)
        kld_loss = -torch.mean(
            0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp() ** 2, dim=1), dim=0
        )

        loss = reconstruction_loss + self.kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": reconstruction_loss,
            "KLD": kld_loss,
            "kld_weight": self.kld_weight,
        }

    def __call__(self, X, y):
        reconstructions, originals, mu, log_var = X
        return self.loss(reconstructions, originals, mu, log_var)["loss"]

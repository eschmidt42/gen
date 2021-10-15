from gen import utils
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class AE(nn.Module):
    """Auto Encoder

    Completely from thin air.
    """

    def __init__(self, n_in: int, n_h: int = 50, n_z: int = 10, p: float = 0.01):
        super().__init__()
        utils.logger.info(f"n_in {n_in}, n_h {n_h}, n_z {n_z}")
        self.in_layer = nn.Linear(n_in, n_h)

        self.encoder = nn.Sequential(
            self.in_layer,
            nn.BatchNorm1d(n_h),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(n_h, n_z),
        )  #  self.mean_layer

        self.decoder = nn.Sequential(
            nn.Linear(n_z, n_h),
            nn.BatchNorm1d(n_h),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(n_h, n_in),
            nn.BatchNorm1d(n_in),
        )

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x_cat, x_cont):
        return self.encoder(x_cont)

    def forward(self, x_cat, x_cont):
        z = self.encode(x_cat, x_cont)
        return self.decode(z), x_cont

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.n_z)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate_reconstruction(
        self, x_cat: Tensor, x_cont: Tensor, *args, **kwargs
    ) -> Tensor:
        return self.forward(x_cat, x_cont)[0]


class AE_Loss:
    def __init__(self):
        pass

    def loss(self, reconstructions, originals) -> dict:
        reconstruction_loss = F.mse_loss(reconstructions, originals)
        loss = reconstruction_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": reconstruction_loss,
        }

    def __call__(self, X, y):
        reconstructions, originals = X
        return self.loss(reconstructions, originals)["loss"]

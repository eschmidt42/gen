"""
# https://github.com/VincentStimper/normalizing-flows
# https://github.com/acids-ircam/pytorch_flows/blob/master/flows_01.ipynb
# https://github.com/LukasRinder/normalizing-flows/blob/master/normalizingflows/flow_catalog.py
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        :param z: input variable, first dimension is batch dim
        :return: transformed z and log of absolute determinant
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")


class Planar(Flow):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """

    def __init__(self, shape, act="tanh", u=None, w=None, b=None):
        """
        Constructor of the planar flow
        :param shape: shape of the latent variable z
        :param h: nonlinear function h of the planar flow (see definition of f above)
        :param u,w,b: optional initialization for parameters
        """
        super().__init__()
        lim_w = np.sqrt(2.0 / np.prod(shape))
        lim_u = np.sqrt(2)

        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))

        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError("Nonlinearity is not implemented.")

    def forward(self, z):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        if self.act == "tanh":
            inner = torch.sum(self.w * self.u)
            u = self.u + (
                torch.log(1 + torch.exp(inner)) - 1 - inner
            ) * self.w / torch.sum(self.w ** 2)
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            inner = torch.sum(self.w * self.u)
            u = self.u + (
                torch.log(1 + torch.exp(inner)) - 1 - inner
            ) * self.w / torch.sum(
                self.w ** 2
            )  # constraint w.T * u neq -1, use >
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        z_ = z + u * self.h(lin.unsqueeze(1))
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin)))
        return z_, log_det

    def inverse(self, z):
        if self.act != "leaky_relu":
            raise NotImplementedError("This flow has no algebraic inverse.")
        lin = torch.sum(self.w * z, list(range(2, self.w.dim())), keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        a = ((lin + self.b) / (1 + inner) < 0) * (
            self.h.negative_slope - 1.0
        ) + 1.0  # absorb leakyReLU slope into u
        u = a * (
            self.u
            + (torch.log(1 + torch.exp(inner)) - 1 - inner)
            * self.w
            / torch.sum(self.w ** 2)
        )
        z_ = z - 1 / (1 + inner) * (lin + u * self.b)
        log_det = -torch.log(torch.abs(1 + torch.sum(self.w * u)))
        if log_det.dim() == 0:
            log_det = log_det.unsqueeze(0)
        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(1)
        return z_, log_det


class NormalizingFlow(nn.Module):
    """
    VAE using normalizing flows to express approximate distribution
    """

    def __init__(self, prior, encoder=None, flows=None, decoder=None):

        super().__init__()

        self.encoder = encoder
        self.flows = nn.ModuleList(flows)
        self.prior = prior
        self.decoder = decoder

    def forward(self, x_cat, x_cont, num_samples=1):

        z, log_q = self.encoder(x_cont, num_samples=num_samples)
        # print(f'{log_q=}')
        # Flatten batch and sample dim
        # z = z.view(-1, *z.size()[2:])

        # log_q = log_q.view(-1, *log_q.size()[2:])
        # log_q = log_q.view(-1, log_q.size()[0])
        # print(f'{log_q=}')

        for flow in self.flows:
            z, log_det = flow(z)
            # print(f'{log_det=}')
            log_q -= log_det  # .sum()
        # print(f'{z=}')
        log_p = self.prior.log_prob(z)

        if self.decoder is not None:
            log_p += self.decoder.log_prob(x_cont, z)

        r, _ = self.decoder(z)
        # Separate batch and sample dimension again
        # z = z.view(-1, num_samples, *z.size()[1:])
        # log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        # log_p = log_p.view(-1, num_samples, *log_p.size()[1:])
        # z = reconstructed x?
        return r, x_cont, log_q, log_p


# gaussian encoder
class NNDiagGaussian(nn.Module):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """

    def __init__(
        self,
        mean_encoder_model: nn.Module = None,
        logvar_encoder_model: nn.Module = None,
    ):
        super().__init__()
        self.z_mean = mean_encoder_model
        self.z_logvar = logvar_encoder_model

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, eps

    def forward(self, x, num_samples=1):

        # batch_size = len(x)
        # mean_std = self.net(x)
        # n_hidden = mean_std.size()[1] // 2
        # mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        # std = torch.exp(0.5 * mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1))
        # eps = torch.randn((batch_size, num_samples) + tuple(mean.size()[2:]), device=x.device)
        # z = mean + std * eps
        # print(f'x {x.shape}')
        mu = self.z_mean(x)
        logvar = self.z_logvar(x)
        z, eps = self.reparameterize(mu, logvar)
        # print(f'zsize {z.size()} zdim {z.dim()}')
        # TODO: ensure log_p makes sense
        # log_p = - .5 * np.log(2*np.pi) - torch.sum(logvar + .5 * eps*eps)
        # log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi)\
        #         - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, z.dim())))
        # print(f'{mu=}, \n{logvar=}')
        std = torch.sqrt(logvar.exp())
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(
            2 * np.pi
        ) - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), dim=1)
        return z, log_p

    def log_prob(self, z, x):

        # if z.dim() == 1:
        #     z = z.unsqueeze(0)
        # if z.dim() == 2:
        #     z = z.unsqueeze(0)
        # print(f'{z=}, \n{x=}')
        mu = self.z_mean(x)
        logvar = self.z_logvar(x)
        # print(f'{mu=}, \n{logvar=}')
        # n_hidden = mean_std.size()[1] // 2
        # mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        # var = torch.exp(mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1))
        # log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi)\
        #         - 0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)
        log_p = -0.5 * np.log(2 * np.pi) - 0.5 * torch.sum(
            logvar + (z - mu) ** 2 / logvar.exp(), dim=1
        )
        return log_p


class Flow_Loss:
    def __init__(self):
        pass

    def loss(self, log_q, log_p):
        _q = log_q.mean()
        _p = log_p.mean()
        out = {"loss": _q - _p, "q": _q, "p": _p}
        return out

    def __call__(self, X, y):
        reconstructions, originals, log_q, log_p = X
        return self.loss(log_q, log_p)["loss"]

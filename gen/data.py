from gen import utils
import torch
import pandas as pd
from sklearn import datasets
import numpy as np


class DataGenerator:
    @classmethod
    def generate(self, method: str, num_points: int):
        f = getattr(self, method, None)
        assert f is not None, f'Unexpectedly got "{method}" for paramter `method`.'
        return getattr(self, method, None)(num_points)

    @classmethod
    def checkerboard(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a checkerboard pattern"
        )
        x1 = torch.rand(num_points) * 4 - 2
        x2_ = torch.rand(num_points) - torch.randint(0, 2, [num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        data = torch.stack([x1, x2]).t() * 2
        return pd.DataFrame(data, columns=[f"x_{i}" for i in [0, 1]])

    @classmethod
    def gaussian(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a gaussian pattern"
        )
        x1 = torch.randn(num_points)
        x2 = 0.5 * torch.randn(num_points)
        return pd.DataFrame(
            torch.stack((x1, x2)).t(), columns=[f"x_{i}" for i in [0, 1]]
        )

    @classmethod
    def crescent(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a crescent pattern"
        )
        x1 = torch.randn(num_points)
        x2_mean = 0.5 * x1 ** 2 - 1
        x2_var = torch.exp(torch.Tensor([-2]))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(num_points)
        return pd.DataFrame(
            torch.stack((x2, x1)).t(), columns=[f"x_{i}" for i in [0, 1]]
        )

    @classmethod
    def crescentcube(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a crescentcube pattern"
        )
        x1 = torch.randn(num_points)
        x2_mean = 0.2 * x1 ** 3
        x2_var = torch.ones(x1.shape)
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(num_points)
        return pd.DataFrame(
            torch.stack((x2, x1)).t(), columns=[f"x_{i}" for i in [0, 1]]
        )

    @classmethod
    def sinewave(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a sinewave pattern"
        )
        x1 = torch.randn(num_points)
        x2_mean = torch.sin(5 * x1)
        x2_var = torch.exp(-2 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(num_points)
        return pd.DataFrame(
            torch.stack((x1, x2)).t(), columns=[f"x_{i}" for i in [0, 1]]
        )

    @classmethod
    def abs(self, num_points: int):
        utils.logger.info(f"Generating {num_points} data points using an abs pattern")
        x1 = torch.randn(self.num_points)
        x2_mean = torch.abs(x1) - 1.0
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(num_points)
        return pd.DataFrame(
            torch.stack((x1, x2)).t(), columns=[f"x_{i}" for i in [0, 1]]
        )

    @classmethod
    def sign(self, num_points: int):
        utils.logger.info(f"Generating {num_points} data points using a sign pattern")
        x1 = torch.randn(num_points)
        x2_mean = torch.sign(x1) + x1
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(num_points)
        return pd.DataFrame(
            torch.stack((x1, x2)).t(), columns=[f"x_{i}" for i in [0, 1]]
        )

    @classmethod
    def twomoons(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a twomoons pattern"
        )
        data = datasets.make_moons(n_samples=num_points, noise=0.1, random_state=0)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return pd.DataFrame(data, columns=[f"x_{i}" for i in [0, 1]])

    @classmethod
    def twospirals(self, num_points: int):
        utils.logger.info(
            f"Generating {num_points} data points using a twospirals pattern"
        )
        n = torch.sqrt(torch.rand(num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        x = x / 3 + torch.randn_like(x) * 0.1
        return pd.DataFrame(x, columns=[f"x_{i}" for i in [0, 1]])

"""Provides the inverted pendulum SDE."""
import dataclasses
import torch
from .controlled_sde import ControlledSDE


@dataclasses.dataclass
class PendulumData:
    "Parameters of the pendulum."
    gravity: float
    pendulum_length: float
    ball_mass: float
    friction: float
    maximum_torque: float


default_pendulum_data = PendulumData(
    gravity=9.81,
    pendulum_length=0.5,
    ball_mass=0.15,
    friction=0.1,
    maximum_torque=6.0
)


class PendulumDrift(torch.nn.Module):
    """
    Drift module for the inverted pendulum.
    """

    def __init__(self, pendulum_data: PendulumData):
        super().__init__()
        # precompute the constant coefficients so that the SDE can be written
        # as d phi_t = (a1 sin(theta_t) + a2 u - a3 phit) dt + diffusion
        self.a1 = pendulum_data.gravity / pendulum_data.pendulum_length
        denom = pendulum_data.ball_mass * (pendulum_data.pendulum_length ** 2)
        self.a2 = pendulum_data.maximum_torque / denom
        self.a3 = pendulum_data.friction / denom

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """forward function of the inverted pendulum drift module"""
        # use indexing instead of torch.split so auto_LiRPA's IBP can trace
        # through without hitting the inhomogeneous-eps issue in slice_concat
        phi   = x[:, 0:1]
        theta = x[:, 1:2]
        f_phi   = self.a1 * torch.sin(theta) + self.a2 * u - self.a3 * phi
        f_theta = phi
        return torch.cat([f_phi, f_theta], dim=1)


class PendulumDiffusion(torch.nn.Module):
    """
    Diffusion module for the inverted pendulum.
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor, _u: torch.Tensor):
        """forward function of the inverted pendulum diffusion module"""
        # constant diffusion: sigma on velocity, zero on angle
        g_phi   = torch.full((x.shape[0], 1), self.sigma, device=x.device)
        g_theta = torch.zeros_like(g_phi)
        return torch.cat([g_phi, g_theta], dim=1)


class InvertedPendulum(ControlledSDE):
    """
    Stochastic inverted pendulum.

    The control signal is normalized (i.e. in [-1,1]) and multiplied by
    the maximum torque afterwards.
    """

    def __init__(self, policy: torch.nn.Module,
                 pendulum_data: PendulumData = default_pendulum_data,
                 volatility_scale: float = 2.0):
        # initialize the drift and diffusion modules
        drift = PendulumDrift(pendulum_data)
        diffusion = PendulumDiffusion(volatility_scale)
        # construct the SDE
        super().__init__(policy, drift, diffusion, "diagonal", "ito")

    def n_dimensions(self) -> int:
        return 2

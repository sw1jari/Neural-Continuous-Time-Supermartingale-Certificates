"""Hardcoded interval-Hessian generator bounds, no auto_LiRPA.

GBM: linear drift (f = Ax), state-dependent diagonal diffusion (g = sigma*x).
Pendulum: nonlinear drift involving sin and a neural policy; constant diagonal
diffusion (g = [sigma, 0]).  Policy bounds come from a manual forward IBP pass
through the TanhPolicy Sequential.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', '..'))

from mlp_hessian_ibp import (
    MLPHessianIBP,
    diagonal_diffusion_bounds,
    drift_dot_grad_bounds,
    linear_drift_dot_grad_bounds,
    make_generator_bound_fn,
)
from mlp_hessian_ibp.interval import sin_bounds as _sin_bounds

# GBM drift after applying the policy u = -x:
#   f1 = -0.5*x1 + x2 + (-x1) = -1.5*x1 + x2
#   f2 = -x1 - 0.5*x2 + (-x2) = -x1 - 1.5*x2
# i.e. f = A x, A = [[-1.5, 1], [-1, -1.5]]
_GBM_DRIFT = torch.tensor([[-1.5, 1.0], [-1.0, -1.5]])

# GBM diffusion: g(x) = 0.2 * x (diagonal, per axis)
_GBM_SIGMA = 0.2


class GBMHardcodedIBPVerifier:
    """Interval upper bound on LV for the GBM problem, fully hand-written."""

    def __init__(self, net: nn.Module):
        self.certifier = MLPHessianIBP(net)
        self._drift = _GBM_DRIFT

    def generator_upper_bound(
        self,
        x_L: torch.Tensor,
        x_U: torch.Tensor,
    ) -> torch.Tensor:
        """Return an interval upper bound on LV over each box [x_L, x_U].

        Input shapes are [batch, 2]; output shape is [batch, 1].
        """
        drift = self._drift.to(x_L.device, dtype=x_L.dtype)
        summary = self.certifier.bound(x_L, x_U, order='diag')

        drift_L, drift_U = linear_drift_dot_grad_bounds(
            drift, x_L, x_U,
            summary.jacobian_lower, summary.jacobian_upper,
        )
        g_L = _GBM_SIGMA * x_L
        g_U = _GBM_SIGMA * x_U
        diff_L, diff_U = diagonal_diffusion_bounds(summary, g_L, g_U)

        return drift_U + diff_U


def _policy_ibp(policy: nn.Sequential,
                x_L: torch.Tensor, x_U: torch.Tensor):
    """Forward IBP through a Linear->Tanh->...->Linear Sequential policy."""
    h_L, h_U = x_L, x_U
    for layer in policy:
        if isinstance(layer, nn.Linear):
            W, b = layer.weight, layer.bias
            W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
            new_L = h_L @ W_pos.T + h_U @ W_neg.T + b
            new_U = h_U @ W_pos.T + h_L @ W_neg.T + b
            h_L, h_U = new_L, new_U
        elif isinstance(layer, nn.Tanh):
            h_L, h_U = torch.tanh(h_L), torch.tanh(h_U)
    return h_L, h_U


class PendulumHardcodedIBPVerifier:
    """Interval upper bound on LV for the inverted pendulum, no auto_LiRPA.

    State: x = [phi (angular velocity), theta (angle)].
    Drift: f = [a1*sin(theta) + a2*u(x) - a3*phi, phi].
    Diffusion: g = [sigma, 0] (constant, only phi is noisy).

    The bound uses make_generator_bound_fn from mlp_hessian_ibp, which needs:
      - drift_interval(x_L, x_U) -> (f_L, f_U), shape [batch, 2]
      - covariance_interval(x_L, x_U) -> (C_L, C_U), shape [batch, 2, 2]
        where C = g g^T, constant [[sigma^2, 0], [0, 0]] here
    """

    def __init__(self, net: nn.Module, policy: nn.Sequential,
                 a1: float = 19.62, a2: float = 160.0, a3: float = 13.33,
                 sigma: float = 2.0):
        certifier = MLPHessianIBP(net)
        self._sigma = sigma

        def drift_interval(x_L, x_U):
            phi_L, phi_U   = x_L[:, 0:1], x_U[:, 0:1]
            theta_L, theta_U = x_L[:, 1:2], x_U[:, 1:2]
            s_L, s_U = _sin_bounds(theta_L.squeeze(1), theta_U.squeeze(1))
            s_L, s_U = s_L.unsqueeze(1), s_U.unsqueeze(1)
            u_L, u_U = _policy_ibp(policy, x_L, x_U)
            f_phi_L = a1 * s_L + a2 * u_L - a3 * phi_U
            f_phi_U = a1 * s_U + a2 * u_U - a3 * phi_L
            return (torch.cat([f_phi_L, phi_L], dim=1),
                    torch.cat([f_phi_U, phi_U], dim=1))

        def covariance_interval(x_L, x_U):
            # g = [sigma, 0], covariance C = g g^T = [[sigma^2,0],[0,0]], constant
            batch = x_L.shape[0]
            C = torch.zeros(batch, 2, 2, dtype=x_L.dtype, device=x_L.device)
            C[:, 0, 0] = sigma ** 2
            return C, C

        self._bound_fn = make_generator_bound_fn(
            certifier, drift_interval, covariance_interval
        )

    def generator_upper_bound(self, x_L: torch.Tensor,
                              x_U: torch.Tensor) -> torch.Tensor:
        """Upper bound on LV over each box [x_L, x_U]; output shape [batch, 1]."""
        _, ub = self._bound_fn(x_L, x_U)
        return ub


class HardcodedCellVerificationSystem:
    """Cell-splitting verifier that calls GBMHardcodedIBPVerifier directly."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self._corners = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        )

    def verify(
        self,
        verifier: GBMHardcodedIBPVerifier,
        locations: torch.Tensor,
        magnitude: torch.Tensor,
        depth: int = 0,
    ) -> torch.Tensor:
        x_L = locations - magnitude
        x_U = locations + magnitude
        ub = verifier.generator_upper_bound(x_L, x_U)
        mask = ub.squeeze(-1) >= 0.0
        counterexamples = locations[mask]

        if torch.numel(counterexamples) > 0:
            print(
                f"Could not verify decrease at {counterexamples.shape[0]} "
                "cells. Splitting further"
            )
            if depth < self.max_depth:
                half_m = 0.5 * magnitude
                corners = self._corners.to(locations.device)
                new_cells = (counterexamples.unsqueeze(1) + half_m * corners.unsqueeze(0)).reshape(-1, locations.shape[1])
                counterexamples = self.verify(
                    verifier, new_cells, half_m, depth + 1
                )
        return counterexamples

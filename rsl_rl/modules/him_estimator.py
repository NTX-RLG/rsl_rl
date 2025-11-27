# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.networks import MLP
from rsl_rl.utils.utils import resolve_nn_activation


LINEAR_VELOCITY_DIM: int = 3


class HimEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_one_step_priveleged_obs: int,
        enc_hidden_dims: tuple[int] | list[int] = [256, 128, 64],
        proj_hidden_dims: tuple[int] | list[int] = [256, 256],  # Projector hidden dims for Barlow Twins
        estimation_targets: str | list[str] | None = None,
        estimate_offsets: dict[str, int] | None = None,
        height_map_dim: int | None = None,
        height_map_hidden_dims: tuple[int] | list[int] = (512, 512, 256),
        height_map_dropout: float = 0.1,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        barlow_lambda: float = 5e-3,  # Barlow Twins: weight for off-diagonal terms
        projector_output_dim: int = 256,  # Projector output dimension (higher dim for redundancy reduction)
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "Estimator_CL.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_priveleged_obs = num_one_step_priveleged_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.barlow_lambda = barlow_lambda
        self.projector_output_dim = projector_output_dim

        if estimation_targets is None:
            estimation_targets = ["linear_velocity"]
        elif isinstance(estimation_targets, str):
            estimation_targets = [estimation_targets]
        else:
            estimation_targets = list(estimation_targets)

        estimation_targets = list(dict.fromkeys(t.lower() for t in estimation_targets))
        if not estimation_targets:
            raise ValueError("At least one estimation target must be specified.")

        supported_targets = {"linear_velocity", "height_map"}
        invalid = set(estimation_targets) - supported_targets
        if invalid:
            raise ValueError(f"Unsupported estimation targets: {sorted(invalid)}")

        self.estimation_targets = tuple(estimation_targets)

        self.target_dims = {}
        for target in self.estimation_targets:
            if target == "linear_velocity":
                self.target_dims[target] = LINEAR_VELOCITY_DIM
            elif target == "height_map":
                if height_map_dim is None or height_map_dim <= 0:
                    raise ValueError(
                        "height_map_dim must be provided and positive when height map estimation is enabled."
                    )
                self.target_dims[target] = height_map_dim

        self.estimate_dim = sum(self.target_dims.values())

        if not isinstance(estimate_offsets, dict) or not estimate_offsets:
            raise ValueError("estimate_offsets must be a non-empty dict with per-target privileged offsets.")

        self.target_offsets = {}
        for target in self.estimation_targets:
            if target not in estimate_offsets:
                raise ValueError(
                    f"Missing estimate offset for target '{target}'. Pass estimate_offsets from the actor configuration."
                )
            self.target_offsets[target] = estimate_offsets[target]

        # Shared Encoder (Backbone) - processes temporal observations
        enc_input_dim = temporal_steps * self.num_one_step_obs
        self.encoder = MLP(enc_input_dim, enc_hidden_dims[-1], enc_hidden_dims[:-1], activation)
        # Prediction heads (auxiliary estimation task)
        self.pred_heads = nn.ModuleDict()
        self.pred_activations = nn.ModuleDict()
        for target in self.estimation_targets:
            head, head_activation = self._build_prediction_head(
                enc_hidden_dims[-1],
                target,
                height_map_dim,
                height_map_hidden_dims,
                height_map_dropout,
                activation,
            )
            self.pred_heads[target] = head
            self.pred_activations[target] = head_activation

        # Projector (MLP head) - projects representations to high-dimensional space
        # This is the key component in Barlow Twins for redundancy reduction
        sizes = [enc_hidden_dims[-1]] + proj_hidden_dims + [projector_output_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(projector_output_dim, affine=False)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference.
        Returns the state estimate and encoder representation (NOT projector output).

        Note: In Barlow Twins, the projector is only used during training for
        computing the cross-correlation matrix. For inference/policy, we use
        the encoder's representation directly.
        """
        repr = self.encoder(obs_history.detach())
        estimate = self._predict_from_latent(repr)
        # Return encoder representation, NOT projector output
        return estimate.detach(), repr.detach()

    def update(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        next_obs: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """
        Update using Barlow Twins self-supervised learning.

        Args:
            obs: history observations at t-5:t (first view)
            critic_obs: critic observations containing ground truth state
            next_obs: history observations at t-4:t+1 (second view, shifted by 1 step)
            lr: optional learning rate
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Extract ground truth state for estimation loss
        targets_state = {}
        for target in self.estimation_targets:
            dim = self.target_dims[target]
            if dim == 0:
                continue
            start = -self.num_one_step_priveleged_obs + self.num_one_step_obs + self.target_offsets[target]
            targets_state[target] = self._slice_target_state(critic_obs, start, dim).detach()

        # Pass both views through the SAME encoder (key point of Barlow Twins)
        # IMPORTANT: Do NOT detach inputs - we need gradients to flow back to encoder!
        repr_1 = self.encoder(obs)
        repr_2 = self.encoder(next_obs)

        # Split into state estimate(s) and representations
        state_estimates = {
            target: self.pred_activations[target](self.pred_heads[target](repr_1))
            for target in self.estimation_targets
        }

        # Project representations to high-dimensional space (Barlow Twins projector)
        z_1 = self.projector(repr_1)
        z_2 = self.projector(repr_2)

        # Compute Barlow Twins loss
        # Cross-correlation matrix C: element C[i,j] is correlation between feature i and j
        batch_size = z_1.size(0)
        # empirical cross-correlation matrix
        c = self.bn(z_1).T @ self.bn(z_2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        # Total Barlow Twins loss
        barlow_loss = on_diag + self.barlow_lambda * off_diag

        # State estimation loss (auxiliary task)
        estimation_loss = torch.tensor(0.0, device=barlow_loss.device)
        for target, state in targets_state.items():
            estimation_loss = estimation_loss + F.mse_loss(state, state_estimates[target])

        # Combined loss
        losses = estimation_loss + barlow_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), barlow_loss.item()

    def _build_prediction_head(
        self,
        input_dim: int,
        target_name: str,
        height_map_dim: int | None,
        height_map_hidden_dims: tuple[int] | list[int],
        height_map_dropout: float,
        activation: str,
    ) -> tuple[nn.Module, nn.Module]:
        if target_name == "height_map":
            layers = []
            prev_dim = input_dim
            for hidden_dim in height_map_hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(resolve_nn_activation(activation))
                if height_map_dropout > 0:
                    layers.append(nn.Dropout(height_map_dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, height_map_dim))
            return nn.Sequential(*layers), nn.Tanh()

        if target_name == "linear_velocity":
            return MLP(input_dim, LINEAR_VELOCITY_DIM, [128, 64], activation), nn.Identity()

        raise ValueError(f"Unsupported estimation target '{target_name}'.")

    def _predict_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.estimation_targets:
            return torch.zeros(latent.size(0), 0, device=latent.device)
        estimates = [self.pred_activations[t](self.pred_heads[t](latent)) for t in self.estimation_targets]
        return torch.cat(estimates, dim=-1)

    def predict_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Public helper so exporters can reuse the prediction heads."""
        return self._predict_from_latent(latent)

    def _slice_target_state(self, critic_obs: torch.Tensor, relative_start: int, dim: int) -> torch.Tensor:
        """Convert relative offsets (which may be negative) into valid tensor slices."""
        if dim <= 0:
            raise ValueError("Target dimension must be positive.")

        critic_dim = critic_obs.size(-1)
        end = relative_start + dim

        start_idx = relative_start if relative_start >= 0 else critic_dim + relative_start
        end_idx = end if end > 0 else critic_dim + end

        if not (0 <= start_idx < critic_dim) or not (0 < end_idx <= critic_dim) or end_idx <= start_idx:
            raise IndexError(
                f"Invalid slice for target state: start={relative_start}, dim={dim}, critic_dim={critic_dim}."
            )

        return critic_obs[:, start_idx:end_idx]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

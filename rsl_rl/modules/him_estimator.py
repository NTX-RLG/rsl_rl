# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import torch
import torch.distributed as dist
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
        enc_hidden_dims: Sequence[int] = [256, 128, 64],
        proj_hidden_dims: Sequence[int] = [256, 256],  # Projector hidden dims for Barlow Twins
        height_map_dim: int | None = None,
        height_map_offset: int | None = None,
        linear_velocity_offset: int = 0,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        barlow_lambda: float = 5e-3,  # Barlow Twins: weight for off-diagonal terms
        projector_output_dim: int = 256,  # Projector output dimension (higher dim for redundancy reduction)
        terrain_latent_dim: int | None = None,
        terrain_latent_hidden_dims: Sequence[int] = (256, 256),
        terrain_latent_weight: float = 1.0,
        terrain_latent_dropout: float = 0.1,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_priveleged_obs = num_one_step_priveleged_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.barlow_lambda = barlow_lambda
        self.projector_output_dim = projector_output_dim
        self.terrain_latent_weight = terrain_latent_weight
        self.height_map_dim = height_map_dim if height_map_dim and height_map_dim > 0 else None
        self.height_map_offset = height_map_offset
        self.linear_velocity_offset = linear_velocity_offset
        self.estimate_dim = LINEAR_VELOCITY_DIM

        # Shared Encoder (Backbone) - processes temporal observations
        enc_input_dim = temporal_steps * self.num_one_step_obs
        self.encoder = MLP(enc_input_dim, enc_hidden_dims[-1], enc_hidden_dims[:-1], activation)
        # Linear velocity prediction head (auxiliary estimation task)
        self.linear_velocity_head = MLP(enc_hidden_dims[-1], LINEAR_VELOCITY_DIM, [128, 64], activation)

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

        # Terrain-specific latent modeling (optional)
        self.terrain_latent_head: nn.Module | None = None
        self.height_teacher: nn.Module | None = None
        terrain_latent_dim = terrain_latent_dim if terrain_latent_dim and terrain_latent_dim > 0 else None
        enable_terrain_latent = terrain_latent_dim is not None and self.height_map_dim is not None
        if enable_terrain_latent:
            if self.height_map_offset is None:
                raise ValueError("height_map_offset must be provided when terrain latent modeling is enabled.")
            self.terrain_latent_head = self._build_terrain_latent_head(
                enc_hidden_dims[-1],
                terrain_latent_dim,
                terrain_latent_hidden_dims,
                activation,
                terrain_latent_dropout,
            )
            self.height_teacher = HeightMapTeacherEncoder(
                num_samples=self.height_map_dim,
                latent_dim=terrain_latent_dim,
                mlp_hidden_dims=terrain_latent_hidden_dims,
                dropout=terrain_latent_dropout,
                activation=activation,
            )
            self.terrain_latent_dim = terrain_latent_dim
        else:
            self.terrain_latent_dim = 0
            self.terrain_latent_weight = 0.0
            self.height_map_offset = None

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.inference_mode()
    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for inference.
        Returns the state estimate and encoder representation (NOT projector output).

        Note: In Barlow Twins, the projector is only used during training for
        computing the cross-correlation matrix. For inference/policy, we use
        the encoder's representation directly.
        """
        repr = self.encoder(obs_history)
        estimate = self.linear_velocity_head(repr)
        terrain_latent = None
        if self.terrain_latent_head is not None:
            terrain_latent = self.terrain_latent_head(repr)
        # Return encoder representation, NOT projector output
        return estimate, repr, None if terrain_latent is None else terrain_latent

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
        start = -self.num_one_step_priveleged_obs + self.num_one_step_obs + self.linear_velocity_offset
        target_state = self._slice_target_state(critic_obs, start, LINEAR_VELOCITY_DIM).detach()

        height_teacher_state = None
        if self.height_teacher is not None and self.height_map_dim is not None and self.height_map_offset is not None:
            start = -self.num_one_step_priveleged_obs + self.num_one_step_obs + self.height_map_offset
            height_teacher_state = self._slice_target_state(critic_obs, start, self.height_map_dim).detach()

        # Pass both views through the SAME encoder (key point of Barlow Twins)
        # IMPORTANT: Do NOT detach inputs - we need gradients to flow back to encoder!
        repr_1 = self.encoder(obs)
        repr_2 = self.encoder(next_obs)

        # Linear velocity estimate from representation
        lin_vel_estimate = self.linear_velocity_head(repr_1)

        # Project representations to high-dimensional space (Barlow Twins projector)
        z_1 = self.projector(repr_1)
        z_2 = self.projector(repr_2)

        # Compute Barlow Twins loss
        # Cross-correlation matrix C: element C[i,j] is correlation between feature i and j
        batch_size = z_1.size(0)
        # empirical cross-correlation matrix
        z1_norm = self.bn(z_1)
        z2_norm = self.bn(z_2)
        c = z1_norm.T @ z2_norm

        # sum the cross-correlation matrix between all gpus
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c)
            world_size = dist.get_world_size()
            c.div_(batch_size * world_size)
        else:
            c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        # Total Barlow Twins loss
        barlow_loss = on_diag + self.barlow_lambda * off_diag

        # State estimation loss (auxiliary task)
        estimation_loss = F.mse_loss(target_state, lin_vel_estimate)

        terrain_latent_loss = torch.tensor(0.0, device=barlow_loss.device)
        if self.terrain_latent_head is not None and height_teacher_state is not None:
            pred_latent = self.terrain_latent_head(repr_1)
            with torch.no_grad():
                teacher_latent = self.height_teacher(height_teacher_state)
            terrain_latent_loss = F.mse_loss(pred_latent, teacher_latent)

        # Combined loss
        losses = estimation_loss + barlow_loss + self.terrain_latent_weight * terrain_latent_loss

        self.optimizer.zero_grad()
        losses.backward()
        self._reduce_gradients()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), barlow_loss.item()

    def _reduce_gradients(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        grads = [param.grad.view(-1) for param in self.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        all_grads /= dist.get_world_size()

        offset = 0
        for param in self.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def predict_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Public helper so exporters can reuse the prediction head."""
        return self.linear_velocity_head(latent)

    def _build_terrain_latent_head(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int] | list[int],
        activation: str,
        dropout: float,
    ) -> nn.Module:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(resolve_nn_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _slice_target_state(self, critic_obs: torch.Tensor, relative_start: int, dim: int) -> torch.Tensor:
        """Convert relative offsets (which may be negative) into valid tensor slices."""
        if dim <= 0:
            raise ValueError("Target dimension must be positive.")

        critic_dim = critic_obs.size(-1)

        start_idx = relative_start if relative_start >= 0 else critic_dim + relative_start

        if start_idx < 0 or start_idx + dim > critic_dim:
            raise IndexError(
                f"Invalid slice: start={relative_start} (idx={start_idx}), dim={dim}, critic_dim={critic_dim}."
            )

        return critic_obs.narrow(dim=-1, start=start_idx, length=dim)

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


class HeightMapTeacherEncoder(nn.Module):
    """Convolutional encoder that compresses ~200 local height samples into a compact latent."""

    def __init__(
        self,
        num_samples: int,
        latent_dim: int,
        mlp_hidden_dims: tuple[int] | list[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if num_samples is None or num_samples <= 0:
            raise ValueError("num_samples must be positive for height-map encoding.")
        self.num_samples = num_samples

        mlp_layers: list[nn.Module] = []
        prev_dim = num_samples
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(resolve_nn_activation(activation))
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, latent_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, height_samples: torch.Tensor) -> torch.Tensor:
        return self.mlp(height_samples)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

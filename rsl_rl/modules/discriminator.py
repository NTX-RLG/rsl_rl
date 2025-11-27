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
from torch import autograd

DISC_LOGIT_INIT_SCALE: float = 1.0


class Discriminator(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        observation_horizon: int,
        device: str,
        reward_coef: float = 0.1,
        reward_lerp: float = 0.3,
        shape: tuple[int] | list[int] = [1024, 512],
        style_reward_function: str = "quad_mapping",
        mask_dims: list[tuple[int, int]] | None = None,  # List of dimension ranges to mask, e.g., [(start1, end1), (start2, end2)]
        mask_joint_names: list[str] | None = None,  # List of joint names to mask (both position and velocity)
        joint_names: list[str] | None = None,  # List of all joint names in order
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape

        # Setup mask for specific dimensions
        self.mask_dims = mask_dims
        self.mask_joint_names = mask_joint_names
        self.joint_names = joint_names
        self._setup_mask()

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.LeakyReLU())
            curr_in_dim = hidden_dim
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.discriminator_logits = torch.nn.Linear(hidden_dim, 1)
        self.train()

    def _setup_mask(self) -> None:
        """Setup mask tensor for masking specific dimensions in the observation."""
        # Import MotionLoaderE1 indices locally to avoid circular imports
        from rsl_rl.utils.motion_loader import MotionLoaderE1

        # Create a mask tensor (1 = keep, 0 = mask out)
        mask = torch.ones(self.observation_dim, dtype=torch.float32, device=self.device)

        # Track if any masking is applied
        has_mask = False

        # Method 1: Mask by dimension ranges
        if self.mask_dims is not None and len(self.mask_dims) > 0:
            has_mask = True
            for start_idx, end_idx in self.mask_dims:
                if start_idx >= 0 and end_idx <= self.observation_dim:
                    mask[start_idx:end_idx] = 0.0
                else:
                    print(
                        f"Warning: Invalid mask range ({start_idx}, {end_idx}) for observation_dim"
                        f" {self.observation_dim}"
                    )

        # Method 2: Mask by joint names (both position and velocity)
        if self.mask_joint_names is not None and len(self.mask_joint_names) > 0:
            if self.joint_names is None:
                print("Warning: mask_joint_names provided but joint_names is None. Skipping joint-based masking.")
            else:
                has_mask = True
                masked_joints = []

                # Find indices of joints to mask
                joint_indices = []
                for mask_joint in self.mask_joint_names:
                    try:
                        idx = self.joint_names.index(mask_joint)
                        joint_indices.append(idx)
                        masked_joints.append(mask_joint)
                    except ValueError:
                        print(f"Warning: Joint '{mask_joint}' not found in joint_names. Skipping.")

                if joint_indices:
                    # Calculate the relative indices in the AMP observation
                    # AMP observation starts from JOINT_POSE
                    joint_pos_size = MotionLoaderE1.JOINT_POS_SIZE
                    joint_vel_start_relative = MotionLoaderE1.JOINT_VEL_START_IDX - MotionLoaderE1.JOINT_POSE_START_IDX

                    for joint_idx in joint_indices:
                        # Mask joint position
                        pos_idx = joint_idx
                        if pos_idx < joint_pos_size:
                            mask[pos_idx] = 0.0

                        # Mask joint velocity
                        vel_idx = joint_vel_start_relative + joint_idx
                        if vel_idx < self.observation_dim:
                            mask[vel_idx] = 0.0

                    print(f"Discriminator: Masking joints {masked_joints} (pos & vel)")

        # If no masking is applied, set mask to None
        if not has_mask:
            self.mask = None
            return

        # Expand mask to cover all frames in observation horizon
        # Shape: (observation_dim * observation_horizon,)
        self.mask = mask.repeat(self.observation_horizon)

        # Count masked dimensions for logging
        masked_count = (self.mask == 0).sum().item()
        total_dims = self.observation_dim * self.observation_horizon
        print(f"Discriminator: Masking {masked_count}/{total_dims} dimensions total")

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mask to input observations.

        Args:
            x: Input tensor of shape (batch_size, observation_dim * observation_horizon)

        Returns:
            Masked input tensor
        """
        if self.mask is None:
            return x
        return x * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_mask(x)
        return self.discriminator_logits(self.architecture(x))

    def get_disc_weights(self) -> list[torch.Tensor]:
        weights = []
        for m in self.architecture.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        return weights

    def get_disc_logit_weights(self) -> torch.Tensor:
        return torch.flatten(self.discriminator_logits.weight)

    def eval_disc(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_mask(x)
        return self.discriminator_logits(self.architecture(x))

    def compute_grad_pen(self, expert_data: torch.Tensor, lambda_: float = 10) -> torch.Tensor:
        # Enable gradient computation only for expert_data
        expert_data.requires_grad_(True)
        disc = self.eval_disc(expert_data)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=torch.ones(disc.size(), device=disc.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        # Use more efficient computation: sum first, then pow
        grad_pen = lambda_ * (grad.pow(2).sum(dim=1).mean())
        return grad_pen

    def compute_wgan_grad_pen(
        self,
        expert_data: torch.Tensor,
        policy_data: torch.Tensor,
        k: float = 2,
        p: float = 6,
    ) -> torch.Tensor:
        expert_data.requires_grad_(True)
        policy_data.requires_grad_(True)

        expert_d = self.eval_disc(expert_data)
        expert_grad = autograd.grad(
            outputs=expert_d,
            inputs=expert_data,
            grad_outputs=torch.ones(expert_d.size(), device=expert_d.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # Optimize: avoid view operation and use direct computation
        expert_grad_norm = expert_grad.pow(2).sum(1).pow(p / 2)

        policy_d = self.eval_disc(policy_data)
        policy_grad = autograd.grad(
            outputs=policy_d,
            inputs=policy_data,
            grad_outputs=torch.ones(policy_d.size(), device=policy_d.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # Optimize: avoid view operation and use direct computation
        policy_grad_norm = policy_grad.pow(2).sum(1).pow(p / 2)

        grad_pen = (expert_grad_norm.mean() + policy_grad_norm.mean()) * k * 0.5
        return grad_pen

    def compute_weight_decay(self, lambda_: float = 0.0001) -> torch.Tensor:
        disc_weights = self.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        weight_decay = lambda_ * torch.sum(torch.square(disc_weights))
        return weight_decay

    def compute_logit_reg(self, lambda_: float = 0.05) -> torch.Tensor:
        logit_weights = self.get_disc_logit_weights()
        disc_logit_loss = lambda_ * torch.sum(torch.square(logit_weights))
        return disc_logit_loss

    def predict_amp_reward(
        self,
        state_buf: torch.Tensor,
        task_reward: torch.Tensor,
        state_normalizer: Any | None = None,
        style_reward_normalizer: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.eval()
            # Avoid unnecessary clone when normalizer is not used
            if state_normalizer is not None:
                # Batch normalize: reshape all frames at once for efficiency
                batch_size = state_buf.shape[0]
                state_flat = state_buf.view(-1, state_buf.shape[-1])
                state_flat_norm = state_normalizer.normalize_torch(state_flat, self.device)
                state_buf_norm = state_flat_norm.view(batch_size, -1)
                d = self.eval_disc(state_buf_norm)
            else:
                d = self.eval_disc(state_buf.flatten(1, 2))

            if self.style_reward_function == "quad_mapping":
                style_reward = torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = -torch.log(
                    torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device))
                )
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    d_clone = d.clone()
                    style_reward = style_reward_normalizer.normalize_torch(d_clone, self.device)
                    style_reward_normalizer.update(d.cpu().numpy())
                else:
                    style_reward = torch.exp(torch.tanh(0.3 * d)) - torch.exp(-1 * torch.ones_like(d))
            else:
                raise ValueError("Unexpected style reward mapping specified")
            style_reward *= (1.0 - self.reward_lerp) * self.reward_coef
            task_reward = task_reward.unsqueeze(-1) * self.reward_lerp
            reward = style_reward + task_reward
            self.train()
        return reward.squeeze(), style_reward.squeeze()

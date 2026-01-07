# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn
import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

from .him_estimator import LINEAR_VELOCITY_DIM, HimEstimator


class HimActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_one_step_obs: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        encoder_hidden_dims: tuple[int] | list[int] = [
            256,
            128,
            64,
        ],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims: tuple[int] | list[int] = [256, 256],  # Projector hidden dims for Barlow Twins
        projector_output_dim: int = 256,  # Projector output dim (only used in training)
        height_map_dim: int | None = None,
        height_map_offset: int | None = None,
        linear_velocity_offset: int = 0,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        use_height_latent: bool = True,
        terrain_latent_dim: int = 128,
        terrain_latent_hidden_dims: tuple[int] | list[int] = (256, 256),
        terrain_latent_weight: float = 1.0,
        terrain_latent_dropout: float = 0.1,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        self.estimate_dim = LINEAR_VELOCITY_DIM
        self.linear_velocity_offset = linear_velocity_offset
        self.height_map_dim = height_map_dim if height_map_dim and height_map_dim > 0 else 0
        self.use_height_latent = False
        self.height_latent_dim = 0
        self.actor_estimate_dim = self.estimate_dim

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.num_one_step_obs = num_one_step_obs
        history_size = int(num_actor_obs / num_one_step_obs)

        privileged_budget = max(num_critic_obs - self.num_one_step_obs, 0)
        if self.height_map_dim > privileged_budget:
            warnings.warn(
                "height_map_dim exceeds available critic privileged observations. Disabling height map inputs.",
                stacklevel=2,
            )
            self.height_map_dim = 0

        self.use_height_latent = bool(use_height_latent and terrain_latent_dim > 0 and self.height_map_dim > 0)
        self.height_latent_dim = terrain_latent_dim if self.use_height_latent else 0

        # Encoder output dimension (for actor input)
        encoder_latent_dim = encoder_hidden_dims[-1]

        self.height_map_offset = height_map_offset
        if self.height_map_dim == 0:
            self.height_map_offset = None
        elif self.height_map_offset is None:
            self.height_map_offset = num_critic_obs - self.num_one_step_obs - self.height_map_dim
            if self.height_map_offset < 0:
                warnings.warn(
                    "Computed height map offset is negative for the provided critic layout. Disabling height map inputs.",
                    stacklevel=2,
                )
                self.height_map_dim = 0
                self.height_map_offset = None
                self.use_height_latent = False
                self.height_latent_dim = 0

        # Actor: receives [current_obs, estimate, encoder_representation]
        actor_input_dim = self.num_one_step_obs + self.actor_estimate_dim + encoder_latent_dim + self.height_latent_dim
        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Estimator
        self.him_estimator = HimEstimator(
            temporal_steps=history_size,
            num_one_step_obs=self.num_one_step_obs,
            num_one_step_priveleged_obs=num_critic_obs,
            enc_hidden_dims=encoder_hidden_dims,
            proj_hidden_dims=projector_hidden_dims,
            height_map_dim=self.height_map_dim if self.height_map_dim > 0 else None,
            height_map_offset=self.height_map_offset,
            linear_velocity_offset=self.linear_velocity_offset,
            projector_output_dim=projector_output_dim,
            terrain_latent_dim=self.height_latent_dim if self.use_height_latent else None,
            terrain_latent_hidden_dims=terrain_latent_hidden_dims,
            terrain_latent_weight=terrain_latent_weight,
            terrain_latent_dropout=terrain_latent_dropout,
        )
        print(f"Estimator Encoder: {self.him_estimator.encoder}")
        print(f"Estimator Projector: {self.him_estimator.projector}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        with torch.no_grad():
            estimates, latent, terrain_latent = self.him_estimator(obs)
        height_latent = self._format_height_latent(terrain_latent, obs.size(0), obs.device, obs.dtype)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], estimates, latent, height_latent), dim=-1)
        # Compute mean
        mean = self.actor(actor_input)
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        estimates, latent, terrain_latent = self.him_estimator(obs)
        height_latent = self._format_height_latent(terrain_latent, obs.size(0), obs.device, obs.dtype)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], estimates, latent, height_latent), dim=-1)
        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def _format_height_latent(
        self, terrain_latent: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.height_latent_dim == 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if terrain_latent is None:
            return torch.zeros(batch_size, self.height_latent_dim, device=device, dtype=dtype)
        return terrain_latent.to(device=device, dtype=dtype)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

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

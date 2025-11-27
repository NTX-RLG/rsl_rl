# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization

from .him_estimator import HimEstimator, LINEAR_VELOCITY_DIM


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
        encoder_hidden_dims: tuple[int] | list[int] = [256, 128, 64],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims: tuple[int] | list[int] = [256, 256],  # Projector hidden dims for Barlow Twins
        projector_output_dim: int = 256,  # Projector output dim (only used in training)
        estimation_targets: str | list[str] | None = None,
        estimate_offsets: dict[str, int] | None = None,
        height_map_dim: int | None = None,
        height_map_hidden_dims: tuple[int] | list[int] = (512, 512, 256),
        height_map_dropout: float = 0.1,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

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

        # Encoder output dimension (for actor input)
        encoder_latent_dim = encoder_hidden_dims[-1]
        estimate_offsets = estimate_offsets.copy() if isinstance(estimate_offsets, dict) else {}

        self.target_offsets = {}
        for target in self.estimation_targets:
            offset = estimate_offsets.get(target)
            if offset is None:
                if target == "linear_velocity":
                    offset = 0
                else:
                    offset = num_critic_obs - self.num_one_step_obs - self.target_dims[target]
                    if offset < 0:
                        raise ValueError(
                            "Computed height map estimate offset is negative. Verify critic observation layout or set"
                            " estimate_offsets manually."
                        )
            self.target_offsets[target] = offset

        # Actor: receives [current_obs, estimate, encoder_representation]
        self.actor = MLP(
            self.num_one_step_obs + self.estimate_dim + encoder_latent_dim, num_actions, actor_hidden_dims, activation
        )
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
            estimation_targets=self.estimation_targets,
            estimate_offsets=self.target_offsets,
            height_map_dim=height_map_dim,
            height_map_hidden_dims=height_map_hidden_dims,
            height_map_dropout=height_map_dropout,
            projector_output_dim=projector_output_dim,
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
            estimates, latent = self.him_estimator(obs)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], estimates, latent), dim=-1)
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
        estimates, latent = self.him_estimator(obs)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], estimates, latent), dim=-1)
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

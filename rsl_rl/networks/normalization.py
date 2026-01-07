# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: int | tuple[int] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        mean_x = torch.mean(x, dim=0, keepdim=True)
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)

        self.count += count_x
        rate = count_x / self.count
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def synchronize(self) -> None:
        """
        Synchronizes the running statistics across all connected GPUs.
        This merges moments across workers using count-weighted formulas.

        NOTE: Directly averaging mean/variance is incorrect when each rank sees
        non-identically distributed data (common in RL) and can cause large
        normalization jumps.
        """
        if not dist.is_available() or not dist.is_initialized():
            return

        # Use population moments: E[x] and E[x^2]
        # where var = E[x^2] - mean^2 (unbiased=False in update).
        device = self._mean.device
        dtype = torch.float64

        local_count = self.count.to(device=device, dtype=dtype)
        if local_count.item() <= 0:
            return

        local_mean = self._mean.to(dtype=dtype)
        local_var = self._var.to(dtype=dtype)
        local_ex2 = local_var + local_mean.square()

        local_sum = local_mean * local_count
        local_sum_ex2 = local_ex2 * local_count

        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum_ex2, op=dist.ReduceOp.SUM)

        if local_count.item() <= 0:
            return

        global_mean = local_sum / local_count
        global_ex2 = local_sum_ex2 / local_count
        global_var = torch.clamp(global_ex2 - global_mean.square(), min=0.0)

        # Update local stats (keep original dtype)
        self._mean.copy_(global_mean.to(dtype=self._mean.dtype))
        self._var.copy_(global_var.to(dtype=self._var.dtype))
        self._std = torch.sqrt(self._var)
        # Update count (used for `until` gating)
        self.count.copy_(local_count.to(dtype=self.count.dtype))

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self, shape: int | tuple[int] | list[int], eps: float = 1e-2, gamma: float = 0.99, until: int | None = None
    ) -> None:
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg

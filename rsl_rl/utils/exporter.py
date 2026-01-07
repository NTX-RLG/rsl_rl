# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os

import onnx
import torch
import torch.nn.functional as F


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            if hasattr(policy, "him_estimator"):
                self.him_estimator = copy.deepcopy(policy.him_estimator)
                self.forward = self.forward_him
                self.num_one_step_obs = policy.num_one_step_obs
                self.estimate_dim = policy.him_estimator.estimate_dim
                self.height_latent_dim = getattr(policy, "height_latent_dim", 0)

        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward_him(self, obs):
        x_in = self.normalizer(obs)
        z = self.him_estimator.encoder(x_in)
        estimate = self.him_estimator.predict_from_latent(z)
        terrain_latent = None
        terrain_head = getattr(self.him_estimator, "terrain_latent_head", None)
        if terrain_head is not None:
            terrain_latent = terrain_head(z)

        pieces = [x_in[:, -self.num_one_step_obs :]]
        pieces.append(estimate)
        pieces.append(z)
        if self.height_latent_dim > 0:
            if terrain_latent is None:
                terrain_latent = torch.zeros(x_in.size(0), self.height_latent_dim, device=x_in.device, dtype=x_in.dtype)
            pieces.append(terrain_latent)
        return self.actor(torch.cat(pieces, dim=-1))

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            if hasattr(self, "him_estimator"):
                obs = torch.zeros(1, self.him_estimator.encoder[0].in_features)
                torch.onnx.export(
                    self,
                    obs,
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
            else:
                obs = torch.zeros(1, self.actor[0].in_features)
                torch.onnx.export(
                    self,
                    obs,
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    metadata = {
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos[0].cpu().tolist(),
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
    }
    # metadata = {
    #     "joint_names": [env.scene["robot"].data.joint_names[i] for i in [0, 1, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19]],
    #     "joint_stiffness": env.scene["robot"].data.joint_stiffness[0, [0, 1, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19]].cpu().tolist(),
    #     "joint_damping": env.scene["robot"].data.joint_damping[0, [0, 1, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19]].cpu().tolist(),
    #     "default_joint_pos": env.scene["robot"].data.default_joint_pos[0, [0, 1, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19]].cpu().tolist(),
    #     "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
    # }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)

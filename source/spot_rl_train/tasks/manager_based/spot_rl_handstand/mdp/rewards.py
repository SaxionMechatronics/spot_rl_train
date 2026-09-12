# Copyright (c) 2026, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation from a desired base orientation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (N, 3)

    # Approximately 10 degrees from fully vertical
    target = torch.tensor([1.0, 0.0, 0.0]).to(g_b.device).unsqueeze(0)

    return torch.linalg.norm(g_b - target, dim=1)


def illegal_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize when the contact force exceeds threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    illegal = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    return illegal


def no_double_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize both front feet losing ground contact at the same time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force_mag = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]

    both_in_air = torch.all(force_mag < threshold, dim=1).float()
    return both_in_air


def contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float = 50.0,
) -> torch.Tensor:
    """Penalize contact forces with an exponential kernel.

    Maps the summed peak force magnitude through exp(-sum / std) and returns
    1 - exp(...) so the penalty is 0 when forces are near zero and approaches
    1 for large forces.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    force_mag = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)

    max_force_mag = torch.max(force_mag, dim=1)[0]
    total_force = max_force_mag.sum(dim=-1)

    return 1.0 - torch.exp(-total_force / std)


def lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base linear velocity in all 3 axes using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)


def ang_vel_xz_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xz-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, [0, 2]]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)

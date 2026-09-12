# Copyright (c) 2026, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class actuator_stiffness_curriculum(ManagerTermBase):
    """Reduce actuator stiffness from initial_stiffness to final_stiffness as mean completed episode length grows.

    The action scale of the joint position action term is adapted alongside the stiffness so that the commanded
    torque authority (stiffness * action_scale) stays constant over the curriculum::

        action_scale(stiffness) = action_scale_ref * (final_stiffness / stiffness)

    where ``action_scale_ref`` is the scale tuned for ``final_stiffness``. The scale therefore grows from
    ``action_scale_ref * final_stiffness / initial_stiffness`` up to ``action_scale_ref`` at the end of the curriculum.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._mean_ep_len: float = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        initial_stiffness: float,
        final_stiffness: float,
        min_episode_length: float,
        max_episode_length: float,
        exponent: float = 1.0,
        ema_alpha: float = 0.05,
        action_scale_ref: float = 0.2,
        action_term_name: str = "joint_pos",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> dict[str, float]:
        asset: Articulation = env.scene[asset_cfg.name]

        if len(env_ids) > 0:
            completed_mean = env.episode_length_buf[env_ids].float().mean().item()
            self._mean_ep_len = (1.0 - ema_alpha) * self._mean_ep_len + ema_alpha * completed_mean

        frac = (self._mean_ep_len - min_episode_length) / (max_episode_length - min_episode_length)
        frac = max(0.0, min(1.0, frac))
        new_stiffness_val = initial_stiffness + frac**exponent * (final_stiffness - initial_stiffness)

        for actuator in asset.actuators.values():
            actuator.stiffness[:] = new_stiffness_val

        # adapt the action scale to the current stiffness
        new_action_scale = action_scale_ref * final_stiffness / new_stiffness_val
        action_term = env.action_manager.get_term(action_term_name)
        if isinstance(action_term._scale, torch.Tensor):
            action_term._scale[:] = new_action_scale
        else:
            action_term._scale = new_action_scale

        # logged as "Curriculum/<term_name>/stiffness" and "Curriculum/<term_name>/action_scale"
        return {"stiffness": new_stiffness_val, "action_scale": new_action_scale}


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

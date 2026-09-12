# Copyright (c) 2026, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from dataclasses import replace as _replace

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import spot_rl_train.tasks.manager_based.spot_rl_handstand.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip


@configclass
class SpotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = _replace(
        SPOT_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "spot_hip": _replace(SPOT_CFG.actuators["spot_hip"], stiffness=60.0),
            "spot_knee": _replace(SPOT_CFG.actuators["spot_knee"], stiffness=60.0),
        },
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    height_scanner = None

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore[arg-type]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore[arg-type]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        },
    )


@configclass
class SpotRewardsCfg:
    """Reward terms for the MDP."""

    base_orientation = RewardTermCfg(
        func=mdp.base_orientation_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    illegal_contact = RewardTermCfg(
        func=mdp.illegal_contact_penalty,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["hr_foot", "hl_foot"]), "threshold": 1.0},
    )

    base_height = RewardTermCfg(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.75})

    base_ang_vel = RewardTermCfg(func=mdp.ang_vel_xz_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")})

    base_lin_vel = RewardTermCfg(func=mdp.lin_vel_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})

    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

    hop_penalty = RewardTermCfg(
        func=mdp.no_double_air_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["fr_foot", "fl_foot"]), "threshold": 1.0},
    )

    foot_slip = RewardTermCfg(
        func=mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    joint_acc = RewardTermCfg(
        func=mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    joint_torques = RewardTermCfg(
        func=mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    joint_vel = RewardTermCfg(
        func=mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    terminated = RewardTermCfg(func=mdp.is_terminated, weight=-500.0)


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )


@configclass
class SpotCurriculumCfg:
    """Curriculum terms for the MDP."""

    actuator_stiffness = CurrTerm(
        func=mdp.actuator_stiffness_curriculum,
        params={
            "initial_stiffness": 60.0,
            "final_stiffness": 40.0,
            "min_episode_length": 0.0,
            "max_episode_length": 1000.0,
            "exponent": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SpotRlHandstandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Spot robot in a flat environment."""

    scene: SpotSceneCfg = SpotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()
    curriculum: SpotCurriculumCfg = SpotCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class SpotRlHandstandEnvCfg_PLAY(SpotRlHandstandEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.robot = _replace(
            SPOT_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={  # type: ignore[assignment]
                "spot_hip": _replace(SPOT_CFG.actuators["spot_hip"], stiffness=40.0),
                "spot_knee": _replace(SPOT_CFG.actuators["spot_knee"], stiffness=40.0),
            },
        )
        self.curriculum = None  # type: ignore[assignment]

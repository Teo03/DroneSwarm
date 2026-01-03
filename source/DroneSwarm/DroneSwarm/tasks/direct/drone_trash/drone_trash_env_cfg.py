# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Drone Trash Collection environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .robots.quadcopter import CRAZYFLIE_CFG


@configclass
class DroneTrashEnvCfg(DirectRLEnvCfg):
    """Configuration for the drone trash collection environment."""

    # Environment
    episode_length_s = 30.0  # Longer episodes for trash collection
    decimation = 2
    action_space = 5  # thrust, roll, pitch, yaw moments + gripper
    observation_space = 23  # See observation space design in plan
    state_space = 0
    debug_vis = True

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Terrains/flat_plane.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # Default number of parallel environments
        env_spacing=15.0,  # Spacing between environments (> grid_size to avoid overlap)
        replicate_physics=True,  # Required for automatic collision filtering
        filter_collisions=True,  # Prevents collisions between cloned environments
    )

    # Robot configuration
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Trash configuration (rigid body sphere)
    trash: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Trash",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Bin configuration (static cylinder - visual only, detection is computed)
    # Note: We'll spawn this as a visual marker, not a rigid body

    # Drone control parameters (from droneswarm_env)
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Task-specific parameters (relaxed for learning)
    grid_size = 10.0  # Environment grid size in meters
    pickup_radius = 1.0  # Horizontal distance for pickup
    bin_radius = 2.5  # Horizontal distance for deposit
    hover_speed_threshold = 0.7  # Max speed for pickup (forgiving)
    gripper_close_threshold = 0.4  # Gripper threshold for grab
    gripper_open_threshold = 0.4  # Gripper threshold for release
    gripper_speed = 0.2  # Gripper speed per step
    max_vel = 50.0  # Normalization constant for velocity
    max_omega = 50.0  # Normalization constant for angular velocity

    # Reward scales
    pickup_reward = 1.0
    deposit_reward = 5.0
    completion_reward = 10.0
    drop_penalty = -1.0
    out_of_bounds_penalty = -1.0

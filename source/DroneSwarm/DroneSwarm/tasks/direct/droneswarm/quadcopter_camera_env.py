# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter Camera Environment - Fly to visible goal using RGB camera."""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class QuadcopterCameraEnvCfg(DirectRLEnvCfg):
    """Configuration for Quadcopter with Camera - goal reaching task."""

    # Environment
    episode_length_s = 10.0
    decimation = 2
    action_space = 4  # thrust, roll, pitch, yaw moments
    state_space = 0
    debug_vis = False

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        use_fabric=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Camera - front-facing on the drone
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.02),
            rot=(0.5, -0.5, 0.5, -0.5),  # Forward-facing
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=10.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=64,
        height=64,
    )
    write_image_to_file = False  # Set True to debug camera view

    # Observation space: camera image + drone state (like real IMU data)
    # State: linear_vel(3) + angular_vel(3) + orientation_quat(4) + height(1) = 11
    observation_space = {
        "camera": [64, 64, 3],
        "state": 11,
    }

    # Terrain - flat USD plane for cleaner visuals
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
        num_envs=256,
        env_spacing=25.0,
        replicate_physics=True,
    )

    # Robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Goal marker - large bright red sphere visible to camera
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.5,  # Bigger for easier visibility
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Bright red
                emissive_color=(0.5, 0.0, 0.0),  # Glowing red
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0)),
    )

    # Walls to isolate each environment visually (prevent seeing neighbor's goals)
    wall_height = 6.0
    wall_distance = 10.0  # Distance from center (half of env_spacing minus margin)

    # Front wall (blocks view forward into next env)
    wall_front: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WallFront",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 20.0, wall_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(wall_distance, 0.0, wall_height / 2)),
    )

    # Back wall
    wall_back: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WallBack",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 20.0, wall_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-wall_distance, 0.0, wall_height / 2)),
    )

    # Left wall
    wall_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 0.2, wall_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, wall_distance, wall_height / 2)),
    )

    # Right wall
    wall_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WallRight",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 0.2, wall_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -wall_distance, wall_height / 2)),
    )

    # Task parameters
    goal_reached_threshold = 0.5  # Distance to consider goal reached
    max_goal_distance = 5.0  # Maximum distance to spawn goal

    # Reward scales (kept small for stable training)
    distance_to_goal_reward_scale = 0.1
    goal_reached_bonus = 1.0
    survival_bonus = 0.01  # Small reward for staying alive each step
    hover_bonus_scale = 0.05  # Small reward for stable hovering
    lin_vel_penalty_scale = -0.0005
    ang_vel_penalty_scale = -0.0002


class QuadcopterCameraEnv(DirectRLEnv):
    """Quadcopter environment using RGB camera to navigate to visible goal.

    Task: Fly to the red goal sphere using only camera vision.

    Observations:
        - "policy": RGB camera image (64x64x3)

    Actions:
        - Thrust (1D): vertical force
        - Moments (3D): roll, pitch, yaw torques
    """

    cfg: QuadcopterCameraEnvCfg

    def __init__(self, cfg: QuadcopterCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["distance_to_goal", "goal_reached", "lin_vel", "ang_vel"]
        }

        # Robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Goal reached tracking
        self._goals_reached = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Set up scene with drone, camera, goal marker, and isolation walls."""
        # Drone
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Goal marker (visible red sphere)
        self._goal = RigidObject(self.cfg.goal)
        self.scene.rigid_objects["goal"] = self._goal

        # Walls to isolate each environment visually
        self._wall_front = RigidObject(self.cfg.wall_front)
        self._wall_back = RigidObject(self.cfg.wall_back)
        self._wall_left = RigidObject(self.cfg.wall_left)
        self._wall_right = RigidObject(self.cfg.wall_right)
        self.scene.rigid_objects["wall_front"] = self._wall_front
        self.scene.rigid_objects["wall_back"] = self._wall_back
        self.scene.rigid_objects["wall_left"] = self._wall_left
        self.scene.rigid_objects["wall_right"] = self._wall_right

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """Apply forces to drone."""
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """Get camera + state observations (like a real drone with IMU)."""
        # Camera observation
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0

        # Save raw image for debugging (before normalization)
        if self.cfg.write_image_to_file:
            from isaaclab.sensors import save_images_to_file
            save_images_to_file(camera_data, "quadcopter_camera.png")

        # Normalize camera for policy (mean subtraction)
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_normalized = camera_data - mean_tensor

        # State observation (what a real drone IMU provides)
        # Linear velocity in body frame (3D)
        lin_vel = self._robot.data.root_lin_vel_b
        # Angular velocity in body frame (3D)
        ang_vel = self._robot.data.root_ang_vel_b
        # Orientation quaternion (4D) - wxyz format
        quat = self._robot.data.root_quat_w
        # Height above ground (1D)
        height = self._robot.data.root_pos_w[:, 2:3]

        # Concatenate state: [lin_vel(3), ang_vel(3), quat(4), height(1)] = 11
        state = torch.cat([lin_vel, ang_vel, quat, height], dim=1)

        return {"policy": {"camera": camera_normalized.clone(), "state": state.clone()}}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        reward = torch.zeros(self.num_envs, device=self.device)

        # Survival bonus - reward for staying alive
        reward += self.cfg.survival_bonus

        # Distance to goal
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)

        # Distance reward (closer = better)
        distance_reward = 1.0 - torch.tanh(distance_to_goal / 2.0)
        reward += distance_reward * self.cfg.distance_to_goal_reward_scale

        # Goal reached bonus
        goal_reached = distance_to_goal < self.cfg.goal_reached_threshold
        reward[goal_reached] += self.cfg.goal_reached_bonus
        self._goals_reached[goal_reached] += 1

        # Hover bonus - reward for stable flight (low velocity)
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        hover_reward = torch.exp(-0.1 * lin_vel) * self.cfg.hover_bonus_scale
        reward += hover_reward

        # Small velocity penalties
        reward += lin_vel * self.cfg.lin_vel_penalty_scale
        reward += ang_vel * self.cfg.ang_vel_penalty_scale

        # Logging
        self._episode_sums["distance_to_goal"] += distance_reward
        self._episode_sums["goal_reached"] += goal_reached.float() * self.cfg.goal_reached_bonus
        self._episode_sums["lin_vel"] += lin_vel * self.cfg.lin_vel_penalty_scale
        self._episode_sums["ang_vel"] += ang_vel * self.cfg.ang_vel_penalty_scale

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Died if too low, too high, or crashed
        drone_pos = self._robot.data.root_pos_w
        too_low = drone_pos[:, 2] < 0.1
        too_high = drone_pos[:, 2] > 5.0

        # Check if upside down
        projected_gravity = self._robot.data.projected_gravity_b
        upside_down = projected_gravity[:, 2] > 0.5

        died = too_low | too_high | upside_down

        # Goal reached - successful termination (ends episode but not a failure)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        goal_reached = distance_to_goal < self.cfg.goal_reached_threshold

        # Goal reached counts as time_out (not died) so it's not penalized
        time_out = time_out | goal_reached

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance.item()
        extras["Metrics/goals_reached"] = self._goals_reached[env_ids].float().mean().item()
        self.extras["log"].update(extras)

        # Reset
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._goals_reached[env_ids] = 0

        # Reset drone position
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = 1.0  # Start at 1m height
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Spawn goal in front of drone (visible to camera)
        self._spawn_goal(env_ids)

    def _spawn_goal(self, env_ids: torch.Tensor):
        """Spawn goal in front of drone at random position."""
        num_envs = len(env_ids)

        # Random position in front of drone
        # X: 2-5m in front, Y: -2 to 2m side, Z: 0.5-2m height
        goal_state = self._goal.data.default_root_state[env_ids].clone()
        goal_state[:, 0] = torch.zeros(num_envs, device=self.device).uniform_(2.0, 5.0)  # Forward
        goal_state[:, 1] = torch.zeros(num_envs, device=self.device).uniform_(-2.0, 2.0)  # Side
        goal_state[:, 2] = torch.zeros(num_envs, device=self.device).uniform_(0.5, 2.0)  # Height
        goal_state[:, :2] += self._terrain.env_origins[env_ids, :2]

        self._goal.write_root_state_to_sim(goal_state, env_ids)
        self._desired_pos_w[env_ids] = goal_state[:, :3]

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

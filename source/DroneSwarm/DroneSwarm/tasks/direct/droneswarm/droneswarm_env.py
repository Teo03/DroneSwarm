# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DroneSwarm Environment with RGB Camera.

A vision-based drone trash collection task where the drone uses
a front-facing RGB camera plus onboard sensors (IMU, altimeter) as input.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .robots.quadcopter import CRAZYFLIE_CFG


# Visualization marker configs
TRASH_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
    }
)

BIN_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cylinder": sim_utils.CylinderCfg(
            radius=2.0,
            height=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.8)),
        ),
    }
)


@configclass
class DroneSwarmEnvCfg(DirectRLEnvCfg):
    """Configuration for the DroneSwarm environment with RGB camera."""

    # Environment
    episode_length_s = 30.0
    decimation = 2
    action_space = 5  # thrust, roll, pitch, yaw moments + gripper
    # Observation space: camera image + state vector
    # State: lin_vel(3) + ang_vel(3) + quat(4) + height(1) + carrying(1) + gripper(1) = 13
    state_space = 13
    debug_vis = False  # Disable debug markers for speed

    # Simulation - optimized for fast training
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=4,  # Render less frequently (was 2)
        use_fabric=True,  # Enable Fabric for faster data transfer
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        # Balanced rendering - visible but still fast
        render=sim_utils.RenderCfg(
            rendering_mode="balanced",
            enable_shadows=True,  # Helps with depth perception
            enable_ambient_occlusion=False,
            enable_reflections=False,
            enable_global_illumination=False,
            enable_translucency=False,
            enable_dl_denoiser=False,
            antialiasing_mode=None,
        ),
    )

    # Camera configuration - front-facing on the drone
    # The camera is attached to the drone body and looks forward (+X direction)
    # IMPORTANT: clipping_range must be < env_spacing to avoid seeing other environments
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.02),  # Slightly in front and above drone body center
            rot=(0.5, -0.5, 0.5, -0.5),  # Points camera forward (along drone's +X axis)
            convention="ros",  # ROS convention: +Z forward, +X right, +Y down
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=15.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15.0),  # Limited to avoid seeing other environments
        ),
        width=64,  # Balance between speed and visibility
        height=64,
    )
    write_image_to_file = False  # Set True to debug camera view

    # Observation space: hybrid camera + state (like QuadcopterCameraEnv)
    # State: lin_vel(3) + ang_vel(3) + quat(4) + height(1) + carrying(1) + gripper(1) = 13
    observation_space = {
        "camera": [64, 64, 3],
        "state": 13,
    }

    # Viewer settings
    viewer = ViewerCfg(eye=(15.0, 15.0, 10.0))

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
    # env_spacing must be > camera far clipping to prevent seeing other environments
    # grid_size=10, so environments need at least 2*grid_size + buffer spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,  # 256 optimal for vision-based training
        env_spacing=25.0,  # Increased spacing to isolate camera views
        replicate_physics=True,
        filter_collisions=True,
    )

    # Robot configuration
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Trash configuration (rigid body sphere - bright orange/red for visibility)
    trash: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Trash",
        spawn=sim_utils.SphereCfg(
            radius=0.3,  # Bigger for easier visibility
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # Bright orange
                emissive_color=(0.3, 0.15, 0.0),  # Glowing orange
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Bin configuration (static cylinder - bright blue, visible to cameras)
    bin: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bin",
        spawn=sim_utils.CylinderCfg(
            radius=2.0,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Static object, doesn't move
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # Bright blue
                emissive_color=(0.0, 0.0, 0.3),  # Glowing blue
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(5.0, 0.0, 0.25),  # Bin at +5m X, slightly above ground
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Walls to isolate each environment visually (prevent seeing neighbor's objects)
    wall_height = 6.0
    wall_distance = 10.0  # Distance from center

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

    # Drone control parameters
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Task-specific parameters
    grid_size = 10.0
    pickup_radius = 2.0  # Larger for easier pickup
    bin_radius = 2.5
    hover_speed_threshold = 1.0  # More lenient hover requirement
    gripper_close_threshold = 0.4
    gripper_open_threshold = 0.4
    gripper_speed = 0.2

    # Simplified rewards (like working QuadcopterCameraEnv)
    distance_to_target_scale = 0.1  # Main guidance toward target
    pickup_bonus = 1.0  # Bonus when pickup happens
    deposit_bonus = 2.0  # Bonus when deposit happens
    carrying_bonus = 0.02  # Per-step reward while carrying (encourages holding trash)
    hover_above_trash_bonus = 0.05  # Reward for hovering above trash (ready to pickup)
    survival_bonus = 0.01  # Small reward for staying alive
    lin_vel_penalty_scale = -0.0005  # Small velocity penalty
    ang_vel_penalty_scale = -0.0002  # Small angular velocity penalty


class DroneSwarmEnv(DirectRLEnv):
    """DroneSwarm environment with RGB camera vision.

    The drone observes the world through a front-facing RGB camera and
    receives proprioceptive state information that would be available
    on a real drone (IMU velocities, orientation, altitude).

    Observations:
        - "policy": RGB camera image (H, W, 3)
        - "state": Proprioceptive state vector (13D):
            - Linear velocity in body frame (3D) - from IMU
            - Angular velocity in body frame (3D) - from gyroscope
            - Quaternion orientation (4D) - from IMU
            - Height above ground (1D) - from altimeter/barometer
            - Carrying status (1D) - binary flag
            - Gripper state (1D) - continuous 0-1

    Actions:
        - Thrust (1D): vertical force
        - Moments (3D): roll, pitch, yaw torques
        - Gripper (1D): gripper open/close command
    """

    cfg: DroneSwarmEnvCfg

    def __init__(self, cfg: DroneSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Gripper state: 0.0 = open, 1.0 = closed
        self._gripper_state = torch.zeros(self.num_envs, device=self.device)

        # Task state
        self._carrying_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._trash_present = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Bin position
        bin_offset = 5.0
        self._bin_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._bin_pos[:, 0] = bin_offset
        self._bin_pos[:, 1] = 0.0
        self._bin_pos[:, 2] = 0.0

        # Logging (simplified like QuadcopterCameraEnv)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["distance_to_target", "pickup", "deposit", "lin_vel", "ang_vel"]
        }

        # Get robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Verify camera setup
        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The drone camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Set up the scene with drone, camera, trash, bin, and isolation walls."""
        # Drone
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Camera - attached to drone body
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Trash object
        self._trash = RigidObject(self.cfg.trash)
        self.scene.rigid_objects["trash"] = self._trash

        # Bin object (static, visible to cameras)
        self._bin = RigidObject(self.cfg.bin)
        self.scene.rigid_objects["bin"] = self._bin

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

        # CPU collision filtering
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting - important for camera visibility
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Thrust control (action 0): maps [-1, 1] -> [0, thrust_to_weight * weight]
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )

        # Moment control (actions 1-3): roll, pitch, yaw moments
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]

        # Gripper control (action 4): updates gripper state
        gripper_cmd = self._actions[:, 4]
        self._gripper_state = (self._gripper_state + gripper_cmd * self.cfg.gripper_speed).clamp(0.0, 1.0)

    def _apply_action(self):
        """Apply thrust and moment to drone."""
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        # Update carried trash position (teleport to drone)
        if self._carrying_mask.any():
            carrying_envs = self._carrying_mask.nonzero(as_tuple=True)[0]
            drone_pos = self._robot.data.root_pos_w[carrying_envs].clone()
            drone_pos[:, 2] -= 0.3  # Offset trash slightly below drone

            trash_state = self._trash.data.root_state_w[carrying_envs].clone()
            trash_state[:, :3] = drone_pos
            trash_state[:, 7:] = 0.0  # Zero velocity
            self._trash.write_root_state_to_sim(trash_state, carrying_envs)

    def _get_observations(self) -> dict:
        """Compute observations from camera and onboard sensors.

        Returns a dictionary with:
            - "policy": RGB camera image normalized to [0, 1] with mean subtraction
            - "state": Proprioceptive state vector (13D)
        """
        # === Camera observation ===
        data_type = "rgb"
        camera_data = self._tiled_camera.data.output[data_type] / 255.0
        # Normalize camera data for better training
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data = camera_data - mean_tensor

        # === Proprioceptive state (what a real drone would have) ===
        drone_pos = self._robot.data.root_pos_w
        drone_quat = self._robot.data.root_quat_w

        # Get environment origins for height calculation
        env_origins_z = self._terrain.env_origins[:, 2]

        # Build state tensor (13D) - like QuadcopterCameraEnv but with gripper info
        state = torch.cat(
            [
                # Linear velocity (body frame) - from IMU - 3D
                self._robot.data.root_lin_vel_b,
                # Angular velocity (body frame) - from gyroscope - 3D
                self._robot.data.root_ang_vel_b,
                # Quaternion (wxyz) - from IMU attitude estimation - 4D
                drone_quat,
                # Height above ground - from altimeter/barometer - 1D
                (drone_pos[:, 2] - env_origins_z).unsqueeze(-1),
                # Carrying status - from gripper sensor - 1D
                self._carrying_mask.float().unsqueeze(-1),
                # Gripper state - from gripper encoder - 1D
                self._gripper_state.unsqueeze(-1),
            ],
            dim=-1,
        )

        if self.cfg.write_image_to_file:
            from isaaclab.sensors import save_images_to_file
            save_images_to_file(camera_data, "droneswarm_rgb.png")

        # Return nested dict format for hybrid observations (camera + state)
        return {"policy": {"camera": camera_data.clone(), "state": state.clone()}}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards - simplified like QuadcopterCameraEnv."""
        reward = torch.zeros(self.num_envs, device=self.device)

        # Survival bonus
        reward += self.cfg.survival_bonus

        drone_pos = self._robot.data.root_pos_w
        trash_pos = self._trash.data.root_pos_w

        # Bin position in world frame
        bin_pos_world = self._bin_pos.clone()
        bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]

        # Distances
        trash_dist = torch.norm(drone_pos[:, :2] - trash_pos[:, :2], dim=-1)
        bin_dist = torch.norm(drone_pos[:, :2] - bin_pos_world[:, :2], dim=-1)
        height_above_trash = drone_pos[:, 2] - trash_pos[:, 2]

        # Current target: trash if not carrying, bin if carrying
        target_dist = torch.where(self._carrying_mask, bin_dist, trash_dist)

        # === DISTANCE REWARD (main guidance) ===
        distance_reward = 1.0 - torch.tanh(target_dist / 2.0)
        reward += distance_reward * self.cfg.distance_to_target_scale

        # === HOVER ABOVE TRASH BONUS (encourages getting in pickup position) ===
        in_pickup_position = (
            (~self._carrying_mask) &
            (self._trash_present) &
            (trash_dist < self.cfg.pickup_radius) &
            (height_above_trash > 0) &
            (height_above_trash < self.cfg.pickup_radius * 2)
        )
        reward += in_pickup_position.float() * self.cfg.hover_above_trash_bonus

        # === CARRYING BONUS (encourages holding trash while moving to bin) ===
        reward += self._carrying_mask.float() * self.cfg.carrying_bonus

        # === PICKUP LOGIC ===
        can_pickup = (
            (~self._carrying_mask) &
            (self._trash_present) &
            (trash_dist < self.cfg.pickup_radius) &
            (height_above_trash > 0) &
            (height_above_trash < self.cfg.pickup_radius * 2) &
            (self._gripper_state >= self.cfg.gripper_close_threshold)
        )

        if can_pickup.any():
            pickup_envs = can_pickup.nonzero(as_tuple=True)[0]
            self._carrying_mask[pickup_envs] = True
            reward[pickup_envs] += self.cfg.pickup_bonus
            self._episode_sums["pickup"][pickup_envs] += self.cfg.pickup_bonus

        # === DEPOSIT LOGIC ===
        gripper_opened = self._gripper_state < self.cfg.gripper_open_threshold
        can_deposit = self._carrying_mask & gripper_opened & (bin_dist < self.cfg.bin_radius)

        if can_deposit.any():
            deposit_envs = can_deposit.nonzero(as_tuple=True)[0]
            self._carrying_mask[deposit_envs] = False
            self._trash_present[deposit_envs] = False
            reward[deposit_envs] += self.cfg.deposit_bonus
            self._episode_sums["deposit"][deposit_envs] += self.cfg.deposit_bonus

        # Drop outside bin - just respawn (no penalty, keep it simple)
        dropped_outside = self._carrying_mask & gripper_opened & (bin_dist >= self.cfg.bin_radius)
        if dropped_outside.any():
            drop_envs = dropped_outside.nonzero(as_tuple=True)[0]
            self._carrying_mask[drop_envs] = False
            self._respawn_trash(drop_envs)

        # === SMALL VELOCITY PENALTIES ===
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        reward += lin_vel * self.cfg.lin_vel_penalty_scale
        reward += ang_vel * self.cfg.ang_vel_penalty_scale

        # Logging
        self._episode_sums["distance_to_target"] += distance_reward * self.cfg.distance_to_target_scale
        self._episode_sums["lin_vel"] += lin_vel * self.cfg.lin_vel_penalty_scale
        self._episode_sums["ang_vel"] += ang_vel * self.cfg.ang_vel_penalty_scale

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        drone_pos = self._robot.data.root_pos_w
        env_origins = self._terrain.env_origins

        rel_pos = drone_pos - env_origins

        # Out of bounds check
        out_of_bounds = (
            (rel_pos[:, 0] < -self.cfg.grid_size) | (rel_pos[:, 0] > self.cfg.grid_size) |
            (rel_pos[:, 1] < -self.cfg.grid_size) | (rel_pos[:, 1] > self.cfg.grid_size) |
            (drone_pos[:, 2] < 0.0) | (drone_pos[:, 2] > self.cfg.grid_size)
        )

        # Crash detection
        projected_gravity = self._robot.data.projected_gravity_b
        upside_down = projected_gravity[:, 2] > 0.5

        on_ground = drone_pos[:, 2] < 0.15
        nearly_flat = projected_gravity[:, 2] > -0.3
        ground_crash = on_ground & nearly_flat

        died = out_of_bounds | upside_down | ground_crash

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = ~self._trash_present

        return died, time_out | success

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
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
        extras["Metrics/trash_deposited"] = torch.count_nonzero(~self._trash_present[env_ids]).item()
        self.extras["log"].update(extras)

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Spread out resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions and state
        self._actions[env_ids] = 0.0
        self._gripper_state[env_ids] = 0.0
        self._carrying_mask[env_ids] = False
        self._trash_present[env_ids] = True

        # Reset drone position
        margin = 3.0
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(-margin, margin)
        default_root_state[:, 1] = torch.zeros(len(env_ids), device=self.device).uniform_(-margin, margin)
        default_root_state[:, 2] = 1.5
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset trash position
        self._respawn_trash(env_ids)

    def _respawn_trash(self, env_ids: torch.Tensor):
        """Respawn trash at random location."""
        margin = 3.0
        num_envs = len(env_ids)

        trash_state = self._trash.data.default_root_state[env_ids].clone()
        trash_state[:, 0] = torch.zeros(num_envs, device=self.device).uniform_(-margin, margin)
        trash_state[:, 1] = torch.zeros(num_envs, device=self.device).uniform_(-margin, margin)
        trash_state[:, 2] = 0.5
        trash_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        trash_state[:, 3] = 1.0
        trash_state[:, 4:7] = 0.0
        trash_state[:, 7:] = 0.0

        self._trash.write_root_state_to_sim(trash_state, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization."""
        if debug_vis:
            if not hasattr(self, "bin_visualizer"):
                marker_cfg = BIN_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/bin"
                self.bin_visualizer = VisualizationMarkers(marker_cfg)

            bin_pos_world = self._bin_pos.clone()
            bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]
            bin_pos_world[:, 2] = 0.15
            self.bin_visualizer.visualize(bin_pos_world)
            self.bin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "bin_visualizer"):
                self.bin_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "bin_visualizer"):
            bin_pos_world = self._bin_pos.clone()
            bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]
            bin_pos_world[:, 2] = 0.15
            self.bin_visualizer.visualize(bin_pos_world)

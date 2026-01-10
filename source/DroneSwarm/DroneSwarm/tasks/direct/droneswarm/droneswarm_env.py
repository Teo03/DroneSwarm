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
            radius=0.15,  # Match trash size
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
    }
)

BIN_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cylinder": sim_utils.CylinderCfg(
            radius=2.0,
            height=0.25,  # Match bin height
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.8)),
        ),
    }
)


@configclass
class DroneSwarmEnvCfg(DirectRLEnvCfg):
    """Configuration for the DroneSwarm environment with RGB camera."""

    # Environment
    episode_length_s = 10.0  # Short episodes for faster learning
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
            radius=0.15,  # Smaller for easier maneuvering
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
            pos=(2.0, 2.0, 0.2),  # Offset to left side of drone's view
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Bin configuration (static cylinder - bright blue, visible to cameras)
    bin: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bin",
        spawn=sim_utils.CylinderCfg(
            radius=2.0,
            height=0.25,  # Shorter bin for easier deposit
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
            pos=(5.0, -3.0, 0.125),  # Bin offset to the side
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
    pickup_radius = 2.0   # Pickup range - generous
    bin_radius = 2.5
    # Gripper hysteresis: need high value to close/pickup, low value to open/drop
    gripper_close_threshold = 0.5  # Must close gripper to pickup
    gripper_open_threshold = 0.3   # Must open gripper clearly to drop
    gripper_speed = 0.2   # Faster gripper
    pickup_cooldown_steps = 5  # Short cooldown after drop

    # Reward shaping - STRONG incentive to go to bin and deposit FAST
    approach_trash_scale = 1.0    # Reward for moving towards trash (when not carrying)
    approach_bin_scale = 3.0      # STRONG reward for moving towards bin when carrying
    dist_to_trash_scale = 0.2     # Continuous reward for being close to trash
    dist_to_bin_scale = 0.5       # STRONG continuous reward for being close to bin
    pickup_bonus = 2.0            # Bonus when pickup happens
    deposit_bonus = 50.0          # MASSIVE bonus for successful deposit!
    time_bonus_scale = 5.0        # Bonus for finishing fast (multiplied by remaining time ratio)
    carrying_penalty = -0.01      # Small PENALTY per step while carrying (encourages depositing fast!)
    drop_penalty = -5.0           # Bigger penalty for dropping outside bin
    crash_penalty = -10.0         # Penalty for crashing
    spin_penalty_scale = -0.005   # Penalty for spinning/rotating too much (angular velocity)
    survival_bonus = 0.0          # No survival bonus - focus on task completion


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
        self._pickup_cooldown = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        # Previous distances for progress reward
        self._prev_trash_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_bin_dist = torch.zeros(self.num_envs, device=self.device)

        # Bin position - offset to the side, not in line with drone/trash
        self._bin_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._bin_pos[:, 0] = 5.0   # Forward
        self._bin_pos[:, 1] = -3.0  # Offset to right side
        self._bin_pos[:, 2] = 0.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["approach_trash", "approach_bin", "pickup", "deposit", "time_bonus", "drop_penalty"]
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
        # Trash carrying handled in _get_observations (after physics)

    def _get_observations(self) -> dict:
        """Compute observations from camera and onboard sensors.

        Returns a dictionary with:
            - "policy": RGB camera image normalized to [0, 1] with mean subtraction
            - "state": Proprioceptive state vector (13D)
        """
        # === KINEMATIC CARRYING (after physics, Factory env pattern) ===
        # Teleport trash to follow drone - must happen AFTER physics step
        if self._carrying_mask.any():
            carrying_envs = self._carrying_mask.nonzero(as_tuple=True)[0]

            # Target position: below drone
            drone_pos = self._robot.data.root_pos_w[carrying_envs].clone()
            target_pos = drone_pos.clone()
            target_pos[:, 2] -= 0.2  # Hang below drone

            # Identity quaternion (no rotation)
            target_quat = torch.zeros(len(carrying_envs), 4, device=self.device)
            target_quat[:, 0] = 1.0  # w=1

            # Write pose AND zero velocity
            target_pose = torch.cat([target_pos, target_quat], dim=-1)
            self._trash.write_root_pose_to_sim(target_pose, carrying_envs)

            # Zero velocity to prevent physics accumulation
            zero_vel = torch.zeros(len(carrying_envs), 6, device=self.device)
            self._trash.write_root_velocity_to_sim(zero_vel, carrying_envs)

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
        """Compute rewards with proper progress tracking."""
        reward = torch.zeros(self.num_envs, device=self.device)

        # Survival bonus
        reward += self.cfg.survival_bonus

        drone_pos = self._robot.data.root_pos_w
        trash_pos = self._trash.data.root_pos_w

        # Bin position in world frame
        bin_pos_world = self._bin_pos.clone()
        bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]

        # Current distances
        trash_dist = torch.norm(drone_pos[:, :2] - trash_pos[:, :2], dim=-1)
        bin_dist = torch.norm(drone_pos[:, :2] - bin_pos_world[:, :2], dim=-1)
        height_above_trash = drone_pos[:, 2] - trash_pos[:, 2]

        # Decrement pickup cooldown
        self._pickup_cooldown = (self._pickup_cooldown - 1).clamp(min=0)

        not_carrying = ~self._carrying_mask & self._trash_present

        # === PROGRESS REWARDS (for moving towards target) ===
        trash_progress = (self._prev_trash_dist - trash_dist) * not_carrying.float()
        approach_trash_reward = trash_progress * self.cfg.approach_trash_scale
        reward += approach_trash_reward

        bin_progress = (self._prev_bin_dist - bin_dist) * self._carrying_mask.float()
        approach_bin_reward = bin_progress * self.cfg.approach_bin_scale
        reward += approach_bin_reward

        # === CONTINUOUS DISTANCE REWARDS (pulls towards target) ===
        # When not carrying: reward being close to trash
        trash_closeness = (1.0 - torch.tanh(trash_dist / 3.0)) * not_carrying.float()
        reward += trash_closeness * self.cfg.dist_to_trash_scale

        # When carrying: reward being close to bin (THIS IS KEY!)
        bin_closeness = (1.0 - torch.tanh(bin_dist / 5.0)) * self._carrying_mask.float()
        reward += bin_closeness * self.cfg.dist_to_bin_scale

        # === CARRYING PENALTY (encourages depositing fast, not just holding) ===
        carrying_penalty = self._carrying_mask.float() * self.cfg.carrying_penalty
        reward += carrying_penalty  # This is negative!

        # === HEIGHT PENALTY WHEN CARRYING (discourage flying up instead of to bin) ===
        drone_height = drone_pos[:, 2]
        height_penalty = torch.clamp(drone_height - 3.0, min=0.0) * 0.1 * self._carrying_mask.float()
        reward -= height_penalty

        # === SPIN PENALTY (discourage excessive rotation/spinning) ===
        ang_vel = self._robot.data.root_ang_vel_b  # Angular velocity in body frame
        spin_magnitude = torch.sum(torch.square(ang_vel), dim=1)  # Sum of squared angular velocities
        spin_penalty = spin_magnitude * self.cfg.spin_penalty_scale
        reward += spin_penalty  # This is negative!

        # === PICKUP LOGIC (simplified) ===
        # Just need to be close to trash horizontally and above it, with gripper closed
        can_pickup = (
            (~self._carrying_mask) &
            (self._trash_present) &
            (self._pickup_cooldown == 0) &  # Cooldown elapsed
            (trash_dist < self.cfg.pickup_radius) &  # Close enough horizontally
            (height_above_trash > -0.5) &  # Above or near trash
            (height_above_trash < 3.0) &   # Not too high above
            (self._gripper_state >= self.cfg.gripper_close_threshold)  # Gripper closed
        )

        if can_pickup.any():
            pickup_envs = can_pickup.nonzero(as_tuple=True)[0]
            self._carrying_mask[pickup_envs] = True
            reward[pickup_envs] += self.cfg.pickup_bonus
            self._episode_sums["pickup"][pickup_envs] += self.cfg.pickup_bonus

        # === DEPOSIT LOGIC (with hysteresis) ===
        gripper_opened = self._gripper_state < self.cfg.gripper_open_threshold  # Gripper clearly open
        can_deposit = self._carrying_mask & gripper_opened & (bin_dist < self.cfg.bin_radius)

        if can_deposit.any():
            deposit_envs = can_deposit.nonzero(as_tuple=True)[0]
            self._carrying_mask[deposit_envs] = False
            self._trash_present[deposit_envs] = False

            # MASSIVE deposit bonus!
            reward[deposit_envs] += self.cfg.deposit_bonus
            self._episode_sums["deposit"][deposit_envs] += self.cfg.deposit_bonus

            # TIME BONUS: reward for finishing fast (more time remaining = more bonus)
            time_remaining_ratio = 1.0 - (self.episode_length_buf[deposit_envs].float() / self.max_episode_length)
            time_bonus = time_remaining_ratio * self.cfg.time_bonus_scale
            reward[deposit_envs] += time_bonus
            self._episode_sums["time_bonus"][deposit_envs] += time_bonus

        # === DROP OUTSIDE BIN (penalty + respawn) ===
        dropped_outside = self._carrying_mask & gripper_opened & (bin_dist >= self.cfg.bin_radius)
        if dropped_outside.any():
            drop_envs = dropped_outside.nonzero(as_tuple=True)[0]
            self._carrying_mask[drop_envs] = False
            self._pickup_cooldown[drop_envs] = self.cfg.pickup_cooldown_steps  # Start cooldown
            reward[drop_envs] += self.cfg.drop_penalty
            self._episode_sums["drop_penalty"][drop_envs] += self.cfg.drop_penalty
            self._respawn_trash(drop_envs)

        # Update previous distances for next step
        self._prev_trash_dist = trash_dist.clone()
        self._prev_bin_dist = bin_dist.clone()

        # === CRASH PENALTY (detect crashes here, before _get_dones) ===
        env_origins = self._terrain.env_origins
        rel_pos = drone_pos - env_origins
        drone_height = drone_pos[:, 2] - env_origins[:, 2]
        projected_gravity = self._robot.data.projected_gravity_b

        # Crash conditions
        ground_crash = drone_height < 0.1  # Hit the ground
        too_high = drone_height > self.cfg.wall_height  # Above walls
        out_of_bounds = (
            (rel_pos[:, 0] < -self.cfg.grid_size) | (rel_pos[:, 0] > self.cfg.grid_size) |
            (rel_pos[:, 1] < -self.cfg.grid_size) | (rel_pos[:, 1] > self.cfg.grid_size)
        )
        upside_down = projected_gravity[:, 2] > 0.3

        crashed = ground_crash | too_high | out_of_bounds | upside_down
        reward += crashed.float() * self.cfg.crash_penalty

        # Logging
        self._episode_sums["approach_trash"] += approach_trash_reward
        self._episode_sums["approach_bin"] += approach_bin_reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions - crashes end episode immediately."""
        drone_pos = self._robot.data.root_pos_w
        env_origins = self._terrain.env_origins

        rel_pos = drone_pos - env_origins
        drone_height = drone_pos[:, 2] - env_origins[:, 2]

        # Crash conditions
        ground_crash = drone_height < 0.1  # Hit the ground
        too_high = drone_height > self.cfg.wall_height  # Above walls (6m)
        out_of_bounds = (
            (rel_pos[:, 0] < -self.cfg.grid_size) | (rel_pos[:, 0] > self.cfg.grid_size) |
            (rel_pos[:, 1] < -self.cfg.grid_size) | (rel_pos[:, 1] > self.cfg.grid_size)
        )

        # Upside down crash
        projected_gravity = self._robot.data.projected_gravity_b
        upside_down = projected_gravity[:, 2] > 0.3

        # Any crash terminates episode
        crashed = ground_crash | too_high | out_of_bounds | upside_down

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = ~self._trash_present

        # crashed = terminated (bad), time_out|success = truncated (neutral/good)
        return crashed, time_out | success

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
        self._pickup_cooldown[env_ids] = 0

        # Reset drone position - fixed position facing forward
        # Layout: drone at (-3, 0), trash at (2, 2), bin at (5, -3)
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = -3.0  # Fixed X position
        default_root_state[:, 1] = 0.0   # Fixed Y position centered
        default_root_state[:, 2] = 1.5   # Fixed height
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset trash position
        self._respawn_trash(env_ids)

        # Initialize previous distances for progress reward
        drone_pos = self._robot.data.root_pos_w[env_ids]
        trash_pos = self._trash.data.root_pos_w[env_ids]
        bin_pos_world = self._bin_pos[env_ids].clone()
        bin_pos_world[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._prev_trash_dist[env_ids] = torch.norm(drone_pos[:, :2] - trash_pos[:, :2], dim=-1)
        self._prev_bin_dist[env_ids] = torch.norm(drone_pos[:, :2] - bin_pos_world[:, :2], dim=-1)

    def _respawn_trash(self, env_ids: torch.Tensor):
        """Respawn trash at fixed location, offset to the side."""
        trash_state = self._trash.data.default_root_state[env_ids].clone()
        # Layout: drone at (-3, 0), trash at (2, 2), bin at (5, -3)
        # This forces the drone to navigate, not fly straight
        trash_state[:, 0] = 2.0   # Fixed X - in front of drone
        trash_state[:, 1] = 2.0   # Fixed Y - offset to left
        trash_state[:, 2] = 0.2   # Fixed Z - just above ground
        trash_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        trash_state[:, 3] = 1.0   # quat w
        trash_state[:, 4:7] = 0.0 # quat xyz
        trash_state[:, 7:] = 0.0  # velocities

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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone Trash Collection Environment.

A drone must pick up trash objects using a gripper and deposit them in a bin.
Ported from PufferLib's puffer_drone_trash environment to IsaacLab.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

from .drone_trash_env_cfg import DroneTrashEnvCfg


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

GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)


class DroneTrashEnv(DirectRLEnv):
    """Drone trash collection environment.

    The drone must:
    1. Navigate to trash objects
    2. Close gripper to pick up trash (when hovering slowly above it)
    3. Navigate to the bin
    4. Open gripper to deposit trash

    Rewards are given for pickup (+1), deposit (+5), and completion (+10).
    Penalties for dropping trash outside bin (-1) and going out of bounds (-1).
    """

    cfg: DroneTrashEnvCfg

    def __init__(self, cfg: DroneTrashEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._last_dist_reward = torch.zeros(self.num_envs, device=self.device)

        # Bin position (closer to center for easier learning)
        bin_offset = 5.0  # Place bin 5m from center (not at edge)
        self._bin_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._bin_pos[:, 0] = bin_offset  # Place at +X direction
        self._bin_pos[:, 1] = 0.0
        self._bin_pos[:, 2] = 0.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pickup_reward",
                "deposit_reward",
                "distance_shaping",
                "penalties",
            ]
        }

        # Get robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Set up the scene with drone, trash, and bin."""
        # Drone
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Trash object
        self._trash = RigidObject(self.cfg.trash)
        self.scene.rigid_objects["trash"] = self._trash

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments (filter_collisions=True in scene cfg handles GPU collision filtering)
        self.scene.clone_environments(copy_from_source=False)

        # For CPU simulation, manually filter collisions between environments
        # This allows collisions with ground but prevents cross-environment collisions
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
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
            # Offset trash slightly below drone
            drone_pos[:, 2] -= 0.3

            # Update trash position
            trash_state = self._trash.data.root_state_w[carrying_envs].clone()
            trash_state[:, :3] = drone_pos
            trash_state[:, 7:] = 0.0  # Zero velocity
            self._trash.write_root_state_to_sim(trash_state, carrying_envs)

    def _get_observations(self) -> dict:
        """Compute observations."""
        drone_pos = self._robot.data.root_pos_w
        drone_quat = self._robot.data.root_quat_w
        trash_pos = self._trash.data.root_pos_w

        # Direction to trash (in world frame, normalized by grid size)
        to_trash = trash_pos - drone_pos
        to_trash_norm = to_trash / self.cfg.grid_size

        # Direction to bin (account for env origins)
        bin_pos_world = self._bin_pos.clone()
        bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]
        to_bin = bin_pos_world - drone_pos
        to_bin_norm = to_bin / self.cfg.grid_size

        # Distance to target (horizontal only)
        target_is_bin = self._carrying_mask.float().unsqueeze(-1)
        target_pos = target_is_bin * bin_pos_world + (1 - target_is_bin) * trash_pos
        to_target = target_pos - drone_pos
        target_dist_horiz = torch.norm(to_target[:, :2], dim=-1) / self.cfg.grid_size

        # Build observation tensor (23D)
        obs = torch.cat(
            [
                # Linear velocity (body frame) - 3D
                self._robot.data.root_lin_vel_b / self.cfg.max_vel,
                # Angular velocity (body frame) - 3D
                self._robot.data.root_ang_vel_b / self.cfg.max_omega,
                # Quaternion (wxyz) - 4D
                drone_quat,
                # Height - 1D
                drone_pos[:, 2:3] / self.cfg.grid_size,
                # Carrying status - 1D
                self._carrying_mask.float().unsqueeze(-1),
                # Gripper state - 1D
                self._gripper_state.unsqueeze(-1),
                # Direction to trash - 3D (clamped)
                to_trash_norm.clamp(-1.0, 1.0),
                # Direction to bin - 3D (clamped)
                to_bin_norm.clamp(-1.0, 1.0),
                # Distance to target - 1D
                target_dist_horiz.unsqueeze(-1).clamp(0.0, 1.0),
                # Nearest drone direction (placeholder for multi-agent) - 3D
                torch.zeros(self.num_envs, 3, device=self.device),
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on task progress."""
        reward = torch.zeros(self.num_envs, device=self.device)

        drone_pos = self._robot.data.root_pos_w
        drone_vel = self._robot.data.root_lin_vel_w
        trash_pos = self._trash.data.root_pos_w
        speed = torch.norm(drone_vel, dim=-1)

        # Bin position in world frame
        bin_pos_world = self._bin_pos.clone()
        bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]

        # Horizontal distances
        to_trash_horiz = drone_pos[:, :2] - trash_pos[:, :2]
        trash_dist_horiz = torch.norm(to_trash_horiz, dim=-1)
        height_above_trash = drone_pos[:, 2] - trash_pos[:, 2]

        to_bin_horiz = drone_pos[:, :2] - bin_pos_world[:, :2]
        bin_dist_horiz = torch.norm(to_bin_horiz, dim=-1)

        # === PICKUP LOGIC ===
        # Conditions: near trash, above it, moving slowly, gripper closed
        can_pickup = (
            (~self._carrying_mask) &
            (self._trash_present) &
            (trash_dist_horiz < self.cfg.pickup_radius) &
            (height_above_trash > 0) &
            (height_above_trash < self.cfg.pickup_radius * 2) &
            (speed < self.cfg.hover_speed_threshold) &
            (self._gripper_state >= self.cfg.gripper_close_threshold)
        )

        if can_pickup.any():
            pickup_envs = can_pickup.nonzero(as_tuple=True)[0]
            self._carrying_mask[pickup_envs] = True
            self._last_dist_reward[pickup_envs] = 0.0
            reward[pickup_envs] += self.cfg.pickup_reward
            self._episode_sums["pickup_reward"][pickup_envs] += self.cfg.pickup_reward

        # === DEPOSIT / DROP LOGIC ===
        # Only when carrying and gripper is opened
        gripper_opened = self._gripper_state < self.cfg.gripper_open_threshold
        can_release = self._carrying_mask & gripper_opened

        if can_release.any():
            release_envs = can_release.nonzero(as_tuple=True)[0]

            # Check if over bin
            over_bin = bin_dist_horiz[release_envs] < self.cfg.bin_radius

            # Successful deposit
            deposit_envs = release_envs[over_bin]
            if len(deposit_envs) > 0:
                self._carrying_mask[deposit_envs] = False
                self._trash_present[deposit_envs] = False
                self._last_dist_reward[deposit_envs] = 0.0
                reward[deposit_envs] += self.cfg.deposit_reward
                self._episode_sums["deposit_reward"][deposit_envs] += self.cfg.deposit_reward

                # Completion bonus (all trash deposited)
                all_deposited = ~self._trash_present[deposit_envs]
                reward[deposit_envs[all_deposited]] += self.cfg.completion_reward

            # Dropped outside bin - respawn trash
            drop_envs = release_envs[~over_bin]
            if len(drop_envs) > 0:
                self._carrying_mask[drop_envs] = False
                self._last_dist_reward[drop_envs] = 0.0
                reward[drop_envs] += self.cfg.drop_penalty
                self._episode_sums["penalties"][drop_envs] += abs(self.cfg.drop_penalty)

                # Respawn trash at random location
                self._respawn_trash(drop_envs)

        # === DISTANCE SHAPING ===
        # Target is bin if carrying, otherwise trash
        target_pos = torch.where(
            self._carrying_mask.unsqueeze(-1).expand(-1, 3),
            bin_pos_world,
            trash_pos,
        )
        to_target = target_pos - drone_pos
        target_dist = torch.norm(to_target[:, :2], dim=-1)  # Horizontal distance

        dist_reward = torch.exp(-0.5 * target_dist)
        shaping = dist_reward - self._last_dist_reward
        self._last_dist_reward = dist_reward.clone()

        # Only apply shaping when trash is present
        shaping = shaping * self._trash_present.float()
        reward += shaping
        self._episode_sums["distance_shaping"] += shaping

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        drone_pos = self._robot.data.root_pos_w
        env_origins = self._terrain.env_origins

        # Relative position to environment origin
        rel_pos = drone_pos - env_origins

        # Out of bounds check
        out_of_bounds = (
            (rel_pos[:, 0] < -self.cfg.grid_size) | (rel_pos[:, 0] > self.cfg.grid_size) |
            (rel_pos[:, 1] < -self.cfg.grid_size) | (rel_pos[:, 1] > self.cfg.grid_size) |
            (drone_pos[:, 2] < 0.0) | (drone_pos[:, 2] > self.cfg.grid_size)
        )

        # Crash detection: drone is tilted too much (upside down)
        # projected_gravity_b gives gravity vector in body frame
        # If drone is upright, projected_gravity_b[:, 2] should be close to -1 (gravity points down in body frame)
        # If drone is upside down, projected_gravity_b[:, 2] > 0
        projected_gravity = self._robot.data.projected_gravity_b

        # Only crash if completely upside down (tilted > 120 degrees) - more forgiving
        upside_down = projected_gravity[:, 2] > 0.5  # ~120 degrees tilt

        # Ground crash: very low AND nearly horizontal (stuck on ground)
        on_ground = drone_pos[:, 2] < 0.15
        nearly_flat = projected_gravity[:, 2] > -0.3  # ~70+ degrees tilt
        ground_crash = on_ground & nearly_flat

        # Combine termination conditions
        died = out_of_bounds | upside_down | ground_crash

        # Apply penalty
        if died.any():
            died_envs = died.nonzero(as_tuple=True)[0]
            self._episode_sums["penalties"][died_envs] += abs(self.cfg.out_of_bounds_penalty)

        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Success (all trash deposited)
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

        # Count termination types
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

        # Reset actions
        self._actions[env_ids] = 0.0

        # Reset gripper state
        self._gripper_state[env_ids] = 0.0

        # Reset task state
        self._carrying_mask[env_ids] = False
        self._trash_present[env_ids] = True
        self._last_dist_reward[env_ids] = 0.0

        # Reset drone position (higher up for more maneuverability)
        margin = 3.0  # Start near center for easier learning
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(-margin, margin)
        default_root_state[:, 1] = torch.zeros(len(env_ids), device=self.device).uniform_(-margin, margin)
        default_root_state[:, 2] = 1.5  # Start higher for stability
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
        """Respawn trash at random location on ground (near center for easier pickup)."""
        margin = 3.0  # Spawn trash close to center
        num_envs = len(env_ids)

        trash_state = self._trash.data.default_root_state[env_ids].clone()
        trash_state[:, 0] = torch.zeros(num_envs, device=self.device).uniform_(-margin, margin)
        trash_state[:, 1] = torch.zeros(num_envs, device=self.device).uniform_(-margin, margin)
        trash_state[:, 2] = 0.5  # Ground level
        trash_state[:, :2] += self._terrain.env_origins[env_ids, :2]

        # Identity quaternion
        trash_state[:, 3] = 1.0
        trash_state[:, 4:7] = 0.0
        trash_state[:, 7:] = 0.0  # Zero velocity

        self._trash.write_root_state_to_sim(trash_state, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization."""
        if debug_vis:
            if not hasattr(self, "bin_visualizer"):
                # Bin marker
                marker_cfg = BIN_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/bin"
                self.bin_visualizer = VisualizationMarkers(marker_cfg)

            # Update bin positions
            bin_pos_world = self._bin_pos.clone()
            bin_pos_world[:, :2] += self._terrain.env_origins[:, :2]
            bin_pos_world[:, 2] = 0.15  # Half height
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

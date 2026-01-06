# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DroneSwarm Environment.

Includes:
- DroneSwarm: Trash collection with gripper (complex)
- QuadcopterCamera: Fly to visible goal (simpler, for testing vision)
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Original trash collection task
gym.register(
    id="Template-DroneSwarm-Direct-v0",
    entry_point=f"{__name__}.droneswarm_env:DroneSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.droneswarm_env:DroneSwarmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DroneSwarmPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Simpler camera-based goal reaching task
gym.register(
    id="Template-QuadcopterCamera-Direct-v0",
    entry_point=f"{__name__}.quadcopter_camera_env:QuadcopterCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_camera_env:QuadcopterCameraEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

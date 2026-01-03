# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Drone Trash Collection Environment.

A drone must pick up trash objects using a gripper and deposit them in a bin.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-DroneSwarm-Trash-Direct-v0",
    entry_point=f"{__name__}.drone_trash_env:DroneTrashEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_trash_env_cfg:DroneTrashEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DroneTrashPPORunnerCfg",
    },
)

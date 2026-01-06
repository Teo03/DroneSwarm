# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for DroneSwarm environment.

NOTE: RSL-RL does not natively support CNN-based policies for image observations.
For camera-based training, it is recommended to use skrl with the
skrl_ppo_cfg.yaml configuration instead.

This config is provided for API compatibility but will use MLP on flattened
image data, which is not recommended for vision-based learning.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DroneSwarmPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner config for DroneSwarm environment.

    WARNING: This config uses MLP on flattened images which is inefficient.
    Use skrl with CNN policy for proper vision-based learning:
        python scripts/skrl/train.py --task Template-DroneSwarm-Direct-v0

    To set a custom run name, use:
        python scripts/rsl_rl/train.py --task Template-DroneSwarm-Direct-v0 run_name=my_experiment
    """

    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 100
    experiment_name = "droneswarm_direct"

    # Wandb logging configuration
    logger = "wandb"
    wandb_project = "DroneSwarm"
    run_name = ""  # Set via CLI: run_name=my_experiment

    # WARNING: MLP policy on flattened 84x84x3 = 21168 inputs is not ideal
    # This is provided for compatibility only
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,  # Don't normalize image data
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],  # Larger network for high-dim input
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

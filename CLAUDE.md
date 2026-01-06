# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DroneSwarm is an Isaac Lab extension for training drones to collect trash using reinforcement learning. It features a vision-based environment where a Crazyflie quadcopter uses RGB camera input to locate trash objects and deposit them in a bin.

The environment is registered as `Template-DroneSwarm-Direct-v0` and uses direct RL (not manager-based).

## Build and Development Commands

**Conda environment:** `env_isaaclab`

```bash
# Activate the conda environment
conda activate env_isaaclab

# Install the extension (requires Isaac Lab in Python environment)
python -m pip install -e source/DroneSwarm

# List available tasks
python scripts/list_envs.py

# Train with RSL-RL (MLP policy - not recommended for vision)
python scripts/rsl_rl/train.py --task=Template-DroneSwarm-Direct-v0 --enable_cameras

# Train with skrl (CNN policy - recommended for vision-based training)
python scripts/skrl/train.py --task=Template-DroneSwarm-Direct-v0 --enable_cameras

# Play/evaluate trained policy (use --num_envs=1 for single-env evaluation)
python scripts/rsl_rl/play.py --task=Template-DroneSwarm-Direct-v0 --enable_cameras --num_envs=1
python scripts/skrl/play.py --task=Template-DroneSwarm-Direct-v0 --enable_cameras --num_envs=1

# Test with dummy agents
python scripts/zero_agent.py --task=Template-DroneSwarm-Direct-v0 --enable_cameras
python scripts/random_agent.py --task=Template-DroneSwarm-Direct-v0 --enable_cameras

# Code formatting
pre-commit run --all-files
```

### Quick Start (copy-paste ready)
```bash
# Quick test (10k timesteps)
conda activate env_isaaclab && python scripts/skrl/train.py --task=Template-DroneSwarm-Direct-v0 --headless --enable_cameras --num_envs=256 +trainer.timesteps=10000 agent.agent.experiment.experiment_name=my_experiment

# Medium run (50k timesteps)
conda activate env_isaaclab && python scripts/skrl/train.py --task=Template-DroneSwarm-Direct-v0 --headless --enable_cameras --num_envs=256 +trainer.timesteps=50000 agent.agent.experiment.experiment_name=my_experiment

# Full training (200k timesteps)
conda activate env_isaaclab && python scripts/skrl/train.py --task=Template-DroneSwarm-Direct-v0 --headless --enable_cameras --num_envs=256 +trainer.timesteps=200000 agent.agent.experiment.experiment_name=my_experiment
```

**Note:** For vision-based training, `--num_envs=256` is optimal. Camera rendering is expensive - more envs increases time significantly without proportional benefit.

### Common Training Arguments
- `--headless` - Run without GUI (faster training)
- `--num_envs=N` - Override number of parallel environments
- `+trainer.timesteps=N` - Set training timesteps (e.g., `+trainer.timesteps=100000`)
- `--video` - Record training videos
- `--seed=N` - Set random seed
- `agent.agent.experiment.experiment_name=my_run` (skrl) or `run_name=my_experiment` (RSL-RL) - Set experiment name

### Timesteps Reference

| Concept | Description |
|---------|-------------|
| **timesteps** | Total policy update steps (set via `+trainer.timesteps=N`) |
| **num_envs** | Parallel environments (set via `--num_envs=N`) |
| **total_env_steps** | `timesteps × num_envs` = actual environment interactions |

**Example configurations (with 256 envs - optimal for vision):**

| Use Case | Timesteps | Total env steps |
|----------|-----------|-----------------|
| Quick test | `+trainer.timesteps=10000` | 2.5M env steps |
| Short run | `+trainer.timesteps=50000` | 12.8M env steps |
| Medium run | `+trainer.timesteps=100000` | 25.6M env steps |
| Long run | `+trainer.timesteps=200000` | 51.2M env steps |
| Full training | `+trainer.timesteps=500000` | 128M env steps |

**Recommended for vision-based RL:** Start with 50k-100k timesteps to assess learning, scale to 500k+ for full training.

## Architecture

### Environment Structure
```
source/DroneSwarm/DroneSwarm/tasks/direct/droneswarm/
├── __init__.py              # Gym environment registration
├── droneswarm_env.py        # Main environment (DroneSwarmEnv, DroneSwarmEnvCfg)
├── robots/quadcopter.py     # Crazyflie configuration (CRAZYFLIE_CFG)
└── agents/
    ├── rsl_rl_ppo_cfg.py    # RSL-RL PPO config (MLP, not ideal for vision)
    └── skrl_ppo_cfg.yaml    # skrl PPO config (CNN, recommended)
```

### Key Environment Details

**Observations:**
- `"policy"`: RGB camera image (64x64x3), normalized with mean subtraction
- `"state"`: Proprioceptive vector (13D): linear vel, angular vel, quaternion, height, carrying flag, gripper state

**Actions (5D):**
- Thrust (vertical force)
- Roll, pitch, yaw moments (3D)
- Gripper command (open/close)

**Task Flow:**
1. Drone navigates to trash using camera vision
2. Gripper closes to pick up trash when hovering over it
3. Drone carries trash to bin location
4. Gripper opens to deposit trash

### Training Frameworks

- **skrl with CNN**: Recommended for vision-based learning. Uses convolutional feature extractor defined in `skrl_ppo_cfg.yaml`.
- **RSL-RL with MLP**: Provided for API compatibility but flattens images to MLP input (inefficient).

### Logs and Outputs
- **Training logs**: `logs/rsl_rl/` or `logs/skrl/`
- **Hydra configs**: `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`
- **Wandb project**: "DroneSwarm"

#### Analyzing Training Logs

Log directory structure:
```
logs/skrl/droneswarm_direct/<run_name>/
├── checkpoints/          # Model checkpoints
├── events.out.tfevents.* # TensorBoard events file
└── params/               # env.yaml and agent.yaml configs
```

**Quick check with script:**
```bash
./scripts/check_training.sh                    # Check most recent run
./scripts/check_training.sh <run_name>         # Check specific run
./scripts/check_training.sh --list             # List available runs with step counts
./scripts/check_training.sh -v <run_name>      # Verbose: more metrics, stats (min/max/mean/std)
./scripts/check_training.sh -vv <run_name>     # Very verbose: all logged data points
./scripts/check_training.sh -p 15 <run_name>   # Show 15 data points instead of 8
./scripts/check_training.sh -r 5000:8000 <run> # Filter to step range 5000-8000
./scripts/check_training.sh -m <run_name>      # Show all available metric names
```

**View logs with TensorBoard:**
```bash
tensorboard --logdir=logs/skrl/droneswarm_direct/<run_name>/
```

**Extract metrics programmatically:**
```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/skrl/droneswarm_direct/<run_name>/')
ea.Reload()

# Available metrics
print(ea.Tags()['scalars'])

# Key metrics to check:
# - Reward / Total reward (mean)     - overall performance
# - Info / Episode_Reward/pickup_reward   - trash pickup rate
# - Info / Episode_Reward/deposit_reward  - successful deposits
# - Info / Episode_Reward/penalties       - crashes/out-of-bounds
# - Loss / Policy loss, Value loss        - training stability

for event in ea.Scalars('Reward / Total reward (mean)'):
    print(f'Step {event.step}: {event.value:.4f}')
```

## Training Workflow

### 1. Run Training
Start a training run with descriptive name and timesteps:
```bash
conda activate env_isaaclab && python scripts/skrl/train.py --task=Template-DroneSwarm-Direct-v0 --headless --enable_cameras \
    --num_envs=256 +trainer.timesteps=50000 agent.agent.experiment.experiment_name=my_experiment
```

Either run the command in background or provide the command to the user and wait for them to indicate when training is complete and results should be checked.

### 2. Check Results
Use the check script to analyze training progress:
```bash
./scripts/check_training.sh my_experiment
./scripts/check_training.sh -v my_experiment  # For more detail
```

Key metrics to evaluate:
- **Pickup reward**: Is the drone learning to find and pick up trash?
- **Deposit reward**: Is the drone successfully depositing trash in the bin?
- **Penalties**: Are crashes/out-of-bounds events decreasing?
- **Total reward trend**: Overall improvement direction

### 3. Iterate and Improve
Based on results, modify:
- **Reward weights** in `droneswarm_env.py` (pickup_reward_scale, deposit_reward_scale, etc.)
- **Network architecture** in `skrl_ppo_cfg.yaml`
- **PPO hyperparameters** (learning_rate, mini_batches, etc.)
- **Environment parameters** (num_envs, episode length, observation space)

## Code Style

- Line length: 120 characters (Black)
- Import sorting: isort with black profile
- Python 3.10+ syntax enforced by pyupgrade
- BSD-3-Clause license headers required

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

### Common Training Arguments
- `--num_envs=N` - Override number of parallel environments
- `--max_iterations=N` - Set training iterations
- `--video` - Record training videos
- `--seed=N` - Set random seed
- `run_name=my_experiment` (RSL-RL) or `agent.agent.experiment.experiment_name=my_run` (skrl) - Set experiment name

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

## Code Style

- Line length: 120 characters (Black)
- Import sorting: isort with black profile
- Python 3.10+ syntax enforced by pyupgrade
- BSD-3-Clause license headers required

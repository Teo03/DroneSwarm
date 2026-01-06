#!/bin/bash
# Check training progress for DroneSwarm runs
# Usage: ./scripts/check_training.sh [options] [run_name]
#
# Options:
#   -l, --list          List available runs
#   -v, --verbose       Show more details (more data points, stats)
#   -vv                 Very verbose (all logged points for key metrics)
#   -p, --points N      Number of data points to show (default: 8)
#   -m, --metrics       Show all available metrics
#   -r, --range S:E     Show data between step S and E (e.g., 1000:5000)
#   -h, --help          Show this help

LOGS_DIR="logs/skrl/droneswarm_direct"
VERBOSE=0
NUM_POINTS=8
SHOW_ALL_METRICS=0
RANGE_START=0
RANGE_END=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--list)
            echo "Available runs (most recent first):"
            echo ""
            for run in $(ls -1t "$LOGS_DIR" 2>/dev/null); do
                event_file=$(ls "$LOGS_DIR/$run"/events.out.tfevents.* 2>/dev/null | head -1)
                if [[ -n "$event_file" ]]; then
                    steps=$(python3 << PYEOF 2>/dev/null
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('$LOGS_DIR/$run')
ea.Reload()
tags = ea.Tags().get('scalars', [])
if 'Reward / Total reward (mean)' in tags:
    data = ea.Scalars('Reward / Total reward (mean)')
    print(f'{data[-1].step:,}' if data else '0')
else:
    print('0')
PYEOF
)
                    echo "  $run  (${steps:-0} steps)"
                fi
            done
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=1
            NUM_POINTS=12
            shift
            ;;
        -vv)
            VERBOSE=2
            NUM_POINTS=20
            shift
            ;;
        -p|--points)
            NUM_POINTS="$2"
            shift 2
            ;;
        -m|--metrics)
            SHOW_ALL_METRICS=1
            shift
            ;;
        -r|--range)
            RANGE_START=$(echo "$2" | cut -d: -f1)
            RANGE_END=$(echo "$2" | cut -d: -f2)
            shift 2
            ;;
        -h|--help)
            head -12 "$0" | tail -11
            exit 0
            ;;
        *)
            RUN_NAME="$1"
            shift
            ;;
    esac
done

# Get run name (default to most recent)
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME=$(ls -1t "$LOGS_DIR" 2>/dev/null | head -1)
    if [[ -z "$RUN_NAME" ]]; then
        echo "No runs found in $LOGS_DIR"
        exit 1
    fi
    echo "Using most recent run: $RUN_NAME"
fi

RUN_DIR="$LOGS_DIR/$RUN_NAME"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Run not found: $RUN_DIR"
    echo "Use --list to see available runs"
    exit 1
fi

# Export for Python
export RUN_DIR
export RUN_NAME
export VERBOSE
export NUM_POINTS
export SHOW_ALL_METRICS
export RANGE_START
export RANGE_END

# Python script to extract and display metrics
python3 << 'PYTHON_SCRIPT'
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorboard.backend.event_processing import event_accumulator
import statistics

run_dir = os.environ.get('RUN_DIR', '')
run_name = os.environ.get('RUN_NAME', '')
verbose = int(os.environ.get('VERBOSE', 0))
num_points = int(os.environ.get('NUM_POINTS', 8))
show_all_metrics = int(os.environ.get('SHOW_ALL_METRICS', 0))
range_start = int(os.environ.get('RANGE_START', 0))
range_end = int(os.environ.get('RANGE_END', 0))

ea = event_accumulator.EventAccumulator(run_dir)
ea.Reload()

scalars = ea.Tags().get('scalars', [])
if not scalars:
    print("No training data found yet")
    sys.exit(0)

def get_metric_data(tag, start=0, end=0):
    """Get all values for a metric, optionally filtered by step range"""
    try:
        events = ea.Scalars(tag)
        if not events:
            return None
        data = [(e.step, e.value) for e in events]
        if start > 0 or end > 0:
            if end == 0:
                end = float('inf')
            data = [(s, v) for s, v in data if start <= s <= end]
        return data
    except:
        return None

def sample_points(data, n=10):
    """Sample evenly spaced points from data"""
    if not data or len(data) <= n:
        return data
    step = len(data) / n
    indices = [int(i * step) for i in range(n)]
    indices[-1] = len(data) - 1
    return [data[i] for i in indices]

def trend_arrow(values):
    """Return trend indicator"""
    if len(values) < 2:
        return "→"
    start, end = values[0], values[-1]
    if end > start * 1.05:
        return "↑"
    elif end < start * 0.95:
        return "↓"
    return "→"

def spark_line(values, width=40):
    """Create ASCII spark line"""
    if not values or len(values) < 2:
        return ""
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return "─" * width
    chars = "▁▂▃▄▅▆▇█"
    line = ""
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    for v in sampled:
        idx = int((v - min_v) / (max_v - min_v) * (len(chars) - 1))
        line += chars[idx]
    return line

def format_num(v, width=9):
    """Smart number formatting"""
    if abs(v) < 0.0001 and v != 0:
        return f"{v:>{width}.2e}"
    elif abs(v) < 0.01:
        return f"{v:>{width}.5f}"
    elif abs(v) < 1:
        return f"{v:>{width}.4f}"
    elif abs(v) < 100:
        return f"{v:>{width}.3f}"
    else:
        return f"{v:>{width}.1f}"

def calc_stats(values):
    """Calculate statistics"""
    if not values:
        return {}
    return {
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'start': values[0],
        'end': values[-1],
        'change': ((values[-1] - values[0]) / abs(values[0]) * 100) if values[0] != 0 else 0
    }

# Header
width = 80 if verbose else 70
print(f"\n{'='*width}")
print(f"  TRAINING PROGRESS: {run_name}")
print(f"{'='*width}")

# Get basic info
reward_data = get_metric_data('Reward / Total reward (mean)', range_start, range_end)
if reward_data:
    total_steps = reward_data[-1][0]
    logged = len(reward_data)
    range_info = f" (filtered: {range_start}-{range_end})" if range_start > 0 or range_end > 0 else ""
    print(f"  Steps: {total_steps:,} | Logged points: {logged}{range_info}")

# Show all available metrics if requested
if show_all_metrics:
    print(f"\n  Available metrics ({len(scalars)}):")
    for s in sorted(scalars):
        data = get_metric_data(s)
        if data:
            print(f"    {s}: {len(data)} points")
    print()

print(f"{'='*width}\n")

# Define metrics to show
core_metrics = [
    ('Reward / Total reward (mean)', 'Total Reward'),
    ('Info / Episode_Reward/pickup_reward', 'Pickup'),
    ('Info / Episode_Reward/deposit_reward', 'Deposit'),
    ('Info / Episode_Reward/penalties', 'Penalties'),
]

extra_metrics = [
    ('Reward / Total reward (max)', 'Reward (max)'),
    ('Reward / Total reward (min)', 'Reward (min)'),
    ('Info / Episode_Reward/distance_shaping', 'Dist Shaping'),
    ('Loss / Policy loss', 'Policy Loss'),
    ('Loss / Value loss', 'Value Loss'),
    ('Loss / Entropy loss', 'Entropy Loss'),
    ('Policy / Standard deviation', 'Policy Std'),
    ('Learning / Learning rate', 'Learn Rate'),
    ('Episode / Total timesteps (mean)', 'Ep Length'),
]

metrics = core_metrics + (extra_metrics if verbose >= 1 else [])

# Get sample data for header
sample_data = get_metric_data('Reward / Total reward (mean)', range_start, range_end)
if not sample_data:
    print("No data in specified range")
    sys.exit(0)

sampled = sample_points(sample_data, num_points)
steps = [s[0] for s in sampled]

# Print metric table
for tag, name in metrics:
    data = get_metric_data(tag, range_start, range_end)
    if not data:
        continue

    values = [v for _, v in data]
    stats = calc_stats(values)
    sampled_data = sample_points(data, num_points)
    sampled_values = [v for _, v in sampled_data]
    sampled_steps = [s for s, _ in sampled_data]

    # Metric name and trend
    trend = trend_arrow(values)
    change_str = f"{stats['change']:+.1f}%" if abs(stats['change']) < 1000 else ">>>"
    print(f"  {name} {trend} ({change_str})")

    # Sparkline
    spark = spark_line(values, 60 if verbose else 50)
    print(f"  {spark}")

    # Data points header
    if verbose >= 2:
        # Show all points in very verbose mode
        print(f"  {'Step':<8} {'Value':>12}")
        print(f"  {'-'*22}")
        for step, val in data:
            print(f"  {step:<8} {format_num(val)}")
    else:
        # Show sampled points
        row = "  "
        for step in sampled_steps:
            row += f"{step:>9}"
        print(row)

        row = "  "
        for val in sampled_values:
            row += format_num(val)
        print(row)

    # Statistics in verbose mode
    if verbose >= 1:
        print(f"  Stats: min={format_num(stats['min']).strip()} max={format_num(stats['max']).strip()} "
              f"mean={format_num(stats['mean']).strip()} std={format_num(stats['std']).strip()}")

    print()

print(f"{'='*width}")

# Assessment
print("\n  ASSESSMENT:")

pickup_data = get_metric_data('Info / Episode_Reward/pickup_reward', range_start, range_end)
deposit_data = get_metric_data('Info / Episode_Reward/deposit_reward', range_start, range_end)
penalties_data = get_metric_data('Info / Episode_Reward/penalties', range_start, range_end)

if pickup_data:
    pickup_values = [v for _, v in pickup_data]
    pickup_stats = calc_stats(pickup_values)
    if pickup_stats['end'] > pickup_stats['start'] * 1.5:
        print(f"  ✓ Pickup improving: {pickup_stats['start']:.5f} → {pickup_stats['end']:.5f} ({pickup_stats['change']:+.1f}%)")
    else:
        print(f"  ○ Pickup stable: {pickup_stats['start']:.5f} → {pickup_stats['end']:.5f} ({pickup_stats['change']:+.1f}%)")

if deposit_data:
    deposit_values = [v for _, v in deposit_data]
    deposit_stats = calc_stats(deposit_values)
    if deposit_stats['max'] > 0.001:
        print(f"  ✓ Some deposits! max={deposit_stats['max']:.5f}")
    else:
        print(f"  ✗ Not depositing yet (max={deposit_stats['max']:.6f})")

if penalties_data:
    penalties_values = [v for _, v in penalties_data]
    penalties_stats = calc_stats(penalties_values)
    if penalties_stats['end'] < penalties_stats['start'] * 0.8:
        print(f"  ✓ Fewer crashes: {penalties_stats['start']:.5f} → {penalties_stats['end']:.5f} ({penalties_stats['change']:+.1f}%)")
    else:
        print(f"  ○ Still crashing: {penalties_stats['start']:.5f} → {penalties_stats['end']:.5f} ({penalties_stats['change']:+.1f}%)")

if reward_data:
    reward_values = [v for _, v in reward_data]
    reward_stats = calc_stats(reward_values)
    if reward_stats['end'] > reward_stats['start'] * 1.1:
        print(f"  ✓ Reward improving: {reward_stats['start']:.4f} → {reward_stats['end']:.4f} ({reward_stats['change']:+.1f}%)")
    else:
        print(f"  ○ Reward stable: {reward_stats['start']:.4f} → {reward_stats['end']:.4f} ({reward_stats['change']:+.1f}%)")

# Verbose assessment
if verbose >= 1:
    print("\n  DETAILED NOTES:")

    # Check for training stability
    policy_loss = get_metric_data('Loss / Policy loss', range_start, range_end)
    if policy_loss:
        pl_values = [v for _, v in policy_loss]
        pl_std = statistics.stdev(pl_values) if len(pl_values) > 1 else 0
        if pl_std < 0.01:
            print(f"  • Policy loss stable (std={pl_std:.4f})")
        else:
            print(f"  • Policy loss varying (std={pl_std:.4f})")

    # Check policy std
    policy_std = get_metric_data('Policy / Standard deviation', range_start, range_end)
    if policy_std:
        ps_values = [v for _, v in policy_std]
        if ps_values[-1] < 0.3:
            print(f"  • Policy converging (std={ps_values[-1]:.3f}) - may be exploiting")
        elif ps_values[-1] > 0.8:
            print(f"  • Policy still exploring (std={ps_values[-1]:.3f})")
        else:
            print(f"  • Policy exploration moderate (std={ps_values[-1]:.3f})")

    # Training speed estimate
    if reward_data and len(reward_data) > 1:
        first_step = reward_data[0][0]
        last_step = reward_data[-1][0]
        # Rough estimate based on usual logging
        print(f"  • Training range: step {first_step:,} to {last_step:,}")

print()
PYTHON_SCRIPT

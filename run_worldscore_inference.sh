#!/bin/bash
# Thin wrapper around scripts/run_worldscore_batch.py — keeps the call site
# uniform with lyra/Lyra-2 and lingbot-world worldscore adapters. All actual
# work (action-string → poses_c2w, yaml gen, infer.py batching, gen.mp4
# renaming into the baseline_hywp layout) lives in the Python script.
#
# Usage:
#   bash run_worldscore_inference.sh \
#       [--instances PATH] [--worldscore-root PATH] [--output PATH] \
#       [--split test|train] [--names "n1 n2 ..."] [--trajs "t1 t2 ..."] \
#       [--forward-speed 0.08] [--frames-per-iter 61] [--device cuda:0] \
#       [--system-config configs/infer_system_config_few_step_14B.yaml]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LW_DIR="$SCRIPT_DIR"
DYNAMIC_WM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTANCES_JSON="$DYNAMIC_WM_ROOT/wm-agent/datasets/worldscore.json"
WORLDSCORE_OUTPUT_ROOT="$DYNAMIC_WM_ROOT/wm-agent/outputs/worldscore"
OUTPUT_DIR="$DYNAMIC_WM_ROOT/wm-agent/outputs/baseline_liveworld"
SPLIT=test
NAMES_FILTER=""
TRAJS_FILTER=""
FORWARD_SPEED=0.08
FRAMES_PER_ITER=61
DEVICE=cuda:0
SYSTEM_CONFIG="configs/infer_system_config_few_step_14B.yaml"

CONDA_ENV=liveworld
ENV_PREFIX=/data/harry/.miniconda3/envs/$CONDA_ENV
PY=$ENV_PREFIX/bin/python
# Put the env's bin dir on PATH so subprocesses (LiveWorld calls `ffmpeg`
# via shutil.which → subprocess) can find env-local binaries.
export PATH="$ENV_PREFIX/bin:$PATH"

# Pass-through CLI parsing (forward everything to the Python script).
EXTRA=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --instances)        INSTANCES_JSON=$2; shift ;;
        --worldscore-root)  WORLDSCORE_OUTPUT_ROOT=$2; shift ;;
        --output)           OUTPUT_DIR=$2; shift ;;
        --split)            SPLIT=$2; shift ;;
        --names)            NAMES_FILTER=$2; shift ;;
        --trajs)            TRAJS_FILTER=$2; shift ;;
        --forward-speed)    FORWARD_SPEED=$2; shift ;;
        --frames-per-iter)  FRAMES_PER_ITER=$2; shift ;;
        --device)           DEVICE=$2; shift ;;
        --system-config)    SYSTEM_CONFIG=$2; shift ;;
        --gpus)
            # shorthand: --gpus 0  →  --device cuda:0  (LiveWorld uses one GPU)
            first=$(awk -F, '{print $1}' <<< "$2")
            DEVICE="cuda:$first"; shift ;;
        --prepare-only)     EXTRA+=("--prepare-only") ;;
        -h|--help)
            sed -n '1,15p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $CONDA_ENV env not found at $(dirname $PY)"; exit 1
fi

cd "$LW_DIR"
exec "$PY" scripts/run_worldscore_batch.py \
    --instances "$INSTANCES_JSON" \
    --worldscore-root "$WORLDSCORE_OUTPUT_ROOT" \
    --output "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --names "$NAMES_FILTER" \
    --trajs "$TRAJS_FILTER" \
    --forward-speed "$FORWARD_SPEED" \
    --frames-per-iter "$FRAMES_PER_ITER" \
    --device "$DEVICE" \
    --system-config "$SYSTEM_CONFIG" \
    "${EXTRA[@]}"

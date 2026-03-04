# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmolVLA inference optimized for NVIDIA Jetson Orin Nano. Runs the `lerobot/smolvla_base` vision-language-action model (~450M params) in closed-loop with MuJoCo simulation using the SO-ARM100 robot.

**Hardware constraints:** 7.4GB unified GPU/CPU memory, target >=5 FPS, <6GB GPU usage.

## Development Commands

```bash
# Setup
./scripts/setup_menagerie.sh          # Download MuJoCo Menagerie
pip install -r requirements.txt

# Run
python main.py                         # Default config
python main.py --fp16 --visualize      # FP16 with viewer
python main.py --instruction "pick up the red cube"

# Test & Benchmark
pytest tests/                          # All tests
pytest tests/test_preprocessing.py -v  # Single test file
python -m benchmarking.benchmark       # Full benchmark
```

## Architecture

```
Reset Env → Capture RGB (top+wrist) → Preprocess → SmolVLA → Action Chunks → Map to Joints → Step Sim → Log → Repeat
```

Key modules:
- `simulation/so100_env.py` - Gym-style env wrapping MuJoCo
- `models/smolvla_loader.py` - Model loading, FP16 support, chunk handling
- `models/preprocessing.py` - Image resize/normalize (512x512, 0-1 range)
- `control/action_mapper.py` - Model output → 6D joint commands with limit clipping
- `inference/run_closed_loop.py` - Main control loop

## SmolVLA Specs

- Outputs 50 action chunks of shape `(batch, 50, 6)`
- Requires 512x512 RGB input normalized to 0-1
- Expects multi-view cameras named `top_cam` and `wrist_cam`
- Trained on LeRobot datasets including SO100 robot data (enables zero-shot)

## SO-ARM100 Joint Limits

| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| Rotation | -1.92 | 1.92 |
| Pitch | -3.32 | 0.174 |
| Elbow | -0.174 | 3.14 |
| Wrist_Pitch | -1.66 | 1.66 |
| Wrist_Roll | -2.79 | 2.79 |
| Jaw | -0.174 | 1.75 |

## Jetson-Specific Issues

```bash
# MuJoCo rendering backend selection
export MUJOCO_GL=egl      # Headless GPU (default)
export MUJOCO_GL=osmesa   # Software fallback
export MUJOCO_GL=glfw     # Windowed (needs display)

# GPU OOM - enable FP16
python main.py --fp16
# Or set use_fp16: true in configs/default.yaml

# Slow first run - model downloads ~2GB, set HF_HOME to fast storage
export HF_HOME=/path/to/nvme/.cache/huggingface
```
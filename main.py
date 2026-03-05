#!/usr/bin/env python3
"""
Edge SmolVLA Inference - Main Entry Point

Runs SmolVLA inference in closed-loop with MuJoCo simulation
on NVIDIA Jetson Orin Nano.

Usage:
    python main.py                              # Default config
    python main.py --config configs/fp16.yaml   # Custom config
    python main.py --instruction "pick up cube" # Custom instruction
    python main.py --visualize                  # Enable viewer
    python main.py --fp16                       # Force FP16 mode
"""

import argparse
import signal
import sys
from pathlib import Path

# Add lerobot to Python path for editable dev install
sys.path.insert(0, "/home/rushabh-jetson/lerobot/src")

from utils.config import load_config
from utils.logging import setup_logging
from simulation.so100_env import SO100Env
from models.smolvla_loader import SmolVLAInference
from inference.run_closed_loop import run_closed_loop


def parse_args():
    parser = argparse.ArgumentParser(
        description="SmolVLA inference on Jetson Orin Nano",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)
    return parser.parse_args()


def setup_signal_handlers():
    def _handler(signum, frame):
        print("\n[INFO] Shutting down ...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main():
    args = parse_args()
    setup_signal_handlers()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.instruction:
        config.setdefault("inference", {})["instruction"] = args.instruction
    if args.fp16:
        config.setdefault("model", {})["use_fp16"] = True
    if args.device:
        config.setdefault("model", {})["device"] = args.device
    if args.duration:
        config.setdefault("benchmark", {})["duration_sec"] = args.duration

    # Setup logging
    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs/")
    log_file = str(Path(log_dir) / f"{log_cfg.get('filename', 'smolvla')}.log")
    logger = setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_file)

    logger.info("=" * 60)
    logger.info("Edge SmolVLA Inference - Jetson Orin Nano")
    logger.info("=" * 60)

    model_cfg = config.get("model", {})
    sim_cfg = config.get("simulation", {})
    inf_cfg = config.get("inference", {})

    device = model_cfg.get("device", "cuda")
    use_fp16 = model_cfg.get("use_fp16", False)
    model_id = model_cfg.get("name", "lerobot/smolvla_base")
    scene_path = sim_cfg.get("scene_path", "simulation/scene.xml")
    instruction = inf_cfg.get("instruction", "pick up the cube")

    logger.info(f"Device: {device} | FP16: {use_fp16}")
    logger.info(f"Instruction: {instruction}")
    logger.info(f"Scene: {scene_path}")

    # Load model BEFORE initializing env: on Jetson unified memory, MuJoCo EGL
    # consumes ~0.5GB at init time, shrinking the CUDA-visible pool. Loading the
    # model first gives the GPU move the full ~3GB headroom it needs.
    logger.info("Loading SmolVLA model ...")
    model = SmolVLAInference(
        model_id=model_id,
        device=device,
        use_fp16=use_fp16,
        instruction=instruction,
    )
    model.load()

    # Initialize environment
    logger.info("Initializing simulation environment ...")
    env = SO100Env(
        scene_path=scene_path,
        image_resolution=sim_cfg.get("image_resolution", 512),
    )

    # Run inference loop
    try:
        metrics = run_closed_loop(env, model, config)
        logger.info(f"Result: {metrics}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

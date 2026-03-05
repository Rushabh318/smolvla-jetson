import logging
import time

import numpy as np

from control.action_mapper import map_action

logger = logging.getLogger(__name__)


def run_closed_loop(env, model, config: dict) -> dict:
    """Run the SmolVLA closed-loop control.

    Args:
        env: SO100Env instance
        model: SmolVLAInference instance (already loaded)
        config: Config dict (from load_config)

    Returns:
        Metrics dict with keys: steps, duration_sec, fps, mean_inference_ms
    """
    duration_sec = config.get("benchmark", {}).get("duration_sec", 30)
    warmup_steps = config.get("inference", {}).get("warmup_iterations", 5)

    obs = env.reset()
    step = 0
    inference_times = []

    logger.info(f"Running closed loop for {duration_sec}s (warmup={warmup_steps} steps) ...")
    loop_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - loop_start
        if elapsed >= duration_sec:
            break

        t0 = time.perf_counter()
        raw_action = model.predict(obs)
        t1 = time.perf_counter()

        if step >= warmup_steps:
            inference_times.append((t1 - t0) * 1000)

        mapped = map_action(raw_action)
        obs = env.step(mapped)

        step += 1
        if step % 50 == 0:
            fps = step / (time.perf_counter() - loop_start)
            logger.info(f"Step {step} | elapsed={elapsed:.1f}s | FPS={fps:.1f}")

    total_time = time.perf_counter() - loop_start
    fps = step / total_time
    mean_inf_ms = float(np.mean(inference_times)) if inference_times else 0.0

    metrics = {
        "steps": step,
        "duration_sec": total_time,
        "fps": fps,
        "mean_inference_ms": mean_inf_ms,
    }

    logger.info(
        f"Done — steps={step}, duration={total_time:.1f}s, "
        f"FPS={fps:.2f}, mean_inference={mean_inf_ms:.1f}ms"
    )
    return metrics

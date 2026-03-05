import numpy as np
import sys
sys.path.insert(0, "/home/rushabh-jetson/smolvla-jetson")

from simulation.so100_env import JOINT_LIMITS


def map_action(raw_action: np.ndarray) -> np.ndarray:
    """Map model output to joint commands.

    Args:
        raw_action: np.ndarray of shape (32,) or (6,) from the model

    Returns:
        np.ndarray of shape (6,) clipped to JOINT_LIMITS
    """
    joints = raw_action[:6]
    return np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

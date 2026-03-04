import numpy as np
import mujoco
from pathlib import Path

JOINT_LIMITS = np.array([
    [-1.92,  1.92],   # Rotation
    [-3.32,  0.174],  # Pitch
    [-0.174, 3.14],   # Elbow
    [-1.66,  1.66],   # Wrist_Pitch
    [-2.79,  2.79],   # Wrist_Roll
    [-0.174, 1.75],   # Jaw
])

HOME_QPOS = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])

# Cube spawn range on table surface (x, y in world frame).
# Table surface at z=0; cube half-height=0.015
CUBE_X_RANGE = (-0.12, 0.12)
CUBE_Y_RANGE = (-0.28, -0.10)
CUBE_Z = 0.015


class SO100Env:
    def __init__(self, scene_path: str, image_resolution: int = 512):
        scene_path = str(Path(scene_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_resolution, width=image_resolution)
        self.image_resolution = image_resolution

        # Cache joint/body IDs
        self._cube_jnt_adr = self._get_cube_qpos_adr()

    def _get_cube_qpos_adr(self) -> int:
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_id < 0:
            raise RuntimeError("'cube' body not found in scene XML")
        jnt_id = self.model.body_jntadr[cube_id]
        return int(self.model.jnt_qposadr[jnt_id])

    def reset(self, randomize_cube: bool = True) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_QPOS
        self.data.ctrl[:] = HOME_QPOS

        # Place cube on table
        adr = self._cube_jnt_adr
        x = np.random.uniform(*CUBE_X_RANGE) if randomize_cube else 0.0
        y = np.random.uniform(*CUBE_Y_RANGE) if randomize_cube else -0.18
        self.data.qpos[adr:adr+3] = [x, y, CUBE_Z]
        self.data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]  # unit quaternion

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> dict:
        action = np.clip(action, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self._get_obs()

    def render_rgb(self, camera_name: str) -> np.ndarray:
        """Returns (H, W, 3) uint8 array."""
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[:6].copy()

    def get_joint_dim(self) -> int:
        return 6

    def _get_obs(self) -> dict:
        return {
            "top": self.render_rgb("top"),
            "wrist": self.render_rgb("wrist"),
            "joint_positions": self.get_joint_positions(),
        }

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

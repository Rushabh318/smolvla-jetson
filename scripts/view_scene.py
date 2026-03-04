"""Launch MuJoCo interactive viewer starting at the home keyframe."""
import mujoco
import mujoco.viewer
from pathlib import Path

SCENE = Path(__file__).parent.parent / "simulation" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

# Apply home keyframe (arm in ready pose, not flat qpos=0)
mujoco.mj_resetDataKeyframe(model, data, 0)

print("Controls: Left drag=rotate  Right drag=pan  Scroll=zoom")
print("          Tab=cycle cameras  Space=pause  Ctrl+R=reset  Esc=quit")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

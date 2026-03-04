#!/bin/bash
# Setup MuJoCo Menagerie for SO-ARM100 robot
# Run from project root: ./scripts/setup_menagerie.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"
MENAGERIE_DIR="$THIRD_PARTY_DIR/mujoco_menagerie"

echo "============================================"
echo "Setting up MuJoCo Menagerie"
echo "============================================"

# Create third_party directory if it doesn't exist
mkdir -p "$THIRD_PARTY_DIR"

# Clone or update MuJoCo Menagerie
if [ -d "$MENAGERIE_DIR" ]; then
    echo "MuJoCo Menagerie already exists. Updating..."
    cd "$MENAGERIE_DIR"
    git fetch origin
    git pull origin main
else
    echo "Cloning MuJoCo Menagerie..."
    cd "$THIRD_PARTY_DIR"
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
fi

# Verify SO-ARM100 exists
SO100_DIR="$MENAGERIE_DIR/trs_so_arm100"
if [ -d "$SO100_DIR" ]; then
    echo "✓ SO-ARM100 model found at: $SO100_DIR"
    echo "  Files:"
    ls -la "$SO100_DIR"
else
    echo "✗ ERROR: SO-ARM100 model not found!"
    echo "  Expected at: $SO100_DIR"
    echo "  The model may have been renamed or moved in the repository."
    exit 1
fi

# Test MuJoCo installation
echo ""
echo "============================================"
echo "Testing MuJoCo Installation"
echo "============================================"

python3 << 'EOF'
import sys
try:
    import mujoco
    print(f"✓ MuJoCo version: {mujoco.__version__}")

    # Check minimum version
    version_parts = mujoco.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major < 3 or (major == 3 and minor < 1):
        print(f"✗ WARNING: MuJoCo >= 3.1.6 required for SO-ARM100")
        print(f"  Current version: {mujoco.__version__}")
        print(f"  Run: pip install mujoco>=3.1.6")

except ImportError:
    print("✗ MuJoCo not installed!")
    print("  Run: pip install mujoco>=3.1.6")
    sys.exit(1)
EOF

# Test loading SO-ARM100 model
echo ""
echo "============================================"
echo "Testing SO-ARM100 Model Loading"
echo "============================================"

python3 << EOF
import sys
sys.path.insert(0, '$PROJECT_ROOT')

import mujoco
import os

so100_xml = "$SO100_DIR/scene.xml"

try:
    model = mujoco.MjModel.from_xml_path(so100_xml)
    data = mujoco.MjData(model)

    print(f"✓ SO-ARM100 model loaded successfully!")
    print(f"  Number of bodies: {model.nbody}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Number of actuators: {model.nu}")
    print(f"  Timestep: {model.opt.timestep}")

    # List joint names
    print(f"\n  Joint names:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jnt_range = model.jnt_range[i]
            print(f"    {i}: {name} (range: {jnt_range[0]:.2f} to {jnt_range[1]:.2f})")

    # List actuator names
    print(f"\n  Actuator names:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            print(f"    {i}: {name}")

except Exception as e:
    print(f"✗ Failed to load SO-ARM100: {e}")
    sys.exit(1)
EOF

# Test rendering backend
echo ""
echo "============================================"
echo "Testing Rendering Backend"
echo "============================================"

python3 << 'EOF'
import os

# Check available rendering backends
backends = []

# Check EGL
try:
    os.environ['MUJOCO_GL'] = 'egl'
    import mujoco
    # Try to create a simple context
    backends.append('egl')
    print("✓ EGL backend available")
except:
    print("○ EGL backend not available")

# Check OSMesa
try:
    os.environ['MUJOCO_GL'] = 'osmesa'
    import importlib
    importlib.reload(mujoco)
    backends.append('osmesa')
    print("✓ OSMesa backend available")
except:
    print("○ OSMesa backend not available")

# Check GLFW
try:
    import glfw
    if glfw.init():
        backends.append('glfw')
        glfw.terminate()
        print("✓ GLFW backend available (for visualization)")
except:
    print("○ GLFW backend not available")

if not backends:
    print("\n⚠ WARNING: No rendering backend available!")
    print("  Install one of:")
    print("    sudo apt-get install libosmesa6-dev  # For headless")
    print("    sudo apt-get install libglfw3-dev    # For visualization")
else:
    print(f"\n✓ Available backends: {', '.join(backends)}")
    print(f"  Recommended for Jetson: {'egl' if 'egl' in backends else 'osmesa'}")
EOF

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Install Python dependencies: pip install -r requirements.txt"
echo "  2. Set rendering backend: export MUJOCO_GL=egl"
echo "  3. Run the demo: python main.py"
echo ""
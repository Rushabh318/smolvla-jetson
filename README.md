# Edge SmolVLA Inference on Jetson Orin Nano

Vision-Language-Action (VLA) model inference optimized for NVIDIA Jetson edge devices. Run SmolVLA in closed-loop simulation with the SO-ARM100 robot.

## Features

- **Zero-shot robotic manipulation** using SmolVLA (~450M parameters)
- **MuJoCo simulation** with SO-ARM100 6-DOF robot arm
- **Multi-camera setup** (top + wrist views) matching training data
- **FP16 inference** for reduced memory and faster execution
- **Benchmarking suite** for latency, FPS, and memory profiling
- **Docker support** for reproducible deployment

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Device | NVIDIA Jetson Orin Nano |
| Memory | 8GB unified (GPU/CPU shared) |
| OS | Ubuntu 22.04 (JetPack 6.x) |
| CUDA | 12.x |
| PyTorch | 2.x with CUDA support |

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/jetson-smolVLA-inference.git
cd jetson-smolVLA-inference
```

### 2. Setup Environment

```bash
# Download MuJoCo Menagerie (includes SO-ARM100)
./scripts/setup_menagerie.sh

# Install Python dependencies
pip install -r requirements.txt

# Set rendering backend
export MUJOCO_GL=egl  # For headless GPU rendering
```

### 3. Run Inference

```bash
# Default configuration
python main.py

# With custom instruction
python main.py --instruction "pick up the red cube"

# With visualization
python main.py --visualize

# With FP16 for faster inference
python main.py --fp16
```

### 4. Run Benchmarks

```bash
# Full benchmark suite
python -m benchmarking.benchmark

# Custom duration
python -m benchmarking.benchmark --duration 60

# FP16 comparison
python -m benchmarking.benchmark --compare-fp16
```

## Project Structure

```
edge_vla/
├── configs/default.yaml      # Runtime configuration
├── simulation/
│   ├── scene.xml             # MuJoCo scene definition
│   ├── so100_env.py          # Gym-style environment
│   └── renderer.py           # Multi-camera rendering
├── models/
│   ├── smolvla_loader.py     # Model loading & inference
│   └── preprocessing.py      # Image/text preprocessing
├── control/
│   └── action_mapper.py      # Model output → joints
├── inference/
│   └── run_closed_loop.py    # Main control loop
├── benchmarking/
│   ├── benchmark.py          # Benchmark runner
│   └── metrics.py            # Timing utilities
├── main.py                   # Entry point
└── Dockerfile                # Jetson deployment
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  use_fp16: true        # Enable half precision
  device: "cuda"        # GPU inference

simulation:
  image_resolution: 512 # Camera resolution
  control_frequency: 50 # Control loop Hz

inference:
  instruction: "pick up the cube"
  chunk_strategy: "first"  # Action selection

benchmark:
  duration_sec: 30
```

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| FPS | ≥ 5 | TBD |
| Inference Latency | < 150ms | TBD |
| GPU Memory | < 6GB | TBD |
| Episode Duration | ≥ 30s | TBD |

## Docker Usage

```bash
# Build image
docker build -t smolvla-jetson:latest .

# Run with GPU
docker run --runtime nvidia --gpus all \
    -v $(pwd)/outputs:/app/outputs \
    smolvla-jetson:latest

# Interactive mode
docker run --runtime nvidia --gpus all -it \
    smolvla-jetson:latest bash
```

## Model Information

**SmolVLA** is a compact Vision-Language-Action model from HuggingFace/LeRobot:

- **Parameters**: ~450M
- **Training Data**: LeRobot community datasets (includes SO100 robot)
- **Action Output**: 50-timestep chunks, 6D actions
- **Input**: 512×512 RGB images (multi-view)
- **Architecture**: SigLIP vision encoder + SmolLM2 language model

## Robot Information

**SO-ARM100** is a 6-DOF robot arm from MuJoCo Menagerie:

| Joint | Range (rad) | Description |
|-------|-------------|-------------|
| Rotation | [-1.92, 1.92] | Shoulder yaw |
| Pitch | [-3.32, 0.17] | Shoulder pitch |
| Elbow | [-0.17, 3.14] | Elbow flex |
| Wrist_Pitch | [-1.66, 1.66] | Wrist pitch |
| Wrist_Roll | [-2.79, 2.79] | Wrist roll |
| Jaw | [-0.17, 1.75] | Gripper |

## Troubleshooting

### MuJoCo Rendering Errors

```bash
# Try different backends
export MUJOCO_GL=osmesa  # Software rendering
export MUJOCO_GL=egl     # GPU headless
export MUJOCO_GL=glfw    # Windowed (needs display)
```

### GPU Out of Memory

```bash
# Enable FP16
python main.py --fp16

# Or in config:
# model.use_fp16: true
```

### Slow Model Download

```bash
# Set cache to fast storage
export HF_HOME=/path/to/nvme/.cache/huggingface
```

## License

Apache 2.0

## Acknowledgments

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) - SmolVLA model
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - SO-ARM100 robot
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - Edge AI platform
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

## Performance Results

| Metric | Target | Achieved |
|--------|--------|----------|
| FPS | ≥ 5 | **9.63** |
| Inference Latency | < 150ms | **62ms** |
| GPU Memory | < 6GB | ~1.1GB model (fp16) |
| Episode Duration | ≥ 30s | ✓ (289 steps / 30s) |

Measured on Jetson Orin Nano 8GB, fp16, 512×512, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.

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
# Enable FP16 (required on Jetson — fp32 activations OOM during VLM forward pass)
python main.py --fp16

# Or in config:
# model.use_fp16: true
```

### Slow Model Download

```bash
# Set cache to fast storage
export HF_HOME=/path/to/nvme/.cache/huggingface
```

## Jetson-Specific Notes

These issues were encountered and solved during development on Jetson Orin Nano 8GB.

### PyTorch wheel: must use Jetson build, not server-ARM build

Standard `pip install torch` on ARM installs a server-ARM wheel targeting sm_80/sm_90 (A100/H100).
Jetson Orin GPU is **sm_87** — running CUDA kernels with the wrong wheel gives:
```
cudaErrorNoKernelImageForDevice
```
Install the Jetson-specific wheel from NVIDIA (e.g. `torch-2.9.1-cp310-cp310-linux_aarch64.whl`).
Verify: `torch.cuda.get_arch_list()` must contain `sm_87`.

### Unified memory and the CUDA caching allocator

Jetson uses unified memory (CPU and GPU share the same physical RAM).
PyTorch's `CUDACachingAllocator` calls NVML internally to pre-allocate blocks, which triggers an
assertion failure on Jetson:
```
NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1123
```
Fix: `export PYTORCH_NO_CUDA_MEMORY_CACHING=1`

Also, `transformers 5.x` calls `caching_allocator_warmup` at import time (same NVML path).
The loader patches this to a no-op before importing any transformers model.

### `torch.cuda.memory_allocated()` returns 0 on Jetson

NvMap (Jetson's unified memory allocator) bypasses PyTorch's standard CUDA allocator tracking.
Use `torch.cuda.mem_get_info()` instead — it queries NVML and reports correctly.

### Load model before initializing MuJoCo

MuJoCo EGL context initialization consumes ~0.5GB of unified memory. On a system with ~3GB free,
this leaves insufficient contiguous unified memory for moving the model to CUDA. Always load the
model first, then initialize the simulation environment.

### FP16 is required for inference (not just recommended)

fp32 VLM forward pass activations for two 512×512 images exhaust available unified memory
(~1.6GB free at inference time). fp16 halves activation memory and enables stable 9.6 FPS.

## License

Apache 2.0

## Acknowledgments

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) - SmolVLA model
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - SO-ARM100 robot
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - Edge AI platform
# TASKS.md - Implementation Task Breakdown

## Project: Edge SmolVLA Inference on Jetson Orin Nano

### Key Research Findings

| Component | Details |
|-----------|---------|
| **SmolVLA Model** | `lerobot/smolvla_base` (~450M params) |
| **Training Data** | LeRobot community datasets, includes SO100 robot data |
| **Action Chunk Size** | 50 timesteps per prediction |
| **Image Input** | 512×512, normalized 0-1 |
| **Camera Setup** | Multi-view: top + wrist (minimum) |
| **Robot** | SO-ARM100 (6 joints: 5 arm + 1 gripper) |
| **Action Dimensions** | 6 (matches SO100 training data) |
| **Zero-Shot Viability** | HIGH - model trained on same robot type |

---

## Phase 1: Project Setup
**Estimated: Day 1**

### Task 1.1: Initialize Repository Structure
- [ ] Create directory structure per PRD
- [ ] Initialize git repository
- [ ] Create `.gitignore`
- [ ] Create `requirements.txt`
- [ ] Create `pyproject.toml` (optional)

### Task 1.2: Setup MuJoCo Menagerie
- [ ] Create `scripts/setup_menagerie.sh`
- [ ] Clone MuJoCo Menagerie to `third_party/`
- [ ] Verify SO-ARM100 model loads (`trs_so_arm100/`)
- [ ] Test basic MuJoCo rendering on Jetson

### Task 1.3: Verify Dependencies
- [ ] Install MuJoCo >= 3.1.6
- [ ] Install PyTorch (already present: 2.10.0+cu126)
- [ ] Install transformers, lerobot
- [ ] Verify CUDA availability
- [ ] Test GPU memory allocation

---

## Phase 2: Simulation Environment
**Estimated: Day 2-3**

### Task 2.1: Create Scene XML
**File:** `simulation/scene.xml`

- [ ] Include SO-ARM100 from menagerie
- [ ] Add ground plane / table surface
- [ ] Add cube object (graspable, dynamic body)
- [ ] Add RGB camera "top_cam" (overhead view)
- [ ] Add RGB camera "wrist_cam" (end-effector view)
- [ ] Configure lighting for consistent rendering
- [ ] Set physics timestep appropriate for control

**Camera Specs (to match SmolVLA training):**
```
top_cam: overhead, facing down at workspace
wrist_cam: attached to end-effector, facing forward
Resolution: 512x512 (configurable)
```

### Task 2.2: Implement Environment Wrapper
**File:** `simulation/so100_env.py`

- [ ] Implement `SO100PickEnv` class
- [ ] `__init__(self, config)` - load scene, initialize MuJoCo
- [ ] `reset()` - reset robot pose, randomize cube position
- [ ] `step(action)` - apply 6D action, advance simulation
- [ ] `render_rgb(camera_name)` - return numpy array (H, W, 3)
- [ ] `get_joint_positions()` - return current joint state
- [ ] `get_joint_dim()` - return 6
- [ ] Action clipping to joint limits
- [ ] Graceful error handling

**Joint Limits Reference:**
| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| Rotation | -1.92 | 1.92 |
| Pitch | -3.32 | 0.174 |
| Elbow | -0.174 | 3.14 |
| Wrist_Pitch | -1.66 | 1.66 |
| Wrist_Roll | -2.79 | 2.79 |
| Jaw (gripper) | -0.174 | 1.75 |

### Task 2.3: Implement Renderer
**File:** `simulation/renderer.py`

- [ ] Implement `MuJoCoRenderer` class
- [ ] Support EGL backend for headless (primary)
- [ ] Support GLFW for visualization (optional)
- [ ] Configurable resolution
- [ ] Multi-camera support
- [ ] Memory-efficient frame capture

---

## Phase 3: Model Integration
**Estimated: Day 3-4**

### Task 3.1: SmolVLA Loader
**File:** `models/smolvla_loader.py`

- [ ] Implement `SmolVLAModel` class
- [ ] Load from HuggingFace: `lerobot/smolvla_base`
- [ ] Move to GPU (CUDA)
- [ ] Support FP32 inference
- [ ] Support FP16 inference (config flag)
- [ ] Handle action chunk output (50 timesteps)
- [ ] Implement action selection strategy (first, average, etc.)
- [ ] Memory optimization for Jetson

**Key Implementation Notes:**
```python
# SmolVLA outputs 50 action chunks
# Each action is 6D for SO100
# Output shape: (batch, 50, 6)
# Strategy: Use first action, or execute chunk with interpolation
```

### Task 3.2: Preprocessing Pipeline
**File:** `models/preprocessing.py`

- [ ] `preprocess_image(np_image)` - resize, normalize, tensorize
- [ ] `preprocess_multi_view(images_dict)` - handle top + wrist
- [ ] `tokenize_instruction(text, tokenizer)` - text processing
- [ ] `prepare_robot_state(joint_positions)` - proprioceptive input
- [ ] Batch dimension handling
- [ ] GPU transfer optimization

**Image Preprocessing Spec:**
```python
# Input: numpy (H, W, 3) uint8
# Resize to 512x512
# Normalize to 0-1 float32
# Convert to torch tensor (1, 3, 512, 512)
# Transfer to GPU
```

### Task 3.3: Action Mapper
**File:** `control/action_mapper.py`

- [ ] `map_action_to_joints(action_vector, joint_limits)`
- [ ] Clip to valid joint ranges
- [ ] Scale from model output range to joint range
- [ ] Handle dimension mismatch gracefully
- [ ] Support delta vs absolute action modes

---

## Phase 4: Inference Loop
**Estimated: Day 4-5**

### Task 4.1: Closed-Loop Controller
**File:** `inference/run_closed_loop.py`

- [ ] Implement `run_episode(env, model, instruction, config)`
- [ ] Main control loop:
  1. Capture multi-view RGB
  2. Get robot proprioceptive state
  3. Preprocess inputs
  4. Run SmolVLA inference
  5. Extract action from chunk
  6. Map to joint commands
  7. Step simulation
  8. Log timing metrics
  9. Repeat
- [ ] Configurable episode duration
- [ ] Configurable control frequency
- [ ] SIGINT handler for graceful shutdown
- [ ] Periodic stats logging

### Task 4.2: Action Chunk Execution
**File:** `inference/chunk_executor.py`

- [ ] Implement chunk execution strategies:
  - Single action (first only)
  - Full chunk execution
  - Overlapping chunk merge
- [ ] Asynchronous inference option
- [ ] Action queue management

---

## Phase 5: Benchmarking
**Estimated: Day 5-6**

### Task 5.1: Metrics Collection
**File:** `benchmarking/metrics.py`

- [ ] `Timer` class - context manager for timing
- [ ] `FPSTracker` - rolling FPS calculation
- [ ] `MemoryLogger` - GPU/system memory tracking
- [ ] `LatencyStats` - mean, std, min, max, p95

### Task 5.2: Benchmark Suite
**File:** `benchmarking/benchmark.py`

- [ ] Single inference latency test
- [ ] Closed-loop FPS benchmark
- [ ] Memory profiling
- [ ] FP32 vs FP16 comparison
- [ ] CSV export with columns:
  - timestamp
  - inference_latency_ms
  - loop_latency_ms
  - fps
  - gpu_memory_mb
  - system_memory_mb

### Task 5.3: Benchmark Runner
- [ ] CLI interface for benchmarks
- [ ] Warmup iterations
- [ ] Configurable duration
- [ ] Results summary printout

---

## Phase 6: Configuration & Logging
**Estimated: Day 6**

### Task 6.1: Configuration System
**File:** `configs/default.yaml`

- [ ] Image resolution settings
- [ ] Control frequency
- [ ] FP16 toggle
- [ ] Device selection (cuda/cpu)
- [ ] Benchmark duration
- [ ] Model path/name
- [ ] Camera names
- [ ] Logging paths
- [ ] Action chunk strategy

### Task 6.2: Logging System
**File:** `utils/logging.py`

- [ ] File-based logging
- [ ] Log rotation
- [ ] Structured log format
- [ ] Separate logs for:
  - System events
  - Inference metrics
  - Errors

---

## Phase 7: Visualization
**Estimated: Day 6-7**

### Task 7.1: Real-time Viewer
**File:** `visualization/viewer.py`

- [ ] Optional GLFW window for live view
- [ ] Display current camera feed
- [ ] Overlay FPS and latency
- [ ] Keyboard controls (pause, reset, quit)
- [ ] Toggle between cameras

### Task 7.2: Recording
**File:** `visualization/recorder.py`

- [ ] Save episode as video (MP4)
- [ ] Save frames as images
- [ ] Configurable recording interval

---

## Phase 8: Testing
**Estimated: Day 7**

### Task 8.1: Unit Tests
**File:** `tests/test_*.py`

- [ ] `test_action_mapper.py` - action clipping, scaling
- [ ] `test_preprocessing.py` - image shapes, normalization
- [ ] `test_environment.py` - reset, step, render

### Task 8.2: Integration Tests
- [ ] Smoke test: 5-second episode
- [ ] Memory leak test: extended run
- [ ] Error handling test: invalid inputs

---

## Phase 9: Docker & Deployment
**Estimated: Day 7-8**

### Task 9.1: Dockerfile
- [ ] Base image: `nvcr.io/nvidia/l4t-pytorch`
- [ ] Install MuJoCo dependencies
- [ ] Install Python packages
- [ ] Copy application code
- [ ] Set entrypoint

### Task 9.2: Documentation
- [ ] Update README.md with:
  - Installation instructions
  - Quick start guide
  - Configuration reference
  - Benchmark results
- [ ] Add architecture diagram

---

## Phase 10: TensorRT (Stretch Goal)
**Estimated: Day 8+**

### Task 10.1: ONNX Export
- [ ] Export SmolVLA to ONNX
- [ ] Verify ONNX inference matches PyTorch

### Task 10.2: TensorRT Engine
- [ ] Build TensorRT engine from ONNX
- [ ] Implement TensorRT inference class
- [ ] Benchmark latency improvement

---

## Acceptance Checklist

- [ ] System runs closed-loop for 30+ seconds
- [ ] Achieves >= 5 FPS
- [ ] Generates benchmark CSV
- [ ] FP16 mode works correctly
- [ ] No hardcoded paths
- [ ] Clean modular structure
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Docker image builds

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| GPU OOM on Jetson | FP16, smaller batch, gradient checkpointing |
| MuJoCo rendering issues | Test EGL early, fallback to OSMesa |
| SmolVLA load failure | Cache model locally, handle gracefully |
| Action dimension mismatch | Validate at startup, clear error messages |
| < 5 FPS performance | Profile bottlenecks, optimize preprocessing |

---

## Dependencies Between Tasks

```
Phase 1 (Setup)
    └── Phase 2 (Simulation)
            └── Phase 3 (Model)
                    └── Phase 4 (Inference Loop)
                            └── Phase 5 (Benchmarking)

Phase 6 (Config/Logging) - Can parallel with Phase 2-4
Phase 7 (Visualization) - Can parallel with Phase 4-5
Phase 8 (Testing) - After Phase 4
Phase 9 (Docker) - After Phase 5
Phase 10 (TensorRT) - Stretch, after Phase 5
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

A high-performance YOLO ONNX inference server written in C++17. It serves object detection requests over HTTP using dynamic batching to maximize throughput. The project is primarily a learning exercise for C++ network programming, multithreading, and AI inference infrastructure.

**Tech stack:** C++17, CMake 3.18+, muduo (Reactor-pattern networking), ONNX Runtime 1.20.0, OpenCV 4.x, nlohmann/json, spdlog, Google Test.

## Build and Run

```bash
# Quick build (uses build.sh)
./build.sh

# Or manually
cmake -B build && cmake --build build -j6

# With full tests (requires ONNX Runtime)
cmake -B build -DBUILD_TESTS=ON && cmake --build build -j6

# Run the server (CPU mode)
./build/bin/inference_server --model models/yolov8n.onnx --port 8080 --log-level info

# Run with GPU (CUDA)
./build/bin/inference_server --model models/yolov8n.onnx --port 8080 --gpu cuda --gpu-id 0

# Custom batch config
./build/bin/inference_server --model models/yolov8n.onnx --port 8080 --threads 4 --batch-size 16 --batch-timeout 5
```

## Tests

Two test targets exist:

| Target | Depends on ONNX Runtime? | Description |
|--------|--------------------------|-------------|
| `test_framework` | No | Always built. Tests ImagePreprocessor and NMS. |
| `test_preprocessing` | No | Built with `-DBUILD_TESTS=ON`. Same preprocessing tests. |
| `test_batch_scheduler` | Yes | Built with `-DBUILD_TESTS=ON` + ONNX Runtime available. Tests BatchConfig and PendingRequest. |

```bash
# Run framework tests (always available)
./build/bin/test_framework

# Run all tests via ctest (requires -DBUILD_TESTS=ON)
cd build && ctest --output-on-failure

# Run a specific test
./build/bin/test_framework --gtest_filter="*Letterbox*"
```

## Architecture

### Data flow (request → response)

```
HTTP Client → muduo TcpServer → HttpContext (state machine parse)
  → RequestHandlers (extract image, route to /infer, /health, /stats)
    → InferenceService.SubmitRequest()
      → BatchScheduler.Submit() (enqueue + std::promise)
        → BatchingLoop (single background thread):
          1. Collect requests (timeout or max_batch_size reached)
          2. Preprocess batch (Letterbox resize + BGR→RGB + normalize)
          3. Build ONNX tensor (HWC → CHW layout)
          4. ModelSession.Run() (ONNX inference)
          5. Parse YOLO output → confidence filter → NMS
          6. promise.set_value() → HTTP handler resumes
            → JSON response → Client
```

### Thread model

- **Main thread:** muduo `EventLoop` for accepting connections
- **N IO threads:** muduo worker loops handling read/write, HTTP parsing, and request submission
- **1 batch thread:** `BatchScheduler::BatchingLoop` — preprocessing, ONNX inference, post-processing

### Key components

| Component | Files | Role |
|-----------|-------|------|
| **HTTP Server** | `server/InferenceServer.cpp`, `server/Handlers.cpp`, `server/HttpContext.cpp` | muduo TcpServer wrapper. Custom HTTP state-machine parser (not httplib). Handles multipart/form-data, binary, and base64 image inputs. |
| **Batch Scheduler** | `batch/BatchScheduler.cpp` | Core algorithm. Single-threaded batching loop collecting requests in a time window or until `max_batch_size`. Does preprocessing, inference, NMS, then resolves promises. |
| **Model Session** | `model/ModelSession.cpp` | ONNX Runtime session wrapper. Supports CPU, CUDA, and TensorRT Execution Providers via `ModelConfig::GpuProvider` enum. CPU EP is always appended as fallback. Falls back to simulation mode when ONNX Runtime is unavailable. |
| **Model Manager** | `model/ModelManager.cpp` | Multi-model registry with thread-safe name-to-session lookup. |
| **Preprocessing** | `preprocessing/ImagePreprocessor.cpp`, `preprocessing/NMS.cpp` | Letterbox resize (aspect-ratio preserving), BGR→RGB, normalization. Non-maximum suppression. |
| **Service** | `inference/InferenceService.cpp` | Glue layer: initializes ModelManager, loads models, creates Preprocessor/NMS/BatchScheduler. |
| **Memory Pools** | `memory/ObjectPool.hpp`, `memory/MemoryPool.hpp` | Header-only object and tensor memory pools (optimization targets, not yet integrated). |

### Configuration files

- `configs/server_config.json` — Server port, threads, batch params (max=8, timeout=10ms, min=1)
- `configs/model_config.json` — YOLOv8 input shape `[1,3,640,640]`, output `[1,84,8400]`, NMS thresholds

### HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Submit image (multipart/form-data). Returns JSON detections + timing. |
| `/health` | GET | `{"status": "healthy", "service": "inference_server", "version": "1.0.0"}` |
| `/stats` | GET | Batch statistics (total requests, batches, avg batch size, pending count) |

## Important Notes

- **Execution Providers:** `ModelConfig::GpuProvider` enum (`NONE`/`CUDA`/`TENSORRT`) controls which ONNX Runtime EP is registered in `ConfigureSessionOptions()`. CPU EP is always appended as the final fallback. CLI: `--gpu cuda` or `--gpu tensorrt`, `--gpu-id 0`. Requires a GPU-capable ONNX Runtime build — the CPU-only `third_party/onnxruntime/` won't work with CUDA/TensorRT.
- **httplib.h:** Present in `third_party/httplib/` but **not actively used**. HTTP parsing is custom (state machine in `HttpContext.cpp`).
- **Known issue:** There is an unresolved crash with `libonnxruntime` during debugging (see `docs/note.md`).
- **Commit format:** Follows Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.) with Chinese descriptions.
- **Models are gitignored:** `models/*.onnx` is in `.gitignore`. Download with `python scripts/download_model.py --model yolov8n --output models/`.

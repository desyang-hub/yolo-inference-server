# 架构设计文档

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client Layer                                 │
│  (Web App / Mobile / Python Script / curl)                         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP / REST API
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Network Layer                                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  muduo::TcpServer (Reactor Pattern)                         │   │
│  │                                                             │   │
│  │  EventLoop ──▶ Acceptor ──▶ TcpConnection ──▶ Buffer       │   │
│  │                                                             │   │
│  │  线程模型:                                                   │   │
│  │  - 1 个主线程 (Accept)                                      │   │
│  │  - N 个 IO 线程 (Read/Write)                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HttpRequest
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Service Layer                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  RequestHandlers                                            │   │
│  │                                                             │   │
│  │  1. Parse HTTP Request                                      │   │
│  │  2. Extract Image (multipart / binary)                      │   │
│  │  3. Build InferenceRequest                                  │   │
│  │  4. Submit to InferenceService                              │   │
│  │  5. Wait for Result (std::future)                           │   │
│  │  6. Serialize Response (JSON)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ PendingRequest
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Inference Layer                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  BatchScheduler (Dynamic Batching)                          │   │
│  │                                                             │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  Pending Queue (Thread-Safe)                         │  │   │
│  │  │                                                     │  │   │
│  │  │  Collect requests within time window:               │  │   │
│  │  │  - Until max_batch_size reached                     │  │   │
│  │  │  - Or timeout exceeded                              │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                             │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  Batching Thread (Single)                            │  │   │
│  │  │                                                     │  │   │
│  │  │  1. Collect batch from queue                        │  │   │
│  │  │  2. Preprocess images → Tensor                      │  │   │
│  │  │  3. Run inference (ONNX Runtime)                    │  │   │
│  │  │  4. Postprocess (NMS)                               │  │   │
│  │  │  5. Split results → promises                        │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ModelManager                                                │   │
│  │                                                             │   │
│  │  ModelSession ──▶ ONNX Runtime Session                     │   │
│  │  (Thread-safe, reusable)                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 数据流详解

### 请求路径

```
Client → TCP → HTTP Parse → PendingRequest → Batch Queue
                                                      │
                                            ┌─────────┴─────────┐
                                            │  Time Window Wait │
                                            │  or Batch Full    │
                                            └─────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  Collect Batch    │
                                            └─────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  Preprocess       │
                                            │  (Letterbox +     │
                                            │   Normalize)      │
                                            └─────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  ONNX Runtime     │
                                            │  Run()            │
                                            └─────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  Postprocess      │
                                            │  (NMS)            │
                                            └─────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  Split Results    │
                                            │  Set Promises     │
                                            └─────────┬─────────┘
                                                      │
    Client ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

## 关键设计决策

### 1. 为什么使用 muduo？

- **经验匹配**: 开发者已有 muduo 项目经验
- **Reactor 模式**: 经典的 Linux 网络编程模式
- **非阻塞 IO**: epoll + 事件驱动，高并发友好

### 2. 为什么使用 Dynamic Batching？

| 指标 | 无 Batch | Batch=8 | Batch=32 |
|------|----------|---------|----------|
| GPU 利用率 | ~30% | ~70% | ~90% |
| 单请求延迟 | 低 | 中 | 高 |
| 吞吐量 (QPS) | 低 | 中 | 高 |

**策略**: 时间窗口 + 大小阈值，平衡延迟和吞吐

### 3. 为什么使用 std::promise/std::future？

```
HTTP Handler Thread              Batch Thread
      │                              │
      │  future = Submit(req)       │
      │  ──────────────────────▶    │
      │                              │
      │  future.get() (阻塞)         │ ProcessBatch()
      │  │                           │
      │  │                           │ promise.set_value()
      │  │                           │ ◀────────────────────
      │  ▼                           │
      │  response ←──────────────────┘
      │
      │  Send HTTP Response
```

### 4. 为什么使用 Status 而不是异常？

- **性能**: 异常有开销（即使不抛出）
- **可控**: 显式错误检查
- **风格**: 接近 Go/Rust 的错误处理模式
- **现代 C++**: C++23 有 std::expected，我们提前实践

## 线程模型

```
┌──────────────────────────────────────────────────────────┐
│  Main Thread                                             │
│  └── EventLoop::loop()                                   │
│      └── Accept new connections                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  IO Threads (N)                                          │
│  └── EventLoop (each thread has its own)                 │
│      ├── Read HTTP requests                              │
│      ├── Parse requests                                  │
│      ├── Submit to BatchScheduler                        │
│      ├── Wait for results                                │
│      └── Send responses                                  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Batch Thread (1)                                        │
│  └── BatchingLoop()                                      │
│      ├── Wait for requests                               │
│      ├── Collect batch                                   │
│      ├── Preprocess                                      │
│      ├── Run ONNX Runtime (inference)                    │
│      ├── Postprocess (NMS)                               │
│      └── Split results                                   │
└──────────────────────────────────────────────────────────┘
```

## 内存模型

```
┌──────────────────────────────────────────────────────────┐
│  Object Pool                                             │
│                                                          │
│  Pre-allocated Buffers:                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Buffer 1│ │ Buffer 2│ │ Buffer 3│ │ Buffer 4│       │
│  │ (free)  │ │ (in use)│ │ (free)  │ │ (in use)│       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                                                          │
│  Benefits:                                               │
│  - No malloc/free during inference                       │
│  - Reduced memory fragmentation                          │
│  - Predictable latency                                   │
└──────────────────────────────────────────────────────────┘
```

## 配置文件

### server_config.json

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "num_threads": 4
  },
  "batch": {
    "max_batch_size": 8,
    "timeout_ms": 10
  },
  "models": [...]
}
```

### model_config.json

```json
{
  "name": "yolov8n",
  "model_path": "models/yolov8n.onnx",
  "input": {
    "shape": [1, 3, 640, 640]
  },
  "preprocess": {
    "scale_to_zero_one": true,
    "rgb_conversion": true
  },
  "postprocess": {
    "conf_threshold": 0.25,
    "nms_threshold": 0.45
  }
}
```

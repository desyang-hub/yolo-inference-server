# 高性能 ONNX Runtime 推理服务引擎

基于 C++ 实现的高性能 ONNX 推理服务，支持多模型加载、动态批处理、异步推理。

## 📋 项目概述

本项目是一个面向学习的**AI 推理基础设施**项目，旨在通过实践掌握：

- **C++ 网络编程**：基于 muduo 框架的 Reactor 模式
- **多线程编程**：生产者-消费者模型、线程安全
- **ONNX Runtime**：C++ API 使用、模型推理优化
- **动态批处理**：Dynamic Batching 算法实现
- **性能优化**：对象池、零拷贝、预处理加速

## 🏗️ 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                      HTTP Client                             │
└──────────────┬───────────────────────────────────────────────┘
               │ HTTP POST /infer
               ▼
┌──────────────────────────────────────────────────────────────┐
│                   InferenceServer                             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              muduo::TcpServer                        │     │
│  │  (Reactor Pattern, EventLoop, TcpConnection)        │     │
│  └───────────┬─────────────────────────────────────────┘     │
│              │                                                │
│  ┌───────────▼─────────────────────────────────────────┐     │
│  │              RequestHandlers                         │     │
│  │  (HTTP 解析, 图像提取, JSON 序列化)                  │     │
│  └───────────┬─────────────────────────────────────────┘     │
│              │                                                │
└──────────────┼────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                   InferenceService                            │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │ Model      │  │ Batch      │  │  Preprocessor / NMS  │   │
│  │ Manager    │  │ Scheduler  │  │                      │   │
│  │            │  │            │  │  - ImagePreprocessor │   │
│  │ - Load()   │  │ - Submit() │  │  - NMS               │   │
│  │ - Get()    │  │ - Batching │  │                      │   │
│  └─────┬──────┘  │ - Process()│  └──────────────────────┘   │
│        │         └──────┬─────┘                              │
│        ▼                ▼                                     │
│  ┌────────────┐  ┌────────────┐                              │
│  │ Model      │  │ Task Queue │                              │
│  │ Session    │  │            │                              │
│  │            │  │ (Thread    │                              │
│  │ - ONNX     │  │  Safe)     │                              │
│  │  Runtime   │  │            │                              │
│  └────────────┘  └────────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境要求

- **操作系统**: Linux (Ubuntu 22.04 / WSL2)
- **编译器**: g++ 11+ (支持 C++17)
- **CMake**: 3.18+
- **依赖库**:
  - OpenCV 4.x
  - muduo 网络库
  - ONNX Runtime
  - nlohmann/json (自动下载)
  - spdlog (自动下载)

### 2. 安装依赖

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    git

# 安装 muduo (如果未安装)
git clone https://github.com/chenshuo/muduo.git
cd muduo
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# 安装 ONNX Runtime
# 方式 1: apt (CPU 版本)
sudo apt-get install -y libonnxruntime-dev

# 方式 2: 源码编译 (支持 GPU)
# git clone --recursive https://github.com/microsoft/onnxruntime
# cd onnxruntime
# ./build.sh --config Release --build_shared_lib
# sudo make install
```

### 3. 下载 ONNX 模型

```bash
# 方式 1: 使用 Python 脚本
pip install ultralytics
python scripts/download_model.py --model yolov8n --output models/

# 方式 2: 手动下载
# 从 https://github.com/ultralytics/assets/releases 下载
# 放入 models/ 目录
```

### 4. 编译

```bash
mkdir build && cd build

# 基本编译
cmake ..
make -j$(nproc)

# 启用测试
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)

# 指定 ONNX Runtime 路径（如果未安装在默认位置）
cmake .. -DONNX_RUNTIME_ROOT=/path/to/onnxruntime
make -j$(nproc)
```

### 5. 运行

```bash
# 基本启动
./build/inference_server \
    --model models/yolov8n.onnx \
    --port 8080 \
    --log-level info

# 自定义配置
./build/inference_server \
    --model models/yolov8n.onnx \
    --port 8080 \
    --threads 4 \
    --batch-size 16 \
    --batch-timeout 5
```

### 6. 测试

```bash
# 健康检查
curl http://localhost:8080/health

# 推理测试
curl -X POST http://localhost:8080/infer \
    -F "image=@test.jpg"

# 性能测试
python scripts/benchmark_client.py \
    --image test.jpg \
    --num-requests 100 \
    --concurrency 10
```

## 📁 项目结构

```
yolo-onnx-inference/
├── CMakeLists.txt                    # 构建配置
├── README.md                         # 本文件
├── IMPLEMENTATION_PLAN.md            # 实施计划
│
├── include/inference/                # 头文件
│   ├── common/                       # 公共组件
│   │   ├── Status.hpp               # 统一状态码
│   │   ├── Result.hpp               # 统一结果类型
│   │   └── Logger.hpp               # 日志封装
│   ├── model/                       # 模型管理
│   │   ├── ModelConfig.hpp          # 模型配置
│   │   ├── ModelSession.hpp         # ONNX Session 封装
│   │   └── ModelManager.hpp         # 多模型管理
│   ├── inference/                   # 推理服务
│   │   ├── InferenceRequest.hpp     # 请求结构
│   │   ├── InferenceResponse.hpp    # 响应结构
│   │   └── InferenceService.hpp     # 服务接口
│   ├── batch/                       # 动态批处理
│   │   ├── BatchConfig.hpp          # 批处理配置
│   │   ├── BatchScheduler.hpp       # 批调度器 (核心!)
│   │   └── PendingRequest.hpp       # 待批请求
│   ├── server/                      # HTTP 服务器
│   │   ├── ServerConfig.hpp         # 服务器配置
│   │   ├── Handlers.hpp             # 请求处理
│   │   └── InferenceServer.hpp      # 服务器主类
│   ├── thread/                      # 线程工具
│   │   ├── TaskQueue.hpp            # 任务队列
│   │   └── ThreadUtils.hpp          # 线程工具
│   ├── preprocessing/               # 预处理/后处理
│   │   ├── ImagePreprocessor.hpp    # 图像预处理
│   │   └── NMS.hpp                  # 非极大值抑制
│   └── memory/                      # 内存管理
│       ├── ObjectPool.hpp           # 对象池
│       └── MemoryPool.hpp           # Tensor 内存池
│
├── src/                             # 源代码
│   ├── main.cpp                     # 程序入口
│   └── ...                          # 对应头文件的实现
│
├── tests/                           # 单元测试
├── configs/                         # 配置文件
├── scripts/                         # 辅助脚本
└── models/                          # ONNX 模型
```

## 🎯 核心概念学习

### 1. ONNX Runtime C++ API

```cpp
// Ort::Env - 全局环境
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");

// Ort::SessionOptions - 配置
Ort::SessionOptions options;
options.SetIntraOpNumThreads(4);
options.SetExecutionMode(Ort::ExecutionMode::kParallel);

// Ort::Session - 推理会话
Ort::Session session(env, "model.onnx", options);

// 运行推理
std::vector<Ort::Value> outputs = session.Run(
    Ort::RunOptions{nullptr},
    input_names, inputs, input_names.size(),
    output_names, output_names.size()
);
```

### 2. Dynamic Batching

```
时间线:
    t=0ms:  收到请求 A
    t=2ms:  收到请求 B
    t=5ms:  收到请求 C, D, E, F
    t=8ms:  收到请求 G, H → 达到 max_batch_size=8
            → 触发推理 (A, B, C, D, E, F, G, H)

    或

    t=0ms:  收到请求 X
    t=10ms: 超时 → 触发推理 (X)  // 即使只有 1 个请求
```

### 3. muduo Reactor 模式

```
EventLoop (主循环)
    ├── Acceptor (监听新连接)
    │       └── TcpConnection → channel (EPOLLIN)
    ├── TcpConnection (已建立连接)
    │       ├── Read channel → messageCallback
    │       └── Write channel → writeCompleteCallback
    └── Timer (定时器)
```

## 📊 性能优化

### 各阶段对比

| 阶段 | 描述 | 预期 QPS |
|------|------|----------|
| Baseline | 单线程，无批处理 | ~10 |
| Phase 2 | HTTP 服务器 | ~15 |
| Phase 3 | 动态批处理 (batch=8) | ~50 |
| Phase 5 | 内存池 + 优化 | ~80 |

*数据基于 CPU 推理，YOLOv8n，640×640 输入*

## 🧪 运行测试

```bash
# 编译测试
cd build
ctest --output-on-failure

# 单独运行某个测试
./test_preprocessing
./test_batch_scheduler
```

## 📝 API 文档

### HTTP 接口

#### POST /infer

提交推理请求。

**请求:**
```bash
curl -X POST http://localhost:8080/infer \
    -F "image=@test.jpg"
```

**响应:**
```json
{
  "request_id": 1,
  "status": "ok",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": {
        "x": 0.5,
        "y": 0.3,
        "width": 0.2,
        "height": 0.4
      }
    }
  ],
  "num_detections": 1,
  "timing": {
    "inference_time_ms": 12.5,
    "preprocessing_time_ms": 2.1,
    "postprocessing_time_ms": 0.5
  }
}
```

#### GET /health

健康检查。

**响应:**
```json
{
  "status": "healthy",
  "service": "inference_server",
  "version": "1.0.0"
}
```

#### GET /stats

获取服务统计。

**响应:**
```json
{
  "total_requests": 1000,
  "total_batches": 125,
  "avg_batch_size": 8.0,
  "pending_requests": 3
}
```

## 🔧 故障排除

### 问题 1: ONNX Runtime 未找到

```
CMake Warning: ONNX Runtime NOT found
```

**解决:**
```bash
# 方式 1: 安装
sudo apt-get install libonnxruntime-dev

# 方式 2: 指定路径
cmake .. -DONNX_RUNTIME_ROOT=/path/to/onnxruntime
```

### 问题 2: 模型加载失败

```
Error: Failed to load model: models/yolov8n.onnx not found
```

**解决:** 确保模型文件存在
```bash
ls -la models/
python scripts/download_model.py --model yolov8n
```

### 问题 3: muduo 链接错误

```
undefined reference to muduo::net::TcpServer
```

**解决:** 确保 muduo 已正确安装
```bash
# 检查 muduo 库
find /usr -name "libmuduo*"

# 重新安装 muduo
cd muduo/build
sudo make install
```

## 📚 学习资源

- [ONNX Runtime C++ API 文档](https://onnxruntime.ai/docs/api/cxx_api/)
- [muduo 网络库](https://github.com/chenshuo/muduo)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [C++ Concurrency in Action](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)

## 📄 许可证

MIT License

# 高性能 ONNX Runtime 推理服务引擎 - 实施计划

## 项目概述

基于 C++ 实现一个支持多模型加载、动态批处理、异步推理的高性能 ONNX 推理服务。
核心是利用你的 **C++ 网络编程 + 多线程 + OpenCV + ONNX Runtime** 技能，构建企业级 AI 推理基础设施。

### 技术栈选择（已确认）

| 组件 | 选型 | 理由 |
|------|------|------|
| **RPC 框架** | HTTP + JSON | 简单直观，便于学习 |
| **网络层** | muduo 框架 | 复用你已有的 muduo 经验，基于 Reactor 模式 |
| **模型类型** | YOLO 系列目标检测 | 输出边界框和类别，实用性强 |
| **构建系统** | CMake 3.18+ | 标准、跨平台 |
| **JSON 库** | nlohmann/json | 头文件库，零配置 |
| **深度学习** | ONNX Runtime C++ API | 跨平台、高性能 |
| **图像处理** | OpenCV | 图片解码、预处理 |

---

## 项目目录结构

```
yolo-onnx-inference/
├── CMakeLists.txt                          # 根构建配置
├── README.md                               # 项目文档
├── plan.md                                 # 原始规划文档
├── IMPLEMENTATION_PLAN.md                  # 本文件 - 实施计划
│
├── include/                                # 公共头文件
│   └── inference/
│       ├── common/
│       │   ├── Result.hpp                  # 统一结果类型（成功/失败）
│       │   ├── Status.hpp                  # 状态码定义
│       │   └── Logger.hpp                  # 日志封装
│       │
│       ├── model/
│       │   ├── ModelManager.hpp            # 模型管理器（多模型加载/卸载）
│       │   ├── ModelSession.hpp            # 单个 ONNX Session 封装
│       │   └── ModelConfig.hpp             # 模型配置（输入输出形状等）
│       │
│       ├── inference/
│       │   ├── InferenceService.hpp        # 推理服务主接口
│       │   ├── InferenceRequest.hpp        # 推理请求数据结构
│       │   └── InferenceResponse.hpp       # 推理响应数据结构
│       │
│       ├── batch/
│       │   ├── BatchScheduler.hpp          # 动态批调度器（核心算法）
│       │   ├── BatchConfig.hpp             # 批处理配置参数
│       │   └── PendingRequest.hpp          # 待处理请求
│       │
│       ├── server/
│       │   ├── InferenceServer.hpp         # 推理服务器（HTTP + muduo）
│       │   ├── Handlers.hpp                # HTTP 请求处理器
│       │   └── ServerConfig.hpp            # 服务器配置
│       │
│       ├── thread/
│       │   ├── TaskQueue.hpp               # 任务队列（有界、线程安全）
│       │   └── ThreadUtils.hpp             # 线程工具
│       │
│       ├── preprocessing/
│       │   ├── ImagePreprocessor.hpp       # 图像预处理（Resize/Normalize）
│       │   └── NMS.hpp                     # NMS 后处理
│       │
│       └── memory/
│           ├── MemoryPool.hpp              # 对象池（Tensor/Mat 复用）
│           └── ObjectPool.hpp              # 通用对象池
│
├── src/                                    # 源代码
│   ├── common/
│   │   └── Logger.cpp
│   ├── model/
│   │   ├── ModelManager.cpp
│   │   └── ModelSession.cpp
│   ├── inference/
│   │   └── InferenceService.cpp
│   ├── batch/
│   │   └── BatchScheduler.cpp             # ★ 核心：动态批处理
│   ├── server/
│   │   ├── InferenceServer.cpp            # ★ 核心：HTTP 服务器
│   │   └── Handlers.cpp                   # ★ 核心：请求处理
│   ├── preprocessing/
│   │   ├── ImagePreprocessor.cpp
│   │   └── NMS.cpp
│   └── main.cpp                            # 程序入口
│
├── tests/                                  # 测试
│   ├── CMakeLists.txt
│   ├── test_model_loading.cpp              # 模型加载测试
│   ├── test_batch_scheduler.cpp            # 批调度器测试
│   ├── test_preprocessing.cpp              # 预处理测试
│   └── benchmark.cpp                       # 性能基准测试
│
├── configs/                                # 配置文件
│   ├── server_config.json                  # 服务器配置
│   └── model_config.json                   # 模型配置
│
├── models/                                 # ONNX 模型（gitignore）
│   └── yolov8n.onnx
│
├── scripts/                                # 脚本
│   ├── download_model.py                   # 模型下载脚本
│   ├── benchmark_client.py                # 基准测试客户端
│   └── test_server.sh                     # 服务器测试脚本
│
├── third_party/                            # 第三方依赖（git submodule）
│   ├── muduo/
│   ├── nlohmann_json/
│   ├── onnxruntime/
│   └── opencv/
│
├── build/                                  # 构建输出（gitignore）
├── .gitignore
└── docs/                                   # 文档
    └── architecture.md                     # 架构设计文档
```

---

## 核心组件设计

### 1. InferenceRequest / InferenceResponse

```cpp
// 推理请求
struct InferenceRequest {
    uint64_t id;               // 请求 ID
    cv::Mat image;             // 输入图像
    std::string model_name;    // 模型名称
    std::chrono::steady_clock::time_point arrival_time; // 到达时间
    std::promise<InferenceResponse> promise;             // 异步返回机制
};

// 推理响应
struct InferenceResponse {
    uint64_t request_id;       // 对应请求 ID
    std::vector<Detection> detections;  // 检测结果
    double inference_time_ms;  // 推理耗时
    Status status;             // 状态（成功/失败）

    struct Detection {
        float x, y, width, height;  // 边界框
        float confidence;            // 置信度
        int class_id;                // 类别 ID
        std::string class_name;      // 类别名称
    };
};
```

### 2. ModelSession（ONNX Session 封装）

```cpp
class ModelSession {
public:
    // 加载 ONNX 模型
    Status Load(const std::string& model_path);

    // 运行推理（输入：cv::Mat，输出：原始输出张量）
    Status Run(const cv::Mat& input, std::vector<OrtValue>& outputs);

    // 获取输入/输出信息
    ModelInfo GetModelInfo() const;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    ModelInfo model_info_;
};
```

**你的学习任务：** 理解 ONNX Runtime C++ API 的核心概念：
- `Ort::Env` — ONNX 运行时环境（全局共享）
- `Ort::SessionOptions` — 会话配置（执行模式、GPU 启用、线程数）
- `Ort::Session` — 模型加载后的推理会话
- `Ort::Value` — 运行时张量类型

### 3. BatchScheduler（动态批调度器）⭐ 核心算法

```cpp
class BatchScheduler {
public:
    // 配置参数
    struct Config {
        size_t max_batch_size = 32;     // 最大批大小
        std::chrono::milliseconds timeout = 5ms; // 等待超时
        size_t min_batch_size = 1;      // 最小批大小
    };

    // 提交请求，返回异步响应
    std::future<InferenceResponse> Submit(InferenceRequest&& request);

    // 启动批调度器
    void Start();

    // 停止批调度器
    void Stop();

private:
    // ★ 核心：批处理循环
    void BatchingLoop();

    // 将一批请求送入模型推理
    void ProcessBatch(std::vector<InferenceRequest>& batch);

    // 将原始输出拆分回各个请求
    void SplitResults(const std::vector<InferenceRequest>& batch,
                      const std::vector<OrtValue>& outputs);

    TaskQueue<InferenceRequest> pending_queue_;  // 待批队列
    std::vector<InferenceRequest> current_batch_; // 当前批
    std::mutex batch_mutex_;
    std::condition_variable cv_;
    Config config_;
    std::thread batching_thread_;
    ModelSession* model_session_;
};
```

**你的学习任务：** 理解 Dynamic Batching 的核心思想：
- 在时间窗口内收集多个请求
- 凑够 batch_size 或超时 → 触发推理
- 将一个大 batch 的输入送入 ONNX Runtime
- 将输出拆分回各个请求
- 平衡延迟与吞吐量的 trade-off

### 4. InferenceServer（HTTP 服务器 + muduo）

```cpp
class InferenceServer {
public:
    // 配置并启动服务器
    Status Start(const ServerConfig& config);

    // 优雅关闭
    void Stop();

private:
    // muduo 回调
    void HandleConnection(const muduo::net::TcpConnectionPtr& conn);
    void HandleMessage(const muduo::net::TcpConnectionPtr& conn,
                       muduo::net::Buffer* buffer);
    void SendResponse(const muduo::net::TcpConnectionPtr& conn,
                      const InferenceResponse& response);

    // HTTP 协议解析
    HttpMessage ParseHttpRequest(const std::string& data);
    std::string SerializeResponse(const InferenceResponse& response);

    muduo::net::EventLoop loop_;
    muduo::net::TcpServer server_;
    InferenceService inference_service_;  // 推理服务
    BatchScheduler batch_scheduler_;       // 批调度器
    ModelManager model_manager_;           // 模型管理器
};
```

**你的学习任务：** 理解 muduo Reactor 模式：
- `EventLoop` — 事件循环（主循环，处理 IO 事件）
- `TcpServer` — TCP 服务器（管理多个连接）
- `TcpConnection` — TCP 连接（读写事件回调）
- `Buffer` — 网络缓冲区（读写数据）
- 如何将 HTTP 请求映射到推理流程

### 5. InferenceService（推理服务）

```cpp
class InferenceService {
public:
    // 初始化服务（加载模型、启动调度器）
    Status Initialize(const ServerConfig& config);

    // 提交推理请求（异步）
    std::future<InferenceResponse> SubmitRequest(InferenceRequest&& request);

    // 获取服务状态/指标
    ServiceMetrics GetMetrics() const;

    // 优雅关闭
    void Shutdown();

private:
    ModelManager model_manager_;
    BatchScheduler batch_scheduler_;
    MemoryPool memory_pool_;
    std::atomic<bool> running_{false};
    ServiceMetrics metrics_;
};
```

### 6. MemoryPool（对象池）

```cpp
template<typename T>
class ObjectPool {
public:
    // 创建池，预分配指定数量的对象
    explicit ObjectPool(size_t pool_size);

    // 从池中获取对象（线程安全）
    T* Acquire();

    // 归还对象到池中（线程安全）
    void Release(T* obj);

    // 重置对象状态
    virtual void Reset(T* obj) = 0;

private:
    std::vector<T> pool_;           // 池中对象
    std::vector<bool> in_use_;      // 使用标记
    std::mutex mutex_;
    std::atomic<size_t> acquired_{0};
};

// 专门用于 Tensor 内存的池
class MemoryPool : public ObjectPool<AlignedBuffer> {
public:
    void Reset(AlignedBuffer* buf) override {
        buf->Reset();
    }
};
```

---

## 数据流（请求到响应）

```
[客户端] --HTTP POST--> [InferenceServer]
                              |
                              v
                        [Handlers.cpp]
                        (解析 HTTP 请求，提取图片数据)
                              |
                              v
                        [InferenceRequest]
                        (构造请求对象，附带 promise)
                              |
                              v
                        [BatchScheduler::Submit()]
                        (加入待批队列，future.wait())
                              |
                    ┌─────────┴─────────┐
                    v                   v
            [等待时间窗口]       [凑够 batch_size]
                    |                   |
                    v                   v
                [触发批处理] <──────────┘
                    |
                    v
            [BatchScheduler::ProcessBatch()]
            (将 batch 图像预处理为 Tensor)
                    |
                    v
            [ModelSession::Run(tensor_batch)]
            (ONNX Runtime 推理)
                    |
                    v
            [SplitResults()]
            (将输出拆分回各个请求)
                    |
                    v
            [NMS 后处理]
            (非极大值抑制)
                    |
                    v
            [promise.set_value()]
            (异步返回结果)
                    |
                    v
            [InferenceServer::SendResponse()]
            (序列化为 JSON，HTTP 响应)
                    |
                    v
              [客户端接收]
```

---

## 实施阶段

### Phase 1：基础框架搭建 (Day 1-2)

**目标：** 建立项目骨架，实现最简单的 "加载模型 → 推理 → 返回" 流程

**实施内容：**
1. **初始化 Git 仓库** + `.gitignore`
2. **CMakeLists.txt** — 配置依赖（ONNX Runtime、OpenCV、muduo、nlohmann/json）
3. **项目目录结构** — 创建所有目录
4. **common/Logger.hpp** — 日志封装（使用 spdlog 或自定义）
5. **common/Status.hpp** — 状态码和错误处理
6. **model/ModelSession** — ONNX Session 封装
7. **main.cpp** — 最简单的单线程推理测试
8. **test_model_loading.cpp** — 验证模型能正确加载

**验收标准：** `./build/inference_server --model yolov8n.onnx --image test.jpg` 能输出检测结果

### Phase 2：HTTP 服务器 + muduo 集成 (Day 3-5)

**目标：** 实现 HTTP 服务器，支持接收请求并返回结果

**实施内容：**
1. **server/ServerConfig.hpp** — 服务器配置（端口、工作线程数）
2. **server/Handlers.hpp/cpp** — HTTP 请求解析和响应序列化
3. **server/InferenceServer.hpp/cpp** — muduo 集成
4. **inference/InferenceRequest.hpp** — 请求数据结构
5. **inference/InferenceResponse.hpp** — 响应数据结构
6. **scripts/benchmark_client.py** — Python 测试客户端

**验收标准：** 启动服务器后，`curl -X POST http://localhost:8080/infer -F "image=@test.jpg"` 能返回 JSON 结果

### Phase 3：动态批处理 (Day 6-8)

**目标：** 实现 Dynamic Batching，提升吞吐量

**实施内容：**
1. **batch/BatchConfig.hpp** — 批处理配置
2. **batch/BatchScheduler.hpp/cpp** — 动态批调度器（核心算法）
3. **thread/TaskQueue.hpp** — 有界任务队列
4. **inference/InferenceService.hpp/cpp** — 推理服务（整合所有组件）
5. **tests/test_batch_scheduler.cpp** — 批调度器单元测试

**验收标准：** 压测显示 QPS 比 Phase 2 提升 ≥ 50%（batch_size > 1 时）

### Phase 4：预处理 + 后处理 (Day 9-10)

**目标：** 实现 YOLO 专用的预处理和后处理

**实施内容：**
1. **preprocessing/ImagePreprocessor.hpp/cpp** — 图像预处理（Resize、Normalize、BGR→RGB）
2. **preprocessing/NMS.hpp/cpp** — 非极大值抑制
3. **YOLO 输出解析** — 将 ONNX 输出转换为 Detection 列表
4. **benchmark.py** 更新 — 对比各阶段性能

**验收标准：** 检测结果 mAP 与 ONNX Runtime 官方示例一致

### Phase 5：性能优化 + 内存池 (Day 11-14)

**目标：** 优化性能，实现对象池复用

**实施内容：**
1. **memory/ObjectPool.hpp** — 通用对象池
2. **memory/MemoryPool.hpp** — Tensor 内存池
3. **ModelManager** — 多模型管理
4. **优雅关闭** — 信号处理、等待队列排空
5. **服务指标** — QPS、延迟、批大小统计
6. **benchmark** — 完整性能对比

**验收标准：** 性能报告展示各阶段对比数据

---

## 依赖安装指南

```bash
# Ubuntu/WSL 上安装依赖

# 1. OpenCV
sudo apt-get install libopencv-dev

# 2. ONNX Runtime (C++)
# 方式 A: apt 安装
sudo apt-get install libonnxruntime-dev

# 方式 B: 源码编译（推荐，可启用 GPU）
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib

# 3. muduo 网络库
git clone https://github.com/chenshuo/muduo.git
cd muduo
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# 4. nlohmann/json (头文件库，无需安装)
# 在 CMakeLists.txt 中使用 FetchContent 获取

# 5. spdlog (日志库)
sudo apt-get install libspdlog-dev
```

---

## Commit 规范

遵循 Conventional Commits 规范：

```
<type>(<scope>): <description>

type:
  feat     - 新功能
  fix      - 修复 bug
  refactor - 代码重构
  docs     - 文档更新
  test     - 测试相关
  chore    - 构建/工具变更
  perf     - 性能优化
```

示例：
```
feat(model): 添加 ONNX Runtime Session 封装
feat(server): 实现 muduo HTTP 服务器
feat(batch): 实现动态批调度器
perf(memory): 添加对象池复用 Tensor
test(batch): 添加批调度器单元测试
```

---

## 学习检查点

每个 Phase 完成后，确保你能回答以下问题：

**Phase 1 后：**
- [ ] ONNX Runtime C++ API 的核心对象有哪些？各自作用是什么？
- [ ] 如何将 cv::Mat 转换为 ONNX Tensor？
- [ ] Ort::Session 是线程安全的吗？为什么？

**Phase 2 后：**
- [ ] muduo Reactor 模式的工作原理是什么？
- [ ] 如何处理 HTTP POST 中的 multipart/form-data？
- [ ] 如何避免 TCP 粘包/拆包问题？

**Phase 3 后：**
- [ ] Dynamic Batching 如何平衡延迟和吞吐量？
- [ ] 为什么需要 std::promise/std::future？
- [ ] 如何处理批处理中的不同尺寸输入？

**Phase 4 后：**
- [ ] YOLO 模型的输入预处理步骤有哪些？
- [ ] NMS 算法的原理是什么？
- [ ] 如何评估检测模型的性能？

**Phase 5 后：**
- [ ] 对象池如何解决内存碎片问题？
- [ ] 如何优雅地关闭一个异步服务？
- [ ] 哪些指标能反映推理服务的性能？

---

## 性能基准测试设计

```python
# benchmark_client.py - 基准测试客户端
import requests
import time
import concurrent.futures
import json

def benchmark(server_url, image_path, num_requests=100, concurrency=10):
    with open(image_path, 'rb') as f:
        image_data = f.read()

    start = time.time()
    latencies = []

    def send_request():
        t0 = time.time()
        response = requests.post(f"{server_url}/infer", files={"image": image_data})
        latency = time.time() - t0
        latencies.append(latency)
        return response.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # 等待完成并捕获异常

    total_time = time.time() - start
    qps = num_requests / total_time

    # 计算延迟分位数
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"QPS: {qps:.2f}")
    print(f"Latency P50: {p50*1000:.2f}ms")
    print(f"Latency P99: {p99*1000:.2f}ms")
    print(f"Total time: {total_time:.2f}s")
```

---

## 最终交付物

1. **可执行推理服务** (`inference_server`)
2. **完整的 CMake 构建系统**
3. **单元测试套件**
4. **性能基准测试**（Python 客户端 + 对比报告）
5. **详细的 README**（安装、使用、架构说明）
6. **Git 提交历史**（规范、可追溯）
7. **学习文档**（关键概念解释、常见问题）

---

## 与 plan.md 的对应

| plan.md 中的核心要求 | 本计划中的实现 |
|---------------------|--------------|
| 无锁队列/工作窃取线程池 | TaskQueue + muduo 线程池 |
| Dynamic Batching | BatchScheduler |
| epoll/io_uring + Reactor | muduo Reactor |
| 零拷贝优化 | MemoryPool + Buffer 传递 |
| 多模型加载 | ModelManager |
| 内存池复用 cv::Mat/Tensor | ObjectPool |

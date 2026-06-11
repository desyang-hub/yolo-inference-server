/**
 * @file BatchScheduler.hpp
 * @brief 动态批调度器
 *
 * 这是整个推理服务的核心组件，实现了 Dynamic Batching 算法。
 *
 * 学习要点：
 * ============================================
 * 架构设计
 * ============================================
 *
 * BatchScheduler 采用单线程批处理模式：
 *
 *   ┌─────────────┐     ┌──────────────────┐     ┌────────────┐
 *   │ HTTP Handler │ ──▶│ pending_queue    │ ──▶│ Batch Thread │
 *   │ (multi-thread│     │ (thread-safe)    │     │ (single)    │
 *   └─────────────┘     └──────────────────┘     └──────┬─────┘
 *                                                        │
 *                                                        ▼
 *                                             ┌──────────────────┐
 *                                             │ 1. Collect batch │
 *                                             │ 2. Preprocess    │
 *                                             │ 3. Run inference │
 *                                             │ 4. Postprocess   │
 *                                             │ 5. Split results │
 *                                             └──────────────────┘
 *
 * ============================================
 * 线程安全
 * ============================================
 * - Submit() 从多个 HTTP Handler 线程调用 → 需要线程安全
 * - ProcessBatch() 在单线程中运行 → 不需要额外锁
 * - pending_queue 使用 mutex + condition_variable
 *
 * ============================================
 * std::condition_variable
 * ============================================
 * - 用于在 Submit() 和 BatchingLoop() 之间通信
 * - Submit() 调用 cv.notify_one() 唤醒等待中的批处理线程
 * - BatchingLoop() 在 cv.wait_for() 中等待，超时或唤醒后处理 batch
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "inference/batch/BatchConfig.hpp"
#include "inference/batch/PendingRequest.hpp"
#include "inference/inference/InferenceRequest.hpp"
#include "inference/inference/InferenceResponse.hpp"

// ONNX Runtime C++ API 前向声明
#ifdef ONNXRUNTIME_FOUND
#include <onnxruntime_cxx_api.h>
#endif

namespace inference {

// 前向声明
class ModelSession;
class ImagePreprocessor;
class NMS;

/**
 * @brief 动态批调度器
 *
 * 核心职责：
 * 1. 接收推理请求
 * 2. 在时间窗口内收集请求
 * 3. 凑够 batch_size 或超时 → 触发推理
 * 4. 将结果拆分回各个请求
 */
class BatchScheduler {
public:
    explicit BatchScheduler(const BatchConfig& config = BatchConfig::Balanced());
    ~BatchScheduler();

    // 禁止拷贝
    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    /**
     * @brief 初始化（设置模型和预处理器）
     */
    void Initialize(ModelSession* session,
                    ImagePreprocessor* preprocessor,
                    NMS* postprocessor);

    /**
     * @brief 提交推理请求
     * @return std::future<InferenceResponse> 异步结果
     *
     * 调用者通过 future.get() 等待结果。
     * 结果会在批处理完成后通过 promise.set_value() 返回。
     */
    std::future<InferenceResponse> Submit(PendingRequest&& request);

    /**
     * @brief 启动批调度器
     */
    void Start();

    /**
     * @brief 停止批调度器（等待当前批处理完成）
     */
    void Stop();

    /**
     * @brief 获取当前等待中的请求数
     */
    size_t PendingCount() const;

    /**
     * @brief 获取统计信息
     */
    struct Stats {
        size_t total_requests = 0;      // 总请求数
        size_t total_batches = 0;       // 总批次数
        double avg_batch_size = 0.0;    // 平均批大小
        double total_inference_time_ms = 0.0; // 总推理时间
    };

    Stats GetStats() const;

private:
    /**
     * @brief 批处理循环（运行在独立线程中）
     *
     * 核心算法：
     * 1. 等待 pending_queue 中有数据 或 超时
     * 2. 收集尽可能多的请求（最多 max_batch_size）
     * 3. 预处理 → 推理 → 后处理
     * 4. 拆分结果，通过 promise 返回
     */
    void BatchingLoop();

    /**
     * @brief 处理一个批次的请求
     */
    void ProcessBatch(std::vector<PendingRequest>& batch);

    // 配置
    BatchConfig config_;

    // 模型和处理器
    ModelSession* session_ = nullptr;
    ImagePreprocessor* preprocessor_ = nullptr;
    NMS* postprocessor_ = nullptr;

    // 待处理队列
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<PendingRequest> pending_queue_;

    // 线程
    std::atomic<bool> running_{false};
    std::thread batching_thread_;

    // 统计
    mutable std::mutex stats_mutex_;
    Stats stats_;

#ifdef ONNXRUNTIME_FOUND
    // ONNX 推理环境缓存
    std::unique_ptr<Ort::MemoryInfo> memory_info_;  // CPU 内存信息
    std::vector<float> input_tensor_pool_;          // 复用输入缓冲区
    std::vector<int64_t> current_batch_shape_;      // 当前 batch 的形状
#endif
};

} // namespace inference

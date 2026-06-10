/**
 * @file BatchScheduler.cpp
 * @brief 动态批调度器实现
 *
 * 这是整个推理服务的核心算法实现。
 */

#include "inference/batch/BatchScheduler.hpp"
#include "inference/common/Logger.hpp"
#include "inference/model/ModelSession.hpp"
#include "inference/preprocessing/ImagePreprocessor.hpp"
#include "inference/preprocessing/NMS.hpp"

#include <algorithm>
#include <numeric>

namespace inference {

BatchScheduler::BatchScheduler(const BatchConfig& config)
    : config_(config) {
    if (!config_.IsValid()) {
        LOG_WARNING("Invalid batch config, using defaults");
        config_ = BatchConfig::Balanced();
    }
    LOG_INFO("BatchScheduler created. max_batch={}, timeout={}ms",
             config_.max_batch_size, config_.timeout.count());
}

BatchScheduler::~BatchScheduler() {
    Stop();
}

void BatchScheduler::Initialize(ModelSession* session,
                                 ImagePreprocessor* preprocessor,
                                 NMS* postprocessor) {
    session_ = session;
    preprocessor_ = preprocessor;
    postprocessor_ = postprocessor;
    LOG_INFO("BatchScheduler initialized with model and processors");
}

std::future<InferenceResponse> BatchScheduler::Submit(PendingRequest&& request) {
    // 创建 future 供调用者等待
    auto future = request.promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_queue_.push_back(std::move(request));
    }

    // 通知批处理线程有新请求
    queue_cv_.notify_one();

    ++stats_.total_requests;
    return future;
}

void BatchScheduler::Start() {
    if (running_.exchange(true)) {
        LOG_WARNING("BatchScheduler already running");
        return;
    }

    batching_thread_ = std::thread(&BatchScheduler::BatchingLoop, this);
    LOG_INFO("BatchScheduler started (thread id: {})",
             batching_thread_.get_id());
}

void BatchScheduler::Stop() {
    if (!running_.exchange(false)) {
        return;
    }

    // 唤醒等待中的批处理线程
    queue_cv_.notify_one();

    if (batching_thread_.joinable()) {
        batching_thread_.join();
        LOG_INFO("BatchScheduler stopped");
    }

    // 处理剩余的请求（直接返回超时）
    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (auto& req : pending_queue_) {
        InferenceResponse response;
        response.request_id = req.id;
        response.status = Status::DeadlineExceeded("Scheduler stopped");
        req.promise.set_value(std::move(response));
    }
    pending_queue_.clear();
}

size_t BatchScheduler::PendingCount() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return pending_queue_.size();
}

BatchScheduler::Stats BatchScheduler::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats s = stats_;
    if (s.total_batches > 0) {
        s.avg_batch_size = static_cast<double>(s.total_requests) / s.total_batches;
    }
    return s;
}

void BatchScheduler::BatchingLoop() {
    LOG_INFO("Batching loop started");

    while (running_) {
        // 步骤 1: 等待请求
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // 等待条件：队列有数据 或 超时 或 停止
        queue_cv_.wait_for(lock, config_.timeout, [this]() {
            return !pending_queue_.empty() || !running_;
        });

        if (!running_ && pending_queue_.empty()) {
            break;
        }

        // 步骤 2: 收集一个 batch
        // 非阻塞地收集所有待处理请求（不超过 max_batch_size）
        size_t batch_size = std::min(pending_queue_.size(), config_.max_batch_size);

        // 从队列中提取 batch
        std::vector<PendingRequest> batch;
        batch.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(std::move(pending_queue_.back()));
            pending_queue_.pop_back();
        }

        lock.unlock();

        if (batch.empty()) {
            continue;
        }

        // 步骤 3: 处理 batch
        ProcessBatch(batch);

        // 更新统计
        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            ++stats_.total_batches;
        }
    }

    LOG_INFO("Batching loop exited");
}

void BatchScheduler::ProcessBatch(std::vector<PendingRequest>& batch) {
    if (!session_ || !preprocessor_) {
        // 模型未初始化，直接返回错误
        for (auto& req : batch) {
            InferenceResponse response;
            response.request_id = req.id;
            response.status = Status::Unavailable("Model or preprocessor not initialized");
            req.promise.set_value(std::move(response));
        }
        return;
    }

    auto batch_start = std::chrono::steady_clock::now();

    // ========================================
    // 步骤 1: 预处理所有图像
    // ========================================
    double preprocessing_time_ms = 0;
    {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<cv::Mat> preprocessed_images;
        preprocessed_images.reserve(batch.size());

        for (auto& req : batch) {
            // 记录进入 batch 的时间
            req.batch_time = std::chrono::steady_clock::now();

            // 预处理（Resize + Normalize + Format）
            auto status = preprocessor_->Preprocess(req.original_image, req.preprocessed_image);
            if (!status.ok()) {
                LOG_ERROR("Preprocessing failed for request {}: {}", req.id, status.ToString());
                InferenceResponse response;
                response.request_id = req.id;
                response.status = status;
                req.promise.set_value(std::move(response));
                continue;
            }
            preprocessed_images.push_back(req.preprocessed_image);
        }

        auto t1 = std::chrono::steady_clock::now();
        preprocessing_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    LOG_DEBUG("Batch[{}] preprocessing: {}ms (size={})",
              batch.size(), preprocessing_time_ms, batch.size());

    // ========================================
    // 步骤 2: 构建输入 Tensor 并推理
    // ========================================
    // TODO: 将预处理的图像合并为一个 batch Tensor
    // 这需要 ONNX Runtime API，在后续 Phase 实现

    double inference_time_ms = 0;
    {
        auto t0 = std::chrono::steady_clock::now();

        // TODO: 调用 session_->Run() 进行推理
        // 此处为框架代码，实际推理在 Phase 3 完善

        auto t1 = std::chrono::steady_clock::now();
        inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ========================================
    // 步骤 3: 后处理（NMS）
    // ========================================
    double postprocessing_time_ms = 0;
    {
        auto t0 = std::chrono::steady_clock::now();

        // TODO: 对推理结果进行 NMS 后处理
        // 在 Phase 4 实现

        auto t1 = std::chrono::steady_clock::now();
        postprocessing_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ========================================
    // 步骤 4: 返回结果
    // ========================================
    for (auto& req : batch) {
        req.complete_time = std::chrono::steady_clock::now();

        InferenceResponse response;
        response.request_id = req.id;
        response.inference_time_ms = inference_time_ms;
        response.preprocessing_time_ms = preprocessing_time_ms;
        response.postprocessing_time_ms = postprocessing_time_ms;
        response.status = Status::Ok();

        // TODO: 填充实际的检测结果
        // 在 Phase 4 实现

        req.promise.set_value(std::move(response));
    }

    auto batch_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

    LOG_INFO("Batch[{}] completed: {}ms (pre={}ms, infer={}ms, post={}ms)",
             batch.size(), total_time, preprocessing_time_ms,
             inference_time_ms, postprocessing_time_ms);
}

} // namespace inference

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

#ifdef ONNXRUNTIME_FOUND
    // 初始化 ONNX 推理环境
    if (session_ && session_->IsLoaded()) {
        auto input_shape = session_->GetInputShape();

        // 计算输入 tensor 大小：batch × 3 × H × W
        size_t input_size = 1;
        for (auto dim : input_shape) {
            input_size *= dim;
        }
        input_tensor_pool_.resize(input_size);

        current_batch_shape_.resize(input_shape.size());
        std::copy(input_shape.begin(), input_shape.end(), current_batch_shape_.begin());

        initialized_ = true;

        LOG_INFO("BatchScheduler ONNX runtime initialized: input_shape={}",
                 fmt::join(input_shape, "x"));
    }
#endif

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
    LOG_INFO("BatchScheduler started");
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
    std::vector<cv::Mat> preprocessed_images;
    preprocessed_images.reserve(batch.size());
    {
        auto t0 = std::chrono::steady_clock::now();
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
    std::vector<Status> batch_statuses(batch.size(), Status::Ok());
    std::vector<std::vector<Detection>> batch_detections(batch.size());
    double inference_time_ms = 0.0;

#ifdef ONNXRUNTIME_FOUND
    if (session_ && session_->IsLoaded() && initialized_) {
        auto t0 = std::chrono::steady_clock::now();

        // 2.1: 准备输入数据 - 将预处理的图像合并为 batch Tensor
        current_batch_shape_[0] = static_cast<int64_t>(batch.size());  // batch_size

        size_t input_tensor_size = 1;
        for (auto dim : current_batch_shape_) {
            input_tensor_size *= dim;
        }
        input_tensor_pool_.resize(input_tensor_size);

        // 将预处理的图像复制到输入 tensor
        // 预处理后的图像格式：HWC (height x width x 3), CV_32F, 归一化到 0-1
        // 需要转换为 CHW 格式用于 ONNX 输入 [batch, 3, H, W]
        float* input_data = input_tensor_pool_.data();
        size_t img_h = preprocessed_images[0].rows;
        size_t img_w = preprocessed_images[0].cols;
        size_t img_hw = img_h * img_w;

        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& img = preprocessed_images[i];
            size_t img_offset = i * 3 * img_hw;
            const float* hwc = img.ptr<float>();

            // HWC -> CHW 转换
            for (size_t y = 0; y < img_h; ++y) {
                for (size_t x = 0; x < img_w; ++x) {
                    size_t hw = y * img_w + x;
                    // Channel 0
                    input_data[img_offset + 0 * img_hw + hw] = hwc[hw * 3 + 0];
                    // Channel 1
                    input_data[img_offset + 1 * img_hw + hw] = hwc[hw * 3 + 1];
                    // Channel 2
                    input_data[img_offset + 2 * img_hw + hw] = hwc[hw * 3 + 2];
                }
            }
        }

        // 创建 Ort::Value 输入张量 - 使用 AllocatorWithDefaultOptions
        std::vector<Ort::Value> ort_inputs;

        // 先创建一个空的 tensor 用于分配内存
        auto value = Ort::Value::CreateTensor<float>(
            Ort::AllocatorWithDefaultOptions{},  // 使用默认 allocator
            current_batch_shape_.data(),
            static_cast<size_t>(current_batch_shape_.size())
        );

        // 获取可修改的数据指针
        float* data = value.GetTensorMutableData<float>();

        // 复制预处理数据到 tensor
        std::memcpy(data, input_data, input_tensor_size * sizeof(float));

        ort_inputs.push_back(std::move(value));

        // 2.2: 执行推理 - 准备输入输出名称指针
        std::vector<std::string> input_names_str = session_->GetInputNames();
        std::vector<std::string> output_names_str = session_->GetOutputNames();
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        for (const auto& name : input_names_str) {
            input_names.push_back(name.c_str());
        }
        for (const auto& name : output_names_str) {
            output_names.push_back(name.c_str());
        }

        std::vector<Ort::Value> ort_outputs;
        auto status = session_->Run(input_names, ort_inputs, output_names, ort_outputs);

        auto t1 = std::chrono::steady_clock::now();
        inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (status.ok() && !ort_outputs.empty()) {
            // 2.3: 解析推理结果
            auto& output_tensor = ort_outputs[0];
            auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();

            // YOLOv8 输出格式：[batch, 84, 8400] 或 [batch, 25200, 84]
            // 84 = 4 (box) + 80 (classes)
            float* output_data = output_tensor.GetTensorMutableData<float>();

            // 解析 batch 中每个请求的输出
            int64_t num_anchors = shape.size() >= 3 ? shape[2] : 8400;
            int64_t num_channels = shape.size() >= 2 ? shape[1] : 84;

            for (size_t batch_idx = 0; batch_idx < batch.size(); ++batch_idx) {
                float* batch_output = output_data + batch_idx * num_anchors * num_channels;
                std::vector<Detection> raw_detections;

                // 解析每个 anchor 的预测
                for (int64_t anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
                    float* det = batch_output + anchor_idx * num_channels;

                    // 提取边界框（归一化坐标：x_center, y_center, width, height）
                    float x_center = det[0];
                    float y_center = det[1];
                    float width = det[2];
                    float height = det[3];

                    // 提取最高置信度类别
                    float max_conf = 0.0f;
                    int class_id = -1;
                    for (int c = 0; c < 80; ++c) {
                        float conf = det[4 + c];
                        if (conf > max_conf) {
                            max_conf = conf;
                            class_id = c;
                        }
                    }

                    // 应用置信度阈值（默认 0.25，与 YOLOv8 一致）
                    if (max_conf > 0.25f) {
                        Detection det_result;
                        det_result.x_center = x_center;
                        det_result.y_center = y_center;
                        det_result.width = width;
                        det_result.height = height;
                        det_result.confidence = max_conf;
                        det_result.class_id = class_id;
                        if (class_id >= 0 && class_id < static_cast<int>(CLASS_NAMES.size())) {
                            det_result.class_name = CLASS_NAMES[class_id];
                        }
                        raw_detections.push_back(det_result);
                    }
                }

                // 2.4: 应用 NMS 后处理
                if (!raw_detections.empty() && postprocessor_) {
                    batch_detections[batch_idx] = postprocessor_->Process(raw_detections);
                }

                LOG_DEBUG("Batch[{}] request[{}] raw detections: {}, after NMS: {}",
                         batch.size(), batch_idx, raw_detections.size(),
                         batch_detections[batch_idx].size());
            }

            if (!status.ok()) {
                for (size_t i = 0; i < batch.size(); ++i) {
                    batch_statuses[i] = status;
                }
            }
        } else {
            LOG_ERROR("Inference failed or no outputs: {}", status.ToString());
            for (size_t i = 0; i < batch.size(); ++i) {
                batch_statuses[i] = status;
            }
        }
    } else {
        LOG_WARNING("ONNX runtime not available or model not loaded");
        inference_time_ms = 0.0;
        for (size_t i = 0; i < batch.size(); ++i) {
            batch_statuses[i] = Status::Unavailable("Model not loaded");
        }
    }
#else
    // ONNX Runtime 未安装时模拟推理
    LOG_WARNING("ONNX Runtime not available - simulating inference");
    inference_time_ms = 0.0;
    for (size_t i = 0; i < batch.size(); ++i) {
        batch_statuses[i] = Status::Unavailable("ONNX Runtime not found");
    }
#endif

    // ========================================
    // 步骤 3: 后处理（NMS）- 已在步骤 2 中完成
    // ========================================
    double postprocessing_time_ms = 0.0;
    if (!batch_detections[0].empty()) {
        auto t0 = std::chrono::steady_clock::now();
        // NMS 已在上面执行
        auto t1 = std::chrono::steady_clock::now();
        postprocessing_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ========================================
    // 步骤 4: 返回结果
    // ========================================
    for (size_t i = 0; i < batch.size(); ++i) {
        batch[i].complete_time = std::chrono::steady_clock::now();

        InferenceResponse response;
        response.request_id = batch[i].id;
        response.inference_time_ms = inference_time_ms;
        response.preprocessing_time_ms = preprocessing_time_ms;
        response.postprocessing_time_ms = postprocessing_time_ms;

        // 填充检测结果
        response.detections = std::move(batch_detections[i]);

        // 处理错误状态
        if (!batch_statuses[i].ok()) {
            response.status = batch_statuses[i];
        } else {
            response.status = Status::Ok();
        }

        batch[i].promise.set_value(std::move(response));
    }

    auto batch_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

    LOG_INFO("Batch[{}] completed: {}ms (pre={}ms, infer={}ms, post={}ms)",
             batch.size(), total_time, preprocessing_time_ms,
             inference_time_ms, postprocessing_time_ms);
}

} // namespace inference

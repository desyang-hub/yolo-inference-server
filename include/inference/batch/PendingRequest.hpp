/**
 * @file PendingRequest.hpp
 * @brief 待批处理请求
 *
 * 学习要点：
 * - 为什么需要 PendingRequest 而不是直接用 InferenceRequest？
 *   PendingRequest 包装了 InferenceRequest 和批处理相关的数据：
 *   - 预处理后的 Tensor 数据
 *   - 原始尺寸信息（用于后处理时的坐标转换）
 *   - 原始图像尺寸（YOLO 检测框需要还原到原图尺寸）
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "inference/inference/InferenceRequest.hpp"
#include "inference/inference/InferenceResponse.hpp"

namespace inference {

/**
 * @brief 待批处理请求
 *
 * 在 BatchScheduler 中，请求经历以下阶段：
 * 1. HTTP Handler 创建 InferenceRequest
 * 2. 提交到 BatchScheduler，转换为 PendingRequest
 * 3. 在 pending_queue 中等待批处理
 * 4. 被收集到 batch，进行预处理
 * 5. 送入模型推理
 * 6. 后处理，通过 promise 返回结果
 */
struct PendingRequest {
    uint64_t id;                               // 请求 ID
    cv::Mat original_image;                    // 原始图像
    cv::Size original_size;                    // 原始尺寸
    std::string model_name;                    // 模型名称

    // 预处理后的数据（由 BatchScheduler 填充）
    cv::Mat preprocessed_image;                // 预处理后的图像

    // 时间追踪
    std::chrono::steady_clock::time_point arrival_time;  // 到达时间
    std::chrono::steady_clock::time_point batch_time;    // 进入 batch 的时间
    std::chrono::steady_clock::time_point complete_time; // 完成时间

    // 异步返回
    std::promise<InferenceResponse> promise;   // 结果 promise

    PendingRequest()
        : id(0),
          original_size(0, 0),
          arrival_time(std::chrono::steady_clock::now()),
          batch_time(arrival_time),
          complete_time(arrival_time) {}

    PendingRequest(uint64_t request_id, cv::Mat img, const std::string& model)
        : id(request_id),
          original_image(std::move(img)),
          original_size(this->original_image.size()),
          model_name(model),
          arrival_time(std::chrono::steady_clock::now()),
          batch_time(arrival_time),
          complete_time(arrival_time) {}
};

} // namespace inference

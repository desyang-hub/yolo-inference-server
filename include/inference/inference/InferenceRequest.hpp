/**
 * @file InferenceRequest.hpp
 * @brief 推理请求数据结构
 *
 * 学习要点：
 * - 为什么需要独立的 Request 对象？
 *   在高性能服务中，请求数据需要在多个组件之间传递：
 *   HTTP Server → Batch Scheduler → Model → Response Handler
 *   使用统一的数据结构可以避免数据转换的开销。
 *
 * - std::promise 的作用：
 *   将异步结果传递回 HTTP Handler。
 *   Batch Scheduler 处理完后调用 promise.set_value()，
 *   HTTP Handler 通过 future.get() 等待结果。
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference/inference/InferenceResponse.hpp"

namespace inference {

/**
 * @brief 推理请求
 *
 * 从 HTTP Server 接收到的原始请求，经过解析后封装为此结构。
 * 然后提交给 Batch Scheduler 进行批处理。
 */
struct InferenceRequest {
    uint64_t id = 0;                    // 请求唯一 ID（用于追踪）
    cv::Mat image;                      // 输入图像（BGR 格式）
    std::string model_name;             // 目标模型名称
    std::chrono::steady_clock::time_point arrival_time; // 到达时间戳

    /** 异步返回机制 */
    std::promise<InferenceResponse> promise;

    /** 构造函数 */
    InferenceRequest()
        : arrival_time(std::chrono::steady_clock::now()) {}

    InferenceRequest(uint64_t request_id, cv::Mat img, const std::string& model)
        : id(request_id),
          image(std::move(img)),
          model_name(model),
          arrival_time(std::chrono::steady_clock::now()) {}

    /** 计算从到达至今的等待时间 */
    double waiting_time_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - arrival_time).count();
    }
};

} // namespace inference

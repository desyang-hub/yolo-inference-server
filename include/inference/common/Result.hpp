/**
 * @file Result.hpp
 * @brief 统一的结果类型
 *
 * 学习要点：
 * - Result<T> 是 Status 的扩展，用于携带成功时的返回值。
 * - 类似于 Rust 的 Result<T, E> 或 Go 的 (value, error) 模式。
 * - C++23 引入了 std::expected，但 C++17 中我们需要自己实现。
 *
 * 使用示例：
 *   Result<std::vector<Detection>> result = service.Run(image);
 *   if (result.ok()) {
 *       auto& detections = result.value();
 *       // 处理检测结果
 *   } else {
 *       LOG_ERROR << "Inference failed: " << result.status().ToString();
 *   }
 */

#pragma once

#include "Status.hpp"
#include "inference/InferenceResponse.hpp"
#include <utility>
#include <vector>

namespace inference {

/**
 * @brief 结果模板类
 *
 * @tparam T 成功时的返回值类型
 *
 * 设计要点：
 * - 使用 std::move 避免不必要的拷贝
 * - status_ 和 value_ 不会同时有效（类似 union）
 * - ok() 检查后可以通过 value() 获取返回值
 */
template<typename T>
class Result {
public:
    /** 从错误状态创建 */
    static Result FromError(Status status) {
        Result result;
        result.status_ = std::move(status);
        return result;
    }

    /** 从成功值创建 */
    static Result FromValue(T value) {
        Result result;
        result.status_ = Status::Ok();
        result.value_ = std::make_unique<T>(std::move(value));
        return result;
    }

    /** 检查是否成功 */
    bool ok() const { return status_.ok(); }

    /** 获取状态 */
    const Status& status() const { return status_; }

    /** 获取返回值（仅在 ok() == true 时有效） */
    const T& value() const { return *value_; }
    T& value() { return *value_; }

    /** 移动获取返回值 */
    T value() && { return std::move(*value_); }

    /** 隐式转换为 bool */
    explicit operator bool() const { return ok(); }

private:
    Status status_;
    std::unique_ptr<T> value_;
};

// 便捷类型别名
using DetectionResult = Result<std::vector<Detection>>;

} // namespace inference

/**
 * @file Status.hpp
 * @brief 统一的状态码和错误处理
 *
 * 学习要点：
 * - 为什么需要统一的状态码？
 *   因为 C++ 中有多种错误处理方式（异常、错误码、std::expected），
 *   统一使用 Status 可以让错误处理更一致，也更接近现代 C++ 最佳实践。
 *
 * - 与 Python try/except 的对比：
 *   C++ 异常有性能开销（即使不抛出），而 Status 对象是轻量级的，
 *   更适合高性能服务的错误传播。
 */

#pragma once

#include <string>
#include <string_view>

namespace inference {

/**
 * @brief 状态码枚举
 *
 * 类似于 HTTP 状态码，但针对推理服务设计。
 * 每个代码都有一个唯一的整数值，便于日志记录和调试。
 */
enum class StatusCode : int {
    kOk = 0,                // 成功
    kInvalidArgs = 1,       // 参数错误
    kNotFound = 2,          // 资源未找到（如模型文件）
    kAlreadyExists = 3,     // 资源已存在
    kUnavailable = 4,       // 服务不可用
    kDataLoss = 5,          // 数据损坏
    kDeadlineExceeded = 6,  // 超时
    kResourceExhausted = 7, // 资源耗尽（如队列满）
    kInternal = 8,          // 内部错误
};

/**
 * @brief 状态类
 *
 * 封装状态码和错误消息，类似于 gRPC 的 Status 或 C++23 的 std::expected。
 *
 * 使用示例：
 *   Status result = model.Load("model.onnx");
 *   if (!result.ok()) {
 *       LOG_ERROR << "Failed to load model: " << result.ToString();
 *       return;
 *   }
 */
class Status {
public:
    // 工厂方法 - 创建各种状态

    /** 创建成功状态 */
    static Status Ok() { return Status{kStatusCode::kOk, {}}; }

    /** 创建错误状态 */
    static Status InvalidArgument(std::string_view msg) {
        return Status{kStatusCode::kInvalidArgs, std::string{msg}};
    }

    static Status NotFound(std::string_view msg) {
        return Status{kStatusCode::kNotFound, std::string{msg}};
    }

    static Status AlreadyExists(std::string_view msg) {
        return Status{kStatusCode::kAlreadyExists, std::string{msg}};
    }

    static Status Unavailable(std::string_view msg) {
        return Status{kStatusCode::kUnavailable, std::string{msg}};
    }

    static Status DataLoss(std::string_view msg) {
        return Status{kStatusCode::kDataLoss, std::string{msg}};
    }

    static Status DeadlineExceeded(std::string_view msg) {
        return Status{kStatusCode::kDeadlineExceeded, std::string{msg}};
    }

    static Status ResourceExhausted(std::string_view msg) {
        return Status{kStatusCode::kResourceExhausted, std::string{msg}};
    }

    static Status InternalError(std::string_view msg) {
        return Status{kStatusCode::kInternal, std::string{msg}};
    }

    /** 检查是否成功 */
    bool ok() const { return code_ == kStatusCode::kOk; }

    /** 获取状态码 */
    kStatusCode code() const { return code_; }

    /** 获取错误消息 */
    std::string_view message() const { return message_; }

    /** 转换为字符串 */
    std::string ToString() const;

    /** 隐式转换为 bool（方便 if (status) 写法） */
    explicit operator bool() const { return ok(); }

private:
    Status(kStatusCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    kStatusCode code_;
    std::string message_;
};

/**
 * @brief 状态码转字符串
 */
std::string StatusCodeToString(kStatusCode code);

} // namespace inference

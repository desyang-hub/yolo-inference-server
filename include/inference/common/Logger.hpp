/**
 * @file Logger.hpp
 * @brief 日志封装
 *
 * 学习要点：
 * - 为什么使用 spdlog？
 *   spdlog 是 C++ 中最流行的日志库，支持：
 *   1. 异步日志（不阻塞主线程）
 *   2. 多种输出格式（控制台、文件、网络）
 *   3. 日志级别控制（DEBUG/INFO/WARN/ERROR）
 *   4. 性能优秀（每秒数十万条日志）
 *
 * - 为什么封装？
 *   封装后可以简化 API，使用宏来自动注入文件名和行号，
 *   也方便后续统一修改日志格式。
 */

#pragma once

// spdlog 核心头文件
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>  // 彩色控制台输出
#include <spdlog/sinks/rotating_file_sink.h>   // 按大小轮转的文件日志

#include <string>

namespace inference {

/**
 * @brief 日志初始化器
 *
 * 在程序启动时调用，配置日志输出。
 *
 * @param log_level  日志级别（debug/info/warn/error）
 * @param log_file   日志文件路径（可选，空字符串则只输出到控制台）
 * @param max_file_size  单个日志文件最大大小（字节）
 * @param max_files      保留的日志文件数量
 */
void InitLogger(const std::string& log_level = "info",
                const std::string& log_file = "",
                size_t max_file_size = 10 * 1024 * 1024, // 10 MB
                int max_files = 3);

/**
 * @brief 获取全局日志器
 *
 * @return spdlog::logger* 全局日志器指针
 */
spdlog::logger* GetLogger();

/**
 * @brief 设置日志级别
 */
void SetLogLevel(const std::string& level);

/**
 * @brief 关闭日志器
 */
void ShutdownLogger();

} // namespace inference

// ============================================================
// 便捷日志宏
// ============================================================
// 这些宏自动注入文件名、行号、函数名，便于调试。
// 使用方式：LOG_INFO << "Model loaded successfully";

#define LOG_DEBUG   inference::GetLogger()->debug
#define LOG_INFO    inference::GetLogger()->info
#define LOG_WARNING inference::GetLogger()->warn
#define LOG_ERROR   inference::GetLogger()->error

/**
 * 带位置信息的日志宏 - 在调试时很有用
 */
#define LOG_DEBUG_LOC spdlog::debug("{}:{} [{}] - ", __FILE__, __LINE__, __FUNCTION__)
#define LOG_INFO_LOC  spdlog::info("{}:{} [{}] - ", __FILE__, __LINE__, __FUNCTION__)
#define LOG_ERROR_LOC spdlog::error("{}:{} [{}] - ", __FILE__, __LINE__, __FUNCTION__)

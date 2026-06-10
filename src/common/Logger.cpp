/**
 * @file Logger.cpp
 * @brief 日志封装实现
 */

#include "inference/common/Logger.hpp"
#include <spdlog/common.h>

namespace inference {

namespace {
    // 全局日志器指针
    spdlog::logger* g_logger = nullptr;
}

void InitLogger(const std::string& log_level,
                const std::string& log_file,
                size_t max_file_size,
                int max_files) {
    try {
        // 创建控制台输出 sink（带彩色）
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();

        // 如果指定了日志文件，添加文件 sink（按大小轮转）
        std::vector<spdlog::sink_ptr> sinks{console_sink};
        if (!log_file.empty()) {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_st>(
                log_file, max_file_size, max_files);
            sinks.push_back(file_sink);
        }

        // 创建多输出日志器
        g_logger = std::make_shared<spdlog::logger>("inference", sinks.begin(), sinks.end());

        // 设置日志级别
        SetLogLevel(log_level);

        // 设置日志格式：[时间] [级别] [文件:行] 消息
        g_logger->set_pattern("[%H:%M:%S %z] [%^---%L---%$] %v");

        // 设置为默认日志器（spdlog 全局）
        spdlog::set_default(g_logger);

        // 同步刷新（开发时方便调试，生产环境可以改为异步）
        g_logger->flush_on(spdlog::level::info);

        LOG_INFO("Logger initialized. Level: {}, File: {}",
                 log_level, log_file.empty() ? "console only" : log_file);

    } catch (const spdlog::spdlog_ex& ex) {
        // 日志初始化失败时，使用标准错误输出
        fprintf(stderr, "Logger initialization failed: %s\n", ex.what());
        g_logger = spdlog::stderr_color_mt("fallback");
    }
}

spdlog::logger* GetLogger() {
    if (!g_logger) {
        // 如果未初始化，创建一个简单的控制台日志器
        InitLogger("info", "");
    }
    return g_logger;
}

void SetLogLevel(const std::string& level) {
    spdlog::level::level_enum lvl = spdlog::level::info;

    // 将字符串转换为 spdlog 日志级别
    if (level == "debug") {
        lvl = spdlog::level::debug;
    } else if (level == "info") {
        lvl = spdlog::level::info;
    } else if (level == "warn") {
        lvl = spdlog::level::warn;
    } else if (level == "error") {
        lvl = spdlog::level::err;
    } else if (level == "critical") {
        lvl = spdlog::level::critical;
    }

    if (g_logger) {
        g_logger->set_level(lvl);
    }
    spdlog::set_level(lvl);
}

void ShutdownLogger() {
    spdlog::shutdown();
    g_logger = nullptr;
}

} // namespace inference

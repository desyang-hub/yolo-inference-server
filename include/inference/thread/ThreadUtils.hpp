/**
 * @file ThreadUtils.hpp
 * @brief 线程工具
 */

#pragma once

#include <atomic>
#include <thread>

namespace inference {

/**
 * @brief 获取 CPU 逻辑核心数
 */
inline int GetLogicalCores() {
    return std::thread::hardware_concurrency();
}

/**
 * @brief 获取推荐的推理线程数
 *
 * 原则：
 * - 留出核心给 IO 线程
 * - 至少 1 个核心给推理
 */
inline int GetRecommendedInferenceThreads() {
    int cores = GetLogicalCores();
    if (cores <= 2) return 1;
    return cores - 1; // 留一个核心给 IO
}

} // namespace inference

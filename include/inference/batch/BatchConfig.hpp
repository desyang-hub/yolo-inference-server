/**
 * @file BatchConfig.hpp
 * @brief 批处理配置
 *
 * 学习要点：
 * ============================================
 * Dynamic Batching 原理
 * ============================================
 *
 * 问题：单个请求推理时，GPU 利用率低（GPU 在等数据）
 *
 * 解决：将多个请求合并为一个 Batch 进行推理
 * - 8 张图一起推理 vs 1 张图推理 → GPU 吞吐量提升 4-6 倍
 * - 但延迟也会增加（需要等待凑够 Batch）
 *
 * Dynamic Batching 策略：
 * 1. 收集请求到 pending_queue
 * 2. 满足任一条件就触发推理：
 *    a) pending_queue 中的请求数 >= max_batch_size
 *    b) 等待时间 >= timeout_ms（超时触发，避免无限等待）
 * 3. 将收集到的请求预处理为一个大 Tensor
 * 4. 送入 ONNX Runtime 推理
 * 5. 将结果拆分回各个请求
 *
 * ============================================
 * 参数调优指南
 * ============================================
 *
 * 场景 1: 高吞吐（离线批处理）
 *   max_batch_size = 32, timeout = 50ms
 *   → 最大化 GPU 利用率，牺牲一些延迟
 *
 * 场景 2: 低延迟（实时推理）
 *   max_batch_size = 4, timeout = 2ms
 *   → 快速响应，但 GPU 利用率可能不高
 *
 * 场景 3: 平衡
 *   max_batch_size = 8, timeout = 10ms
 *   → 兼顾吞吐和延迟
 */

#pragma once

#include <chrono>
#include <cstddef>

namespace inference {

/**
 * @brief 动态批处理配置
 */
struct BatchConfig {
    /** 最大批大小 */
    size_t max_batch_size = 8;

    /** 等待超时（达到此时间即使未满 batch 也触发推理） */
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);

    /** 最小额批大小（小于此值不触发，除非超时） */
    size_t min_batch_size = 1;

    /** 是否启用动态批处理 */
    bool enabled = true;

    /**
     * @brief 验证配置参数
     */
    bool IsValid() const {
        return max_batch_size >= 1 &&
               min_batch_size >= 1 &&
               min_batch_size <= max_batch_size &&
               timeout.count() > 0;
    }

    /**
     * @brief 创建高吞吐配置
     */
    static BatchConfig HighThroughput() {
        return {32, std::chrono::milliseconds(50), 1, true};
    }

    /**
     * @brief 创建低延迟配置
     */
    static BatchConfig LowLatency() {
        return {4, std::chrono::milliseconds(2), 1, true};
    }

    /**
     * @brief 创建平衡配置
     */
    static BatchConfig Balanced() {
        return {8, std::chrono::milliseconds(10), 1, true};
    }
};

} // namespace inference

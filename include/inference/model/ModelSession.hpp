/**
 * @file ModelSession.hpp
 * @brief ONNX Runtime Session 封装
 *
 * 学习要点：
 * ============================================
 * ONNX Runtime C++ API 核心概念
 * ============================================
 *
 * 1. Ort::Env (环境)
 *    - ONNX Runtime 的全局环境，管理日志和资源
 *    - 整个应用只需要一个 Env
 *    - 线程安全，可以在多个 Session 之间共享
 *
 * 2. Ort::SessionOptions (会话选项)
 *    - 配置推理行为：线程数、执行模式、内存分配器等
 *    - 执行模式: OrtExecutionMode::kParallel（并行）或 kSequential（串行）
 *
 * 3. Ort::Session (会话)
 *    - 加载模型后的推理会话
 *    - 线程安全！可以在多个线程中同时调用 Run()
 *
 * 4. Ort::Value (值)
 *    - ONNX 中的张量，包含数据和形状信息
 *
 * ============================================
 * 线程安全说明
 * ============================================
 * - Ort::Session.Run() 是线程安全的
 * - 多个线程可以同时调用同一个 Session 的 Run()
 * - 但每个线程需要使用独立的输入/输出 Ort::Value
 *
 * ============================================
 * 性能优化
 * ============================================
 * 1. 预热: 首次推理较慢（内存分配），建议启动时 warmup
 * 2. Batch: 增大 batch_size 提升吞吐量
 * 3. GPU: 启用 CUDA 执行提供器
 * 4. 内存预分配: 复用输入输出缓冲区
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference/common/Status.hpp"
#include "inference/model/ModelConfig.hpp"

// ONNX Runtime C++ API
// 注意：ONNX Runtime 有两种 API 风格
// - C API: onnxruntime/core/session/onnxruntime_c_api.h
// - C++ API: onnxruntime_cxx_api.h (推荐)
#ifdef ONNXRUNTIME_FOUND
#include <onnxruntime_cxx_api.h>
#endif

namespace inference {

/**
 * @brief ONNX Runtime Session 封装类
 *
 * 封装 ONNX Runtime 的核心 API，提供：
 * - 模型加载和验证
 * - 输入预处理和输出后处理
 * - 推理执行
 * - 模型信息获取
 */
class ModelSession {
public:
    ModelSession();
    ~ModelSession();

    // 禁止拷贝（Session 包含资源）
    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    // 允许移动
    ModelSession(ModelSession&&) = default;
    ModelSession& operator=(ModelSession&&) = default;

    /**
     * @brief 加载 ONNX 模型
     * @param config  模型配置
     * @return Status 成功或失败
     */
    Status Load(const ModelConfig& config);

#ifdef ONNXRUNTIME_FOUND
    /**
     * @brief 运行推理
     * @param inputs  输入张量数组
     * @param outputs 输出张量数组（由调用者管理内存）
     * @return Status 成功或失败
     *
     * 注意: inputs 和 outputs 的内存由调用者管理。
     *       此函数只负责执行推理。
     */
    Status Run(const std::vector<const char*>& input_names,
               const std::vector<Ort::Value>& inputs,
               const std::vector<const char*>& output_names,
               std::vector<Ort::Value>& outputs);

    /**
     * @brief 预热模型（执行一次空推理）
     * 解决首次推理较慢的问题
     */
    Status Warmup();

    /**
     * @brief 获取 ONNX Session 指针（高级用法）
     */
    Ort::Session* GetSession() { return session_.get(); }
#endif

    /**
     * @brief 获取模型配置
     */
    const ModelConfig& GetConfig() const { return config_; }

    /**
     * @brief 获取输入名称
     */
    const std::vector<std::string>& GetInputNames() const { return input_names_; }

    /**
     * @brief 获取输出名称
     */
    const std::vector<std::string>& GetOutputNames() const { return output_names_; }

    /**
     * @brief 获取输入形状
     */
    const std::vector<int64_t>& GetInputShape() const { return input_shape_; }

    /**
     * @brief 获取输出形状
     */
    const std::vector<int64_t>& GetOutputShape() const { return output_shape_; }

    /**
     * @brief 检查模型是否已加载
     */
    bool IsLoaded() const { return loaded_; }

private:
#ifdef ONNXRUNTIME_FOUND
    // ONNX Runtime 核心对象
    // Ort::Env 使用全局单例（ONNX Runtime 推荐做法），避免每个 Session 创建独立 Env 导致内存泄漏
    static Ort::Env& GetGlobalEnv();
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // 内部方法
    Status ConfigureSessionOptions();
    Status GetModelInfo();
#endif

    // 模型配置和信息
    ModelConfig config_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    bool loaded_ = false;
};

} // namespace inference

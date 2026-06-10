/**
 * @file InferenceService.hpp
 * @brief 推理服务（整合所有组件）
 *
 * 学习要点：
 * - InferenceService 是推理层的"胶水"，将：
 *   ModelManager, BatchScheduler, ImagePreprocessor, NMS
 *   整合在一起，对外提供统一的推理接口。
 */

#pragma once

#include <future>
#include <memory>
#include <string>

#include "inference/batch/BatchConfig.hpp"
#include "inference/batch/PendingRequest.hpp"
#include "inference/common/Status.hpp"
#include "inference/inference/InferenceResponse.hpp"
#include "inference/model/ModelConfig.hpp"
#include "inference/server/ServerConfig.hpp"

namespace inference {

// 前向声明
class ModelManager;
class BatchScheduler;
class ImagePreprocessor;
class NMS;

/**
 * @brief 推理服务
 *
 * 职责：
 * 1. 初始化所有组件
 * 2. 提供推理接口
 * 3. 管理服务状态
 */
class InferenceService {
public:
    explicit InferenceService(const ServerConfig& config);
    ~InferenceService();

    /**
     * @brief 初始化服务（加载模型、启动批调度器）
     */
    Status Initialize();

    /**
     * @brief 提交推理请求（异步）
     */
    std::future<InferenceResponse> SubmitRequest(PendingRequest&& request);

    /**
     * @brief 优雅关闭
     */
    void Shutdown();

    /**
     * @brief 获取批调度器统计
     */
    BatchScheduler::Stats GetBatchStats() const;

    /**
     * @brief 获取待处理请求数
     */
    size_t GetPendingCount() const;

private:
    ServerConfig config_;

    std::shared_ptr<ModelManager> model_manager_;
    std::shared_ptr<BatchScheduler> batch_scheduler_;
    std::shared_ptr<ImagePreprocessor> preprocessor_;
    std::shared_ptr<NMS> nms_;
};

} // namespace inference

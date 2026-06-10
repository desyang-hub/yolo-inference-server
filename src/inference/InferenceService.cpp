/**
 * @file InferenceService.cpp
 * @brief 推理服务实现
 */

#include "inference/inference/InferenceService.hpp"
#include "inference/common/Logger.hpp"
#include "inference/model/ModelManager.hpp"
#include "inference/model/ModelSession.hpp"
#include "inference/batch/BatchScheduler.hpp"
#include "inference/preprocessing/ImagePreprocessor.hpp"
#include "inference/preprocessing/NMS.hpp"

#include <thread>

namespace inference {

InferenceService::InferenceService(const ServerConfig& config)
    : config_(config) {}

InferenceService::~InferenceService() {
    Shutdown();
}

Status InferenceService::Initialize() {
    LOG_INFO("Initializing InferenceService...");

    // 1. 创建模型管理器
    model_manager_ = std::make_shared<ModelManager>();

    // 2. 加载所有配置的模型
    for (const auto& model_config : config_.model_configs) {
        auto status = model_manager_->Load(model_config);
        if (!status.ok()) {
            LOG_ERROR("Failed to load model '{}': {}",
                      model_config.name, status.ToString());
            return status;
        }
    }

    // 3. 创建预处理器（使用第一个模型的输入尺寸）
    if (!config_.model_configs.empty()) {
        const auto& first_model = config_.model_configs[0];
        int input_h = first_model.input.height();
        int input_w = first_model.input.width();

        preprocessor_ = std::make_shared<ImagePreprocessor>(
            input_w, input_h, first_model.preprocess);

        nms_ = std::make_shared<NMS>(first_model.postprocess);
    }

    // 4. 创建批调度器
    batch_scheduler_ = std::make_shared<BatchScheduler>(config_.batch_config);

    // 5. 初始化批调度器（关联模型和处理器）
    if (model_manager_ && !config_.model_configs.empty()) {
        auto* session = model_manager_->GetSession(config_.model_configs[0].name);
        batch_scheduler_->Initialize(session, preprocessor_.get(), nms_.get());
    }

    // 6. 启动批调度器
    batch_scheduler_->Start();

    // 7. 模型预热（仅 ONNX Runtime 可用时）
#ifdef ONNXRUNTIME_FOUND
    for (const auto& model_config : config_.model_configs) {
        auto* session = model_manager_->GetSession(model_config.name);
        if (session) {
            session->Warmup();
        }
    }
#endif

    LOG_INFO("InferenceService initialized successfully");
    LOG_INFO("  Models loaded: {}", model_manager_->ModelCount());
    LOG_INFO("  Batch scheduler: running");
    LOG_INFO("  Preprocessor: {}x{}",
             config_.model_configs.empty() ? 0 : config_.model_configs[0].input.width(),
             config_.model_configs.empty() ? 0 : config_.model_configs[0].input.height());

    return Status::Ok();
}

std::future<InferenceResponse> InferenceService::SubmitRequest(PendingRequest&& request) {
    if (!batch_scheduler_) {
        InferenceResponse response;
        response.request_id = request.id;
        response.status = Status::Unavailable("Service not initialized");

        auto future = std::make_shared<std::promise<InferenceResponse>>();
        future->set_value(std::move(response));
        return future->get_future();
    }

    return batch_scheduler_->Submit(std::move(request));
}

void InferenceService::Shutdown() {
    LOG_INFO("Shutting down InferenceService...");

    if (batch_scheduler_) {
        batch_scheduler_->Stop();
    }

    if (model_manager_) {
        model_manager_->UnloadAll();
    }

    LOG_INFO("InferenceService shut down completed");
}

BatchScheduler::Stats InferenceService::GetBatchStats() const {
    if (batch_scheduler_) {
        return batch_scheduler_->GetStats();
    }
    return {};
}

size_t InferenceService::GetPendingCount() const {
    if (batch_scheduler_) {
        return batch_scheduler_->PendingCount();
    }
    return 0;
}

} // namespace inference

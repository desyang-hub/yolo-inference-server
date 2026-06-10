/**
 * @file ModelManager.cpp
 * @brief 模型管理器实现
 */

#include "inference/model/ModelManager.hpp"
#include "inference/common/Logger.hpp"

#include <algorithm>

namespace inference {

ModelManager::ModelManager() {
    LOG_INFO("ModelManager created");
}

ModelManager::~ModelManager() {
    UnloadAll();
}

Status ModelManager::Load(const ModelConfig& config) {
    // 检查模型是否已加载
    if (models_.find(config.name) != models_.end()) {
        return Status::AlreadyExists("Model '" + config.name + "' is already loaded");
    }

    // 创建并加载模型
    auto session = std::make_shared<ModelSession>();
    auto status = session->Load(config);

    if (!status.ok()) {
        LOG_ERROR("Failed to load model '{}': {}", config.name, status.ToString());
        return status;
    }

    models_[config.name] = session;
    LOG_INFO("Model '{}' loaded successfully. Total models: {}",
             config.name, models_.size());

    return Status::Ok();
}

ModelSession* ModelManager::GetSession(const std::string& name) {
    auto it = models_.find(name);
    if (it == models_.end()) {
        LOG_WARNING("Model '{}' not found", name);
        return nullptr;
    }
    return it->second.get();
}

bool ModelManager::HasModel(const std::string& name) const {
    return models_.find(name) != models_.end();
}

std::vector<std::string> ModelManager::GetModelNames() const {
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto& pair : models_) {
        names.push_back(pair.first);
    }
    return names;
}

void ModelManager::UnloadAll() {
    for (auto& pair : models_) {
        LOG_INFO("Unloading model '{}'", pair.first);
    }
    models_.clear();
    LOG_INFO("All models unloaded");
}

} // namespace inference

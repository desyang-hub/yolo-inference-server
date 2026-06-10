/**
 * @file ModelManager.hpp
 * @brief 模型管理器（支持多模型加载/卸载）
 *
 * 学习要点：
 * ============================================
 * 为什么需要 ModelManager？
 * ============================================
 *
 * 在生产环境中，推理服务通常需要同时支持多个模型：
 * - /detect → YOLOv8（目标检测）
 * - /classify → ResNet50（图像分类）
 * - /segment → SAM（图像分割）
 *
 * ModelManager 负责：
 * 1. 按名称管理多个 ModelSession
 * 2. 模型的延迟加载和热更新
 * 3. 模型生命周期管理
 *
 * ============================================
 * std::shared_ptr vs std::unique_ptr
 * ============================================
 * - ModelSession 使用 shared_ptr，因为：
 *   多个线程可能同时引用同一个模型进行推理
 * - 内部使用 unique_ptr 管理 Ort::Session
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "inference/common/Status.hpp"
#include "inference/model/ModelConfig.hpp"
#include "inference/model/ModelSession.hpp"

namespace inference {

/**
 * @brief 模型管理器
 *
 * 线程安全：Load() 和 GetSession() 可以在不同线程调用，
 * 但 Load() 和 Unload() 不应该在 GetSession() 运行时被调用。
 */
class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    /**
     * @brief 加载模型
     * @param config  模型配置
     * @return Status 成功或失败（如果模型已加载则返回 AlreadyExists）
     */
    Status Load(const ModelConfig& config);

    /**
     * @brief 获取模型会话（用于推理）
     * @param name    模型名称
     * @return ModelSession* 或 nullptr（如果模型不存在）
     */
    ModelSession* GetSession(const std::string& name);

    /**
     * @brief 检查模型是否已加载
     */
    bool HasModel(const std::string& name) const;

    /**
     * @brief 获取所有已加载的模型名称
     */
    std::vector<std::string> GetModelNames() const;

    /**
     * @brief 获取模型数量
     */
    size_t ModelCount() const { return models_.size(); }

    /**
     * @brief 卸载所有模型
     */
    void UnloadAll();

private:
    // 模型存储：名称 -> ModelSession
    std::unordered_map<std::string, std::shared_ptr<ModelSession>> models_;
};

} // namespace inference

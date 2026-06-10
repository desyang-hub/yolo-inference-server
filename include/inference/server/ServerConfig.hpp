/**
 * @file ServerConfig.hpp
 * @brief 服务器配置
 *
 * 学习要点：
 * - 服务器配置的来源可以是：
 *   1. 命令行参数（-p 8080）
 *   2. 配置文件（server_config.json）
 *   3. 环境变量（INFERENCE_PORT=8080）
 *   4. 默认值
 *   优先级：命令行 > 配置文件 > 环境变量 > 默认值
 */

#pragma once

#include <string>

#include "inference/batch/BatchConfig.hpp"
#include "inference/model/ModelConfig.hpp"

namespace inference {

/**
 * @brief 服务器配置
 */
struct ServerConfig {
    // 网络配置
    std::string host = "0.0.0.0";     // 监听地址
    int port = 8080;                   // 监听端口
    int num_threads = 0;               // 工作线程数（0 = 自动 = CPU 核心数）

    // 日志配置
    std::string log_level = "info";
    std::string log_file;

    // 批处理配置
    BatchConfig batch_config;

    // 模型配置
    std::vector<ModelConfig> model_configs;

    /**
     * @brief 验证配置
     */
    bool IsValid() const {
        return port > 0 && port < 65536 &&
               batch_config.IsValid() &&
               !model_configs.empty();
    }

    /**
     * @brief 创建默认配置
     */
    static ServerConfig Default(const std::string& model_path) {
        ServerConfig config;
        config.model_configs.push_back(CreateYOLOv8Config(model_path));
        config.batch_config = BatchConfig::Balanced();
        return config;
    }
};

} // namespace inference

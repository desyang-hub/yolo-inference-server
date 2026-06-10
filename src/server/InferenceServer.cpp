/**
 * @file InferenceServer.cpp
 * @brief 推理服务器实现
 */

#include "inference/server/InferenceServer.hpp"
#include "inference/server/Handlers.hpp"
#include "inference/inference/InferenceService.hpp"
#include "inference/common/Logger.hpp"

#include <functional>

namespace inference {

InferenceServer::InferenceServer(const ServerConfig& config)
    : config_(config) {}

InferenceServer::~InferenceServer() {
    if (running_) {
        Stop();
    }
}

void InferenceServer::Start() {
    // 初始化
    auto status = Initialize();
    if (!status.ok()) {
        LOG_ERROR("Failed to initialize server: {}", status.ToString());
        return;
    }

    // 配置 muduo TcpServer
    muduo::net::InetAddress addr(config_.port);
    tcp_server_ = std::make_unique<muduo::net::TcpServer>(
        loop_.get(), addr, "InferenceServer");

    // 设置 IO 线程数
    int threads = config_.num_threads > 0 ? config_.num_threads : 1;
    tcp_server_->setThreadNum(threads);

    // 设置回调
    tcp_server_->setConnectionCallback(
        [this](const muduo::net::TcpConnectionPtr& conn) {
            handlers_->OnConnection(conn);
        });

    tcp_server_->setMessageCallback(
        [this](const muduo::net::TcpConnectionPtr& conn,
               muduo::net::Buffer* buffer,
               muduo::Timestamp receiveTime) {
            handlers_->OnMessage(conn, buffer, receiveTime);
        });

    // muduo 版本差异：某些版本没有 enableNAIVEHighWaterMark
#if 0
    tcp_server_->enableNAIVEHighWaterMark(1024 * 1024);
#endif

    // 启动
    tcp_server_->start();
    running_ = true;

    LOG_INFO("InferenceServer started on port {}", config_.port);
    LOG_INFO("  IO threads: {}", threads);
    LOG_INFO("  Models: {}", config_.model_configs.size());
    LOG_INFO("  Batch config: max_size={}, timeout={}ms",
             config_.batch_config.max_batch_size,
             config_.batch_config.timeout.count());

    // 启动事件循环
    loop_->loop();

    // 循环退出后
    running_ = false;
    LOG_INFO("InferenceServer stopped");
}

void InferenceServer::Stop() {
    if (!running_) return;

    LOG_INFO("Shutting down InferenceServer...");

    // 停止推理服务
    if (service_) {
        service_->Shutdown();
    }

    // 停止 muduo
    if (loop_) {
        loop_->quit();
    }

    // muduo TcpServer 没有 stop() 方法，直接退出 loop 即可
}

void InferenceServer::Join() {
    // TODO: 等待服务器完全退出
}

Status InferenceServer::Initialize() {
    // 1. 创建 EventLoop
    loop_ = std::make_unique<muduo::net::EventLoop>();

    // 2. 创建推理服务
    service_ = std::make_shared<InferenceService>(config_);
    auto status = service_->Initialize();
    if (!status.ok()) {
        return Status::InternalError(
            std::string("Failed to initialize inference service: ") + status.ToString());
    }

    // 3. 创建请求处理器
    handlers_ = std::make_shared<RequestHandlers>(service_.get());

    return Status::Ok();
}

} // namespace inference

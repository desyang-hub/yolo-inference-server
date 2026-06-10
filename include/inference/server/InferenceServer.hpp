/**
 * @file InferenceServer.hpp
 * @brief 推理服务器主类
 *
 * 学习要点：
 * ============================================
 * 服务器架构
 * ============================================
 *
 * InferenceServer 是整个服务的入口，整合了所有组件：
 *
 *   InferenceServer
 *   ├── muduo::net::TcpServer    (HTTP 网络层)
 *   ├── RequestHandlers          (请求解析和响应)
 *   ├── InferenceService         (推理服务)
 *   │   ├── ModelManager         (多模型管理)
 *   │   ├── BatchScheduler       (动态批处理)
 *   │   ├── ImagePreprocessor    (图像预处理)
 *   │   └── NMS                  (后处理)
 *   └── ServerConfig             (配置)
 *
 * ============================================
 * muduo TcpServer 使用
 * ============================================
 *
 * TcpServer server(&loop, addr, name);
 * server.setThreadNum(num_threads);
 * server.setConnectionCallback(conn_callback);
 * server.setMessageCallback(msg_callback);
 * server.start();
 * loop.loop();
 *
 * - setThreadNum: 设置 IO 线程数（每个线程一个 EventLoop）
 * - ConnectionCallback: 连接建立/关闭时调用
 * - MessageCallback: 有数据可读时调用
 */

#pragma once

#include <memory>
#include <string>

#include "inference/server/ServerConfig.hpp"

// muduo headers
#include <muduo/net/TcpServer.h>
#include <muduo/net/EventLoop.h>

namespace inference {

// 前向声明
class InferenceService;
class RequestHandlers;

/**
 * @brief 推理服务器
 *
 * 启动后监听指定端口，接收 HTTP 请求并进行推理。
 */
class InferenceServer {
public:
    explicit InferenceServer(const ServerConfig& config);
    ~InferenceServer();

    // 禁止拷贝
    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    /**
     * @brief 启动服务器
     */
    void Start();

    /**
     * @brief 停止服务器（优雅关闭）
     */
    void Stop();

    /**
     * @brief 等待服务器退出
     */
    void Join();

    /**
     * @brief 获取服务状态
     */
    bool IsRunning() const { return running_; }

private:
    /**
     * @brief 初始化所有组件
     */
    Status Initialize();

    ServerConfig config_;

    // muduo 网络组件
    std::unique_ptr<muduo::net::EventLoop> loop_;
    std::unique_ptr<muduo::net::TcpServer> tcp_server_;

    // 推理服务
    std::shared_ptr<InferenceService> service_;

    // 请求处理器
    std::shared_ptr<RequestHandlers> handlers_;

    bool running_ = false;
};

} // namespace inference

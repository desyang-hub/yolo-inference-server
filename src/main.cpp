/**
 * @file main.cpp
 * @brief 高性能 ONNX 推理服务 - 程序入口
 *
 * 使用示例:
 *   ./inference_server --model models/yolov8n.onnx --port 8080
 *   ./inference_server --config configs/server_config.json
 */

#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <signal.h>
#include <string>

#include "inference/common/Logger.hpp"
#include "inference/server/InferenceServer.hpp"
#include "inference/server/ServerConfig.hpp"

// 全局服务器指针（用于信号处理）
inference::InferenceServer* g_server = nullptr;

/**
 * @brief 信号处理函数（优雅关闭）
 */
void SignalHandler(int signum) {
    if (g_server) {
        LOG_INFO("Received signal {}, shutting down...", signum);
        g_server->Stop();
    }
}

/**
 * @brief 打印使用帮助
 */
void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  -m, --model PATH       ONNX model path (required)\n"
              << "  -p, --port PORT        Server port (default: 8080)\n"
              << "  -t, --threads NUM      IO threads (default: auto)\n"
              << "  -b, --batch-size NUM   Max batch size (default: 8)\n"
              << "  -T, --batch-timeout MS Batch timeout ms (default: 10)\n"
              << "  -c, --config FILE      Config file path\n"
              << "  -l, --log-level LEVEL  Log level (debug/info/warn/error)\n"
              << "  -h, --help             Show this help\n"
              << "\nExample:\n"
              << "  " << program << " --model models/yolov8n.onnx --port 8080\n";
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    static struct option long_options[] = {
        {"model",       required_argument, 0, 'm'},
        {"port",        required_argument, 0, 'p'},
        {"threads",     required_argument, 0, 't'},
        {"batch-size",  required_argument, 0, 'b'},
        {"batch-timeout", required_argument, 0, 'T'},
        {"config",      required_argument, 0, 'c'},
        {"log-level",   required_argument, 0, 'l'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    std::string model_path;
    int port = 8080;
    int num_threads = 0;
    int batch_size = 8;
    int batch_timeout = 10;
    std::string log_level = "info";

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:t:b:T:c:l:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                model_path = optarg;
                break;
            case 'p':
                port = std::atoi(optarg);
                break;
            case 't':
                num_threads = std::atoi(optarg);
                break;
            case 'b':
                batch_size = std::atoi(optarg);
                break;
            case 'T':
                batch_timeout = std::atoi(optarg);
                break;
            case 'l':
                log_level = optarg;
                break;
            case 'h':
                PrintUsage(argv[0]);
                return 0;
            default:
                PrintUsage(argv[0]);
                return 1;
        }
    }

    // 检查必需参数
    if (model_path.empty()) {
        std::cerr << "Error: Model path is required.\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // 初始化日志
    inference::InitLogger(log_level);
    LOG_INFO("========================================");
    LOG_INFO("  High-Performance ONNX Inference Server");
    LOG_INFO("========================================");

    // 创建服务器配置
    inference::ServerConfig config;
    config.port = port;
    config.num_threads = num_threads;
    config.log_level = log_level;

    // 配置批处理
    config.batch_config.max_batch_size = batch_size;
    config.batch_config.timeout = std::chrono::milliseconds(batch_timeout);

    // 配置模型
    config.model_configs.push_back(
        inference::CreateYOLOv8Config(model_path));

    // 验证配置
    if (!config.IsValid()) {
        LOG_ERROR("Invalid server configuration");
        return 1;
    }

    // 创建并启动服务器
    inference::InferenceServer server(config);
    g_server = &server;

    // 注册信号处理
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);

    // 启动
    server.Start();

    // 清理
    g_server = nullptr;
    inference::ShutdownLogger();

    return 0;
}

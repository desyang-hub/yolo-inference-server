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
#include "inference/model/ModelConfig.hpp"
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
              << "  -g, --gpu PROVIDER     GPU provider: cuda, tensorrt (default: none=CPU)\n"
              << "  -G, --gpu-id ID        GPU device ID (default: 0)\n"
              << "  -C, --conf-threshold F Confidence threshold (default: 0.80)\n"
              << "  -l, --log-level LEVEL  Log level (debug/info/warn/error)\n"
              << "  -h, --help             Show this help\n"
              << "\nExample:\n"
              << "  " << program << " --model models/yolov8n.onnx --port 8080\n";
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    static struct option long_options[] = {
        {"model",           required_argument, 0, 'm'},
        {"port",            required_argument, 0, 'p'},
        {"threads",         required_argument, 0, 't'},
        {"batch-size",      required_argument, 0, 'b'},
        {"batch-timeout",   required_argument, 0, 'T'},
        {"gpu",             required_argument, 0, 'g'},
        {"gpu-id",          required_argument, 0, 'G'},
        {"config",          required_argument, 0, 'c'},
        {"conf-threshold",  required_argument, 0, 'C'},
        {"log-level",       required_argument, 0, 'l'},
        {"help",            no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    std::string model_path;
    int port = 8080;
    int num_threads = 0;
    int batch_size = 8;
    int batch_timeout = 10;
    std::string gpu_provider = "none";
    int gpu_device_id = 0;
    float conf_threshold = 0.80f;
    std::string log_level = "info";

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:t:b:T:g:G:c:C:l:h", long_options, nullptr)) != -1) {
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
            case 'g':
                gpu_provider = optarg;
                break;
            case 'G':
                gpu_device_id = std::atoi(optarg);
                break;
            case 'l':
                log_level = optarg;
                break;
            case 'C':
                conf_threshold = std::atof(optarg);
                break;
            case 'h':
                PrintUsage(argv[0]);
                return 0;
            default:
                PrintUsage(argv[0]);
                return 1;
        }
    }

    // 解析 GPU provider 字符串
    inference::ModelConfig::GpuProvider provider = inference::ModelConfig::GpuProvider::NONE;
    if (gpu_provider == "cuda") {
        provider = inference::ModelConfig::GpuProvider::CUDA;
    } else if (gpu_provider == "tensorrt") {
        provider = inference::ModelConfig::GpuProvider::TENSORRT;
    } else if (gpu_provider != "none" && !gpu_provider.empty()) {
        std::cerr << "Error: Unknown GPU provider '" << gpu_provider
                  << "'. Use: none, cuda, tensorrt\n";
        PrintUsage(argv[0]);
        return 1;
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
    auto model_config = inference::CreateYOLOv8Config(model_path, 640, 80, provider, conf_threshold);
    model_config.gpu_device_id = gpu_device_id;
    config.model_configs.push_back(std::move(model_config));

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

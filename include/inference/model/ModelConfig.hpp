/**
 * @file ModelConfig.hpp
 * @brief 模型配置
 *
 * 学习要点：
 * - 为什么需要 ModelConfig？
 *   不同的模型有不同的输入输出要求：
 *   - YOLOv8: 输入 640x640x3 BGR，输出特定形状的检测张量
 *   - ResNet: 输入 224x224x3 RGB，输出类别概率
 *   - Segment Anything: 输入可变尺寸，输出分割掩码
 *   通过配置描述这些差异，可以实现通用的推理服务。
 *
 * - ONNX 的数据类型：
 *   ONNX Runtime 支持多种数据类型（float32, float16, int8, uint8 等）。
 *   大多数 YOLO 模型使用 float32 输入。
 */

#pragma once

#include <string>
#include <vector>

namespace inference {

/**
 * @brief 模型输入信息
 */
struct ModelInputInfo {
    std::string name;           // 输入名称（ONNX 图中的节点名）
    std::vector<int64_t> shape; // 输入形状 [batch, channels, height, width]
    std::string type;           // 数据类型（如 "tensor(float)"）

    /** 获取输入高度 */
    int height() const { return shape.size() >= 4 ? shape[2] : 0; }

    /** 获取输入宽度 */
    int width() const { return shape.size() >= 4 ? shape[3] : 0; }

    /** 获取通道数 */
    int channels() const { return shape.size() >= 3 ? shape[1] : 0; }
};

/**
 * @brief 模型输出信息
 */
struct ModelOutputInfo {
    std::string name;           // 输出名称
    std::vector<int64_t> shape; // 输出形状
    std::string type;           // 数据类型
};

/**
 * @brief 预处理配置
 */
struct PreprocessConfig {
    std::string format = "BGR";   // 输入图像格式（BGR/GRAY）
    float mean_r = 0.0f;          // R 通道均值（归一化用）
    float mean_g = 0.0f;          // G 通道均值
    float mean_b = 0.0f;          // B 通道均值
    float std_r = 1.0f;           // R 通道标准差
    float std_g = 1.0f;           // G 通道标准差
    float std_b = 1.0f;           // B 通道标准差
    bool scale_to_zero_one = true; // 是否缩放到 0-1 范围
    bool rgb_conversion = false;   // 是否 BGR → RGB
};

/**
 * @brief YOLO 后处理配置
 */
struct YOLOPostprocessConfig {
    int num_classes = 80;              // 类别数
    float conf_threshold = 0.25f;      // 置信度阈值
    float nms_threshold = 0.45f;       // NMS 阈值
    int max_detections = 300;          // 最大检测数
};

/**
 * @brief 模型配置
 *
 * 描述一个 ONNX 模型的所有必要信息，包括：
 * - 模型路径和名称
 * - 输入输出形状和类型
 * - 预处理参数
 * - 后处理参数
 * - ONNX Runtime 会话选项
 */
struct ModelConfig {
    // 基本配置
    std::string name;                      // 模型名称（用于多模型管理）
    std::string model_path;                // ONNX 模型路径
    std::string model_type = "yolo";       // 模型类型（yolo/classification/segmentation）

    // 输入输出配置
    ModelInputInfo input;
    ModelOutputInfo output;

    // 预处理配置
    PreprocessConfig preprocess;

    // 后处理配置
    YOLOPostprocessConfig postprocess;

    // ---------- ONNX Runtime Execution Provider ----------
    /// Execution Provider 类型
    enum class GpuProvider {
        NONE = 0,   // CPU EP (default)
        CUDA = 1,   // CUDA EP
        TENSORRT = 2 // TensorRT EP (requires CUDA)
    };

    int intra_op_num_threads = 0;           // 0 = 自动
    int inter_op_num_threads = 0;           // 0 = 自动
    GpuProvider gpu_provider = GpuProvider::NONE;
    int gpu_device_id = 0;                  // GPU / CUDA device ID

    // CUDA / TensorRT 高级选项
    size_t gpu_mem_limit = SIZE_MAX;        // GPU 显存上限（SIZE_MAX = 不限制）
    int cudnn_conv_algo_search = 0;         // 0=Exhaustive, 1=Default, 2=Hidden
    int cuda_arena_extend_strategy = 0;     // 0=kNextPowerOfTwo, 1=kSameAsRequested
};

/**
 * @brief 创建 YOLOv8 默认配置
 */
inline ModelConfig CreateYOLOv8Config(const std::string& model_path,
                                       int input_size = 640,
                                       int num_classes = 80,
                                       ModelConfig::GpuProvider provider = ModelConfig::GpuProvider::NONE) {
    ModelConfig config;
    config.name = "yolov8";
    config.model_path = model_path;
    config.model_type = "yolo";

    // YOLOv8 输入: [1, 3, 640, 640]
    config.input.name = "images";
    config.input.shape = {1, 3, input_size, input_size};
    config.input.type = "tensor(float)";

    // YOLOv8 输出: [1, 84, 8400] (num_classes + 4) * anchors
    config.output.name = "output0";
    config.output.shape = {1, num_classes + 4, 8400};
    config.output.type = "tensor(float)";

    // 预处理: 缩放到 0-1，BGR → RGB
    config.preprocess.scale_to_zero_one = true;
    config.preprocess.rgb_conversion = true;

    // 后处理
    config.postprocess.num_classes = num_classes;

    // Execution Provider
    config.gpu_provider = provider;

    return config;
}

} // namespace inference

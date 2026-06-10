/**
 * @file InferenceResponse.hpp
 * @brief 推理响应数据结构
 *
 * 学习要点：
 * - 为什么 Detection 使用简单的 float 而不是 cv::Rect？
 *   cv::Rect 是整数类型，不适合存储检测框的精确坐标。
 *   检测框通常使用 float 存储，精度更高。
 *
 * - 为什么使用 std::chrono::duration？
 *   C++11 引入的 chrono 库提供了类型安全的時間计算。
 *   相比直接用 double，chrono 可以自动处理单位转换，
 *   避免毫秒/秒的混淆。
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "inference/common/Status.hpp"

namespace inference {

/**
 * @brief 检测结果
 */
struct Detection {
    float x_center = 0.0f;       // 边界框中心 X（归一化 0-1）
    float y_center = 0.0f;       // 边界框中心 Y（归一化 0-1）
    float width = 0.0f;          // 边界框宽度（归一化 0-1）
    float height = 0.0f;         // 边界框高度（归一化 0-1）
    float confidence = 0.0f;     // 置信度（0-1）
    int class_id = -1;           // 类别 ID
    std::string class_name;      // 类别名称

    /** 转换为左上角 + 右下角格式 */
    struct {
        float x1, y1, x2, y2;
    } ToCornerFormat() const {
        return {
            x_center - width / 2.0f,
            y_center - height / 2.0f,
            x_center + width / 2.0f,
            y_center + height / 2.0f
        };
    }
};

/**
 * @brief 推理响应
 *
 * 从 Model Session 返回的结果，经过后处理后封装为此结构。
 * 然后序列化为 JSON 返回给客户端。
 */
struct InferenceResponse {
    uint64_t request_id = 0;                  // 对应请求 ID
    std::vector<Detection> detections;        // 检测结果
    double inference_time_ms = 0.0;           // 推理耗时（毫秒）
    double preprocessing_time_ms = 0.0;       // 预处理耗时
    double postprocessing_time_ms = 0.0;      // 后处理耗时
    Status status = Status::Ok();             // 状态

    /** 是否有检测结果 */
    bool has_detections() const {
        return !detections.empty();
    }

    /** 获取置信度最高的检测结果 */
    const Detection& best_detection() const {
        return detections.front(); // 假设已按置信度排序
    }
};

/**
 * @brief COCO 数据集的 80 个类别名称
 * YOLO 模型通常使用 COCO 数据集训练
 */
inline const std::vector<std::string> CLASS_NAMES = {
    "person",        "bicycle",      "car",           "motorcycle",
    "airplane",      "bus",          "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",        "bird",          "cat",
    "dog",           "horse",        "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",
    "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",      "frisbee",      "skis",          "snowboard",
    "sports ball",   "kite",         "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",  "bottle",
    "wine glass",    "cup",          "fork",          "knife",
    "spoon",         "bowl",         "banana",        "apple",
    "sandwich",      "orange",       "broccoli",      "carrot",
    "hot dog",       "pizza",        "donut",         "cake",
    "chair",         "couch",        "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",
    "mouse",         "remote",       "keyboard",      "cell phone",
    "microwave",     "oven",         "toaster",       "sink",
    "refrigerator",  "book",         "clock",         "vase",
    "scissors",      "teddy bear",   "hair drier",    "toothbrush"
};

} // namespace inference

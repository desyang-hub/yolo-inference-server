/**
 * @file ImagePreprocessor.hpp
 * @brief 图像预处理器
 *
 * 学习要点：
 * ============================================
 * 为什么需要预处理？
 * ============================================
 *
 * 原始图像和 ONNX 模型的输入格式不同：
 *
 * 原始图像: H×W×3 (HWC), BGR, uint8 (0-255)
 * 模型输入: 3×H×W (CHW), RGB, float32 (0-1)
 *
 * 预处理步骤：
 * 1. Resize: 将图像缩放到模型输入尺寸（如 640×640）
 *    - 保持宽高比（letterbox）或直接缩放
 * 2. 色彩空间转换: BGR → RGB
 * 3. 归一化: uint8 → float32, 缩放到 0-1
 * 4. 格式转换: HWC → CHW（可选，取决于模型）
 *
 * ============================================
 * Letterbox 缩放
 * ============================================
 *
 * 保持宽高比的缩放方式，在短边填充灰色：
 *
 *   原始 1920×1080 → 目标 640×640
 *   缩放比例 = min(640/1920, 640/1080) = 0.356
 *   缩放后尺寸 = 683×384
 *   填充后 = 640×640（上下各填充 128 像素）
 *
 * 这对 YOLO 检测很重要，因为变形会影响检测精度。
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "inference/common/Status.hpp"
#include "inference/model/ModelConfig.hpp"

namespace inference {

/**
 * @brief 图像预处理器
 *
 * 将原始图像转换为 ONNX 模型可接受的输入格式。
 * 线程安全：每个线程使用自己的工作缓冲区。
 */
class ImagePreprocessor {
public:
    /**
     * @brief 构造函数
     * @param target_width  目标宽度
     * @param target_height 目标高度
     * @param config  预处理配置
     */
    ImagePreprocessor(int target_width,
                       int target_height,
                       const PreprocessConfig& config = PreprocessConfig{});

    /**
     * @brief 预处理图像
     * @param input   输入图像（BGR, uint8）
     * @param output  输出张量（RGB, float32, CHW）
     * @return Status 成功或失败
     */
    Status Preprocess(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 预处理图像（返回 cv::Mat，直接可用于 ONNX）
     */
    cv::Mat Preprocess(const cv::Mat& input);

    /**
     * @brief 获取缩放比例（用于后处理时的坐标还原）
     */
    double GetScaleFactor(const cv::Size& original_size) const;

    /**
     * @brief 获取填充偏移（Letterbox 的偏移量）
     */
    cv::Point GetPaddingOffset(const cv::Size& original_size) const;

    /**
     * @brief 获取目标尺寸
     */
    cv::Size GetTargetSize() const { return target_size_; }

private:
    /**
     * @brief Letterbox 缩放（保持宽高比）
     */
    cv::Mat Letterbox(const cv::Mat& input);

    int target_width_;
    int target_height_;
    cv::Size target_size_;
    PreprocessConfig config_;
};

} // namespace inference

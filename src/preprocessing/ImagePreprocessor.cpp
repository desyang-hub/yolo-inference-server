/**
 * @file ImagePreprocessor.cpp
 * @brief 图像预处理器实现
 */

#include "inference/preprocessing/ImagePreprocessor.hpp"
#include "inference/common/Logger.hpp"

namespace inference {

ImagePreprocessor::ImagePreprocessor(int target_width,
                                      int target_height,
                                      const PreprocessConfig& config)
    : target_width_(target_width),
      target_height_(target_height),
      target_size_(target_width, target_height),
      config_(config) {
    LOG_INFO("ImagePreprocessor created. Target size: {}x{}",
             target_width, target_height);
}

Status ImagePreprocessor::Preprocess(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) {
        return Status::InvalidArgument("Input image is empty");
    }

    if (input.channels() != 3) {
        return Status::InvalidArgument(
            std::string("Expected 3 channels, got ") + std::to_string(input.channels()));
    }

    // 步骤 1: Letterbox 缩放
    auto scaled = Letterbox(input);

    // 步骤 2: 色彩空间转换（BGR → RGB）
    if (config_.rgb_conversion) {
        cv::cvtColor(scaled, scaled, cv::COLOR_BGR2RGB);
    }

    // 步骤 3: 转换为 float32 并归一化
    if (config_.scale_to_zero_one) {
        scaled.convertTo(output, CV_32F, 1.0 / 255.0);
    } else {
        scaled.convertTo(output, CV_32F);
    }

    // 步骤 4: 均值和标准差归一化（如果配置了）
    if (config_.mean_r != 0.0f || config_.mean_g != 0.0f ||
        config_.mean_b != 0.0f ||
        config_.std_r != 1.0f || config_.std_g != 1.0f ||
        config_.std_b != 1.0f) {
        cv::Mat channels[3];
        cv::split(output, channels);

        // R channel
        channels[0] = (channels[0] - config_.mean_r) / config_.std_r;
        // G channel
        channels[1] = (channels[1] - config_.mean_g) / config_.std_g;
        // B channel
        channels[2] = (channels[2] - config_.mean_b) / config_.std_b;

        cv::merge(channels, 3, output);
    }

    return Status::Ok();
}

cv::Mat ImagePreprocessor::Preprocess(const cv::Mat& input) {
    cv::Mat output;
    auto status = Preprocess(input, output);
    if (!status.ok()) {
        LOG_ERROR("Preprocessing failed: {}", status.ToString());
        return cv::Mat();
    }
    return output;
}

double ImagePreprocessor::GetScaleFactor(const cv::Size& original_size) const {
    double scale = std::min(
        static_cast<double>(target_width_) / original_size.width,
        static_cast<double>(target_height_) / original_size.height);
    return scale;
}

cv::Point ImagePreprocessor::GetPaddingOffset(const cv::Size& original_size) const {
    double scale = GetScaleFactor(original_size);
    int new_width = static_cast<int>(original_size.width * scale);
    int new_height = static_cast<int>(original_size.height * scale);

    int top = (target_height_ - new_height) / 2;
    int left = (target_width_ - new_width) / 2;

    return cv::Point(left, top);
}

cv::Mat ImagePreprocessor::Letterbox(const cv::Mat& input) {
    // 计算缩放比例
    double scale = std::min(
        static_cast<double>(target_width_) / input.cols,
        static_cast<double>(target_height_) / input.rows);

    // 计算缩放后的尺寸
    int new_width = static_cast<int>(input.cols * scale);
    int new_height = static_cast<int>(input.rows * scale);

    // 缩放图像
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height),
               0, 0, cv::INTER_LINEAR);

    // 创建目标图像（填充灰色）
    cv::Mat output(target_height_, target_width_, input.type(),
                   cv::Scalar(114, 114, 114)); // YOLO 使用灰色填充

    // 计算放置位置
    int top = (target_height_ - new_height) / 2;
    int left = (target_width_ - new_width) / 2;

    // 将缩放后的图像复制到目标图像
    resized.copyTo(output(cv::Rect(left, top, new_width, new_height)));

    return output;
}

} // namespace inference

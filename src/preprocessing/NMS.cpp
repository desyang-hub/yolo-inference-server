/**
 * @file NMS.cpp
 * @brief 非极大值抑制实现
 */

#include "inference/preprocessing/NMS.hpp"
#include "inference/common/Logger.hpp"
#include "inference/inference/InferenceResponse.hpp"

#include <algorithm>
#include <cmath>

namespace inference {

NMS::NMS(const YOLOPostprocessConfig& config)
    : config_(config) {
    LOG_INFO("NMS created. conf_thresh={}, nms_thresh={}, max_det={}",
             config_.conf_threshold, config_.nms_threshold, config_.max_detections);
}

std::vector<Detection> NMS::Process(const std::vector<Detection>& raw_detections) {
    std::vector<Detection> results;
    results.reserve(config_.max_detections);

    // 步骤 1: 过滤低置信度的检测
    std::vector<Detection> filtered;
    filtered.reserve(raw_detections.size());

    for (const auto& det : raw_detections) {
        if (det.confidence >= config_.conf_threshold) {
            // 设置类别名称
            Detection d = det;
            if (d.class_id >= 0 && d.class_id < static_cast<int>(CLASS_NAMES.size())) {
                d.class_name = CLASS_NAMES[d.class_id];
            }
            filtered.push_back(d);
        }
    }

    // 步骤 2: 按置信度排序（降序）
    std::sort(filtered.begin(), filtered.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // 步骤 3: 对每个类别分别进行 NMS
    std::vector<bool> suppressed(filtered.size(), false);

    for (size_t i = 0; i < filtered.size(); ++i) {
        if (suppressed[i]) continue;

        // 获取当前检测的类别
        int class_id = filtered[i].class_id;

        // 与后续同类的检测比较
        for (size_t j = i + 1; j < filtered.size(); ++j) {
            if (suppressed[j]) continue;
            if (filtered[j].class_id != class_id) continue;

            // 计算 IoU
            float iou = CalculateIoU(filtered[i], filtered[j]);

            // 如果 IoU 超过阈值，抑制检测 j
            if (iou > config_.nms_threshold) {
                suppressed[j] = true;
            }
        }

        // 当前检测未被抑制，加入结果
        results.push_back(filtered[i]);

        // 达到最大检测数，提前退出
        if (results.size() >= config_.max_detections) {
            break;
        }
    }

    return results;
}

std::vector<Detection> NMS::ProcessByClass(const std::vector<Detection>& detections,
                                            int class_id) {
    // 筛选出指定类别的检测
    std::vector<Detection> class_detections;
    for (const auto& det : detections) {
        if (det.class_id == class_id) {
            class_detections.push_back(det);
        }
    }

    // 按置信度排序
    std::sort(class_detections.begin(), class_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // NMS
    std::vector<Detection> results;
    std::vector<bool> suppressed(class_detections.size(), false);

    for (size_t i = 0; i < class_detections.size(); ++i) {
        if (suppressed[i]) continue;

        results.push_back(class_detections[i]);

        for (size_t j = i + 1; j < class_detections.size(); ++j) {
            if (suppressed[j]) continue;

            float iou = CalculateIoU(class_detections[i], class_detections[j]);
            if (iou > config_.nms_threshold) {
                suppressed[j] = true;
            }
        }

        if (results.size() >= config_.max_detections) {
            break;
        }
    }

    return results;
}

float NMS::CalculateIoU(const Detection& a, const Detection& b) {
    // 转换为左上角 + 右下角格式
    auto a_corner = a.ToCornerFormat();
    auto b_corner = b.ToCornerFormat();

    // 计算交集矩形的坐标
    float inter_x1 = std::max(a_corner.x1, b_corner.x1);
    float inter_y1 = std::max(a_corner.y1, b_corner.y1);
    float inter_x2 = std::min(a_corner.x2, b_corner.x2);
    float inter_y2 = std::min(a_corner.y2, b_corner.y2);

    // 交集面积
    float inter_width = std::max(0.0f, inter_x2 - inter_x1);
    float inter_height = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;

    // 如果无交集，IoU = 0
    if (inter_area <= 0.0f) {
        return 0.0f;
    }

    // 计算各自面积
    float a_area = (a_corner.x2 - a_corner.x1) * (a_corner.y2 - a_corner.y1);
    float b_area = (b_corner.x2 - b_corner.x1) * (b_corner.y2 - b_corner.y1);

    // 并集面积
    float union_area = a_area + b_area - inter_area;

    // IoU
    return inter_area / union_area;
}

} // namespace inference

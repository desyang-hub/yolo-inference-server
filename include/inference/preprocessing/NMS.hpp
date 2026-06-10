/**
 * @file NMS.hpp
 * @brief 非极大值抑制（Non-Maximum Suppression）
 *
 * 学习要点：
 * ============================================
 * NMS 是什么？
 * ============================================
 *
 * YOLO 模型对每个锚框都预测一个边界框和置信度。
 * 同一个物体可能被多个锚框检测到，产生大量重叠的检测结果。
 * NMS 的作用是：保留置信度最高的检测，抑制其他重叠的检测。
 *
 * ============================================
 * NMS 算法步骤
 * ============================================
 *
 * 输入: 一堆边界框和置信度
 * 输出: 筛选后的边界框
 *
 * 1. 按置信度从高到低排序
 * 2. 选取置信度最高的框 A，加入结果
 * 3. 计算 A 与其余框的 IoU（Intersection over Union）
 * 4. 移除 IoU > threshold 的框（这些被 A 抑制）
 * 5. 重复 2-4，直到所有框处理完毕
 *
 * ============================================
 * IoU (Intersection over Union)
 * ============================================
 *
 *        A ┌──────┐
 *          │  A∩B  │
 *   B  ┌───┤       ├────┐
 *      │   └──────┘    │
 *      └───────────────┘
 *
 *   IoU = Area(A ∩ B) / Area(A ∪ B)
 *       = Intersection / Union
 *
 *   IoU = 1.0 → 完全重叠
 *   IoU = 0.0 → 完全不重叠
 */

#pragma once

#include <vector>

#include "inference/inference/InferenceResponse.hpp"
#include "inference/model/ModelConfig.hpp"

namespace inference {

/**
 * @brief 非极大值抑制（NMS）
 *
 * 对 YOLO 模型的原始输出进行后处理，去除重叠的检测结果。
 * 线程安全：Process() 可以在多个线程同时调用。
 */
class NMS {
public:
    /**
     * @brief 构造函数
     * @param config  后处理配置
     */
    explicit NMS(const YOLOPostprocessConfig& config = YOLOPostprocessConfig{});

    /**
     * @brief 处理 YOLO 原始输出
     * @param raw_detections  模型原始输出中的检测框
     * @return 经过 NMS 筛选后的检测结果
     *
     * YOLOv8 输出格式: [1, 84, 8400]
     * - 前 4 个通道: 边界框坐标 (x, y, w, h)
     * - 后 80 个通道: 各类别的置信度
     * - 8400: 锚框数量 (80×80×4 + 40×40×4 + 20×20×4)
     */
    std::vector<Detection> Process(const std::vector<Detection>& raw_detections);

    /**
     * @brief 处理单个类别的检测框
     */
    std::vector<Detection> ProcessByClass(
        const std::vector<Detection>& detections, int class_id);

private:
    /**
     * @brief 计算两个边界框的 IoU
     * @param a  边界框 A
     * @param b  边界框 B
     * @return IoU 值 (0-1)
     */
    float CalculateIoU(const Detection& a, const Detection& b);

    YOLOPostprocessConfig config_;
};

} // namespace inference

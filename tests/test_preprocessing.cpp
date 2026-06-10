/**
 * @file test_preprocessing.cpp
 * @brief 预处理和后处理单元测试
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "inference/preprocessing/ImagePreprocessor.hpp"
#include "inference/preprocessing/NMS.hpp"
#include "inference/common/Status.hpp"
#include "inference/inference/InferenceResponse.hpp"

using namespace inference;

// ============================================================
// ImagePreprocessor 测试
// ============================================================

class ImagePreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建预处理器：640x640
        preprocessor = std::make_unique<ImagePreprocessor>(640, 640);
    }

    std::unique_ptr<ImagePreprocessor> preprocessor;
};

TEST_F(ImagePreprocessorTest, BasicPreprocess) {
    // 创建一个测试图像（1920x1080 BGR）
    cv::Mat test_image(1080, 1920, CV_8UC3, cv::Scalar(128, 64, 192));

    // 预处理
    cv::Mat result;
    Status status = preprocessor->Preprocess(test_image, result);

    // 验证
    ASSERT_TRUE(status.ok());
    ASSERT_FALSE(result.empty());
    ASSERT_EQ(result.size(), cv::Size(640, 640));
    ASSERT_EQ(result.type(), CV_32F);
}

TEST_F(ImagePreprocessorTest, LetterboxPreservesAspect) {
    // 创建非正方形图像
    cv::Mat test_image(400, 800, CV_8UC3, cv::Scalar(255, 0, 0));

    cv::Mat result;
    preprocessor->Preprocess(test_image, result);

    // 验证尺寸
    ASSERT_EQ(result.rows, 640);
    ASSERT_EQ(result.cols, 640);

    // 验证填充（灰色边框应该存在）
    double scale = preprocessor->GetScaleFactor(test_image.size());
    EXPECT_NEAR(scale, 0.8, 0.01); // 640/800 = 0.8
}

TEST_F(ImagePreprocessorTest, ScaleFactor) {
    cv::Size original(1920, 1080);
    double scale = preprocessor->GetScaleFactor(original);

    // scale = min(640/1920, 640/1080) = min(0.333, 0.593) = 0.333
    EXPECT_NEAR(scale, 0.333, 0.01);
}

TEST_F(ImagePreprocessorTest, EmptyImage) {
    cv::Mat empty_image;
    cv::Mat result;
    Status status = preprocessor->Preprocess(empty_image, result);

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), Status::kStatusCode::kInvalidArgs);
}

// ============================================================
// NMS 测试
// ============================================================

class NMSTest : public ::testing::Test {
protected:
    void SetUp() override {
        nms_config.conf_threshold = 0.3f;
        nms_config.nms_threshold = 0.45f;
        nms_config.max_detections = 100;
        nms = std::make_unique<NMS>(nms_config);
    }

    YOLOPostprocessConfig nms_config;
    std::unique_ptr<NMS> nms;
};

TEST_F(NMSTest, BasicNMS) {
    // 创建两个重叠的检测
    std::vector<Detection> detections = {
        {0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0, "class0"},  // 高置信度
        {0.51f, 0.51f, 0.2f, 0.2f, 0.8f, 0, "class0"}, // 重叠，较低置信度
    };

    auto results = nms->Process(detections);

    // 应该只保留第一个
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0].confidence, 0.9f, 0.01);
}

TEST_F(NMSTest, NonOverlapping) {
    // 创建两个不重叠的检测
    std::vector<Detection> detections = {
        {0.2f, 0.2f, 0.1f, 0.1f, 0.9f, 0, "class0"},
        {0.8f, 0.8f, 0.1f, 0.1f, 0.8f, 0, "class0"},
    };

    auto results = nms->Process(detections);

    // 两个都应该保留
    ASSERT_EQ(results.size(), 2);
}

TEST_F(NMSTest, DifferentClasses) {
    // 相同位置但不同类别
    std::vector<Detection> detections = {
        {0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0, "class0"},
        {0.5f, 0.5f, 0.2f, 0.2f, 0.8f, 1, "class1"},
    };

    auto results = nms->Process(detections);

    // 不同类别不应该互相抑制
    ASSERT_EQ(results.size(), 2);
}

TEST_F(NMSTest, ConfidenceThreshold) {
    std::vector<Detection> detections = {
        {0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0, "class0"},  // 高于阈值
        {0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0, "class0"},  // 低于阈值
    };

    auto results = nms->Process(detections);

    // 只有第一个应该通过
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0].confidence, 0.5f, 0.01);
}

TEST_F(NMSTest, MaxDetections) {
    // 创建超过 max_detections 的检测
    std::vector<Detection> detections;
    for (int i = 0; i < 150; ++i) {
        // 分散在不同位置，不会互相抑制
        detections.push_back({
            static_cast<float>(i % 10) * 0.1f,
            static_cast<float>(i / 10) * 0.1f,
            0.05f, 0.05f,
            0.9f, 0, "class0"
        });
    }

    nms_config.max_detections = 100;
    nms = std::make_unique<NMS>(nms_config);

    auto results = nms->Process(detections);

    ASSERT_EQ(results.size(), 100u);
}

// ============================================================
// 运行测试
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/**
 * @file test_batch_scheduler.cpp
 * @brief 批调度器单元测试
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "inference/batch/BatchScheduler.hpp"
#include "inference/batch/BatchConfig.hpp"
#include "inference/batch/PendingRequest.hpp"
#include "inference/common/Status.hpp"

using namespace inference;

// ============================================================
// BatchConfig 测试
// ============================================================

TEST(BatchConfigTest, DefaultConfig) {
    BatchConfig config;
    ASSERT_TRUE(config.IsValid());
    EXPECT_EQ(config.max_batch_size, 8u);
    EXPECT_EQ(config.min_batch_size, 1u);
}

TEST(BatchConfigTest, HighThroughput) {
    auto config = BatchConfig::HighThroughput();
    ASSERT_TRUE(config.IsValid());
    EXPECT_EQ(config.max_batch_size, 32u);
}

TEST(BatchConfigTest, LowLatency) {
    auto config = BatchConfig::LowLatency();
    ASSERT_TRUE(config.IsValid());
    EXPECT_EQ(config.max_batch_size, 4u);
}

TEST(BatchConfigTest, InvalidConfig) {
    BatchConfig config;
    config.max_batch_size = 0;
    ASSERT_FALSE(config.IsValid());

    config.max_batch_size = 1;
    config.min_batch_size = 2; // min > max is invalid
    ASSERT_FALSE(config.IsValid());
}

// ============================================================
// PendingRequest 测试
// ============================================================

TEST(PendingRequestTest, Creation) {
    cv::Mat test_image(480, 640, CV_8UC3, cv::Scalar(128));
    PendingRequest request(1, test_image, "yolov8");

    EXPECT_EQ(request.id, 1u);
    EXPECT_EQ(request.model_name, "yolov8");
    EXPECT_EQ(request.original_size, cv::Size(640, 480));
}

TEST(PendingRequestTest, FuturePromise) {
    cv::Mat test_image(480, 640, CV_8UC3, cv::Scalar(128));
    PendingRequest request(1, test_image, "yolov8");

    // 获取 future
    auto future = request.promise.get_future();

    // 设置值
    InferenceResponse response;
    response.request_id = 1;
    response.status = Status::Ok();
    request.promise.set_value(std::move(response));

    // 通过 future 获取
    auto result = future.get();
    EXPECT_EQ(result.request_id, 1u);
    EXPECT_TRUE(result.status.ok());
}

// ============================================================
// BatchScheduler 测试（需要 mock 模型）
// ============================================================

TEST(BatchSchedulerTest, Creation) {
    BatchConfig config;
    config.max_batch_size = 4;
    config.timeout = std::chrono::milliseconds(100); // 较长超时用于测试

    BatchScheduler scheduler(config);
    EXPECT_EQ(scheduler.PendingCount(), 0u);
}

TEST(BatchSchedulerTest, Stats) {
    BatchScheduler scheduler;
    auto stats = scheduler.GetStats();

    EXPECT_EQ(stats.total_requests, 0u);
    EXPECT_EQ(stats.total_batches, 0u);
    EXPECT_NEAR(stats.avg_batch_size, 0.0, 0.01);
}

// ============================================================
// 运行测试
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

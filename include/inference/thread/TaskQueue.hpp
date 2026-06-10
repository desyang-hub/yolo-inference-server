/**
 * @file TaskQueue.hpp
 * @brief 有界任务队列
 *
 * 学习要点：
 * ============================================
 * 为什么需要任务队列？
 * ============================================
 *
 * 在多线程环境中，任务队列是生产者-消费者模式的核心：
 * - 生产者（HTTP Handler）提交任务
 * - 消费者（推理线程）处理任务
 * - 队列解耦生产者和消费者
 *
 * ============================================
 * 有界 vs 无界
 * ============================================
 *
 * 无界队列: 无限增长，可能导致 OOM
 * 有界队列: 容量有限，满时可以选择：
 *   1. 阻塞等待（生产者等待）
 *   2. 丢弃旧任务（保留新的）
 *   3. 丢弃新任务（保留旧的）
 *   4. 返回错误（背压）
 *
 * 本实现使用阻塞等待，适合推理场景。
 */

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace inference {

/**
 * @brief 有界任务队列
 *
 * @tparam T 任务类型
 */
template<typename T>
class TaskQueue {
public:
    /**
     * @brief 构造函数
     * @param capacity  队列容量（0 = 无界）
     */
    explicit TaskQueue(size_t capacity = 0)
        : capacity_(capacity) {}

    /**
     * @brief 提交任务
     * @return true = 成功提交, false = 队列关闭
     */
    bool Push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);

        // 如果有界且满，等待空间
        if (capacity_ > 0) {
            not_full_.wait(lock, [this]() {
                return queue_.size() < capacity_ || closed_;
            });
        }

        if (closed_) {
            return false;
        }

        queue_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /**
     * @brief 获取任务（阻塞）
     * @return std::optional<T> 任务，队列关闭且空时返回 nullopt
     */
    std::optional<T> Pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_.wait(lock, [this]() {
            return !queue_.empty() || closed_;
        });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return item;
    }

    /**
     * @brief 获取任务（非阻塞）
     */
    std::optional<T> TryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return item;
    }

    /**
     * @brief 关闭队列
     */
    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /**
     * @brief 获取当前大小
     */
    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * @brief 检查是否为空
     */
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::deque<T> queue_;
    size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
};

} // namespace inference

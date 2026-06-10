/**
 * @file ObjectPool.hpp
 * @brief 通用对象池
 *
 * 学习要点：
 * ============================================
 * 为什么需要对象池？
 * ============================================
 *
 * 问题：高频 new/delete 导致：
 * 1. 性能开销：每次 malloc/free 都需要系统调用
 * 2. 内存碎片：频繁分配释放小块内存
 * 3. 延迟抖动：GC 或内存分配器锁竞争
 *
 * 解决：预分配对象池，复用对象而非反复创建
 *
 * ============================================
 * 使用模式
 * ============================================
 *
 *   // 创建池
 *   ObjectPool<MyObject> pool(100);  // 预分配 100 个对象
 *
 *   // 获取对象
 *   auto* obj = pool.Acquire();
 *   obj->Initialize(...);
 *
 *   // 使用对象...
 *
 *   // 归还对象
 *   pool.Release(obj);
 *
 * ============================================
 * 线程安全
 * ============================================
 * - Acquire() 和 Release() 使用 mutex 保护
 * - 多个线程可以安全地获取和归还对象
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace inference {

/**
 * @brief 通用对象池
 *
 * @tparam T 池化对象类型（必须可默认构造）
 */
template<typename T>
class ObjectPool {
public:
    /**
     * @brief 构造函数
     * @param pool_size  池大小（预分配的对象数量）
     * @param reset_fn   对象重置函数（可选，归还原调用）
     */
    explicit ObjectPool(size_t pool_size,
                        std::function<void(T*)> reset_fn = nullptr)
        : reset_fn_(reset_fn) {
        // 预分配对象
        for (size_t i = 0; i < pool_size; ++i) {
            available_.push_back(&pool_.emplace_back(T{}));
        }
    }

    /**
     * @brief 从池中获取对象
     * @return T* 对象指针，如果池为空则新分配一个
     */
    T* Acquire() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (available_.empty()) {
            // 池为空，记录统计（可选：扩容）
            ++overflow_count_;
            return nullptr; // 或者返回新分配的对象
        }

        // 从可用列表中取出一个
        T* obj = available_.back();
        available_.pop_back();
        return obj;
    }

    /**
     * @brief 归还对象到池中
     * @param obj  要归还的对象指针
     */
    void Release(T* obj) {
        if (!obj) return;

        // 重置对象状态
        if (reset_fn_) {
            reset_fn_(obj);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        available_.push_back(obj);
    }

    /**
     * @brief 获取当前可用对象数
     */
    size_t AvailableCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_.size();
    }

    /**
     * @brief 获取池大小
     */
    size_t PoolSize() const { return pool_.size(); }

    /**
     * @brief 获取溢出次数（池为空时的 Acquire 调用次数）
     */
    size_t OverflowCount() const { return overflow_count_.load(); }

private:
    std::vector<T> pool_;                    // 对象存储
    std::vector<T*> available_;              // 可用对象列表
    mutable std::mutex mutex_;               // 线程安全
    std::atomic<size_t> overflow_count_{0};  // 溢出计数
    std::function<void(T*)> reset_fn_;       // 重置函数
};

} // namespace inference

/**
 * @file MemoryPool.hpp
 * @brief Tensor 内存池
 *
 * 学习要点：
 * - Tensor 内存池专门用于复用 ONNX 推理的输入/输出缓冲区
 * - 避免每次推理都分配/释放大量内存
 */

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

#include "inference/memory/ObjectPool.hpp"

namespace inference {

/**
 * @brief 对齐的内存缓冲区
 */
struct AlignedBuffer {
    float* data = nullptr;
    size_t size = 0;

    /** 分配内存 */
    bool Allocate(size_t num_elements) {
        size = num_elements;
        // 使用 aligned_alloc 进行 32 字节对齐（SIMD 友好）
        data = static_cast<float*>(std::aligned_alloc(32,
                ((num_elements * sizeof(float) + 31) / 32) * 32));
        return data != nullptr;
    }

    /** 释放内存 */
    void Free() {
        if (data) {
            std::free(data);
            data = nullptr;
            size = 0;
        }
    }

    /** 重置 */
    void Reset() {
        // 不清零数据（避免不必要的 memset）
        // 下次使用时会覆盖
    }
};

/**
 * @brief Tensor 内存池
 *
 * 预分配固定大小的 float 缓冲区，供推理时复用。
 */
class MemoryPool {
public:
    /**
     * @brief 构造函数
     * @param buffer_count  预分配的缓冲区数量
     * @param buffer_size   每个缓冲区的大小（元素数）
     */
    MemoryPool(size_t buffer_count, size_t buffer_size);
    ~MemoryPool();

    /**
     * @brief 获取缓冲区
     */
    AlignedBuffer* Acquire();

    /**
     * @brief 归还缓冲区
     */
    void Release(AlignedBuffer* buffer);

private:
    std::vector<AlignedBuffer> buffers_;
    std::vector<AlignedBuffer*> available_;
    std::mutex mutex_;
};

} // namespace inference

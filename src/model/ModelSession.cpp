/**
 * @file ModelSession.cpp
 * @brief ONNX Runtime Session 封装实现
 *
 * 这是整个推理服务的核心文件之一，展示了如何使用 ONNX Runtime C++ API。
 */

#include "inference/model/ModelSession.hpp"
#include "inference/common/Logger.hpp"

#include <system_error>

#ifdef ONNXRUNTIME_FOUND
#include <cpu_provider_factory.h> // OrtSessionOptionsAppendExecutionProvider_CPU
#endif

namespace inference {

ModelSession::ModelSession()
    : loaded_(false)
{
    LOG_INFO("ModelSession created");
#ifdef ONNXRUNTIME_FOUND
    LOG_INFO("  ONNX Runtime C++ API available");
#else
    LOG_WARNING("  ONNX Runtime NOT found - running in framework mode");
#endif
}

ModelSession::~ModelSession() = default;

Status ModelSession::Load(const ModelConfig& config) {
    LOG_INFO("Loading model: {} from {}", config.name, config.model_path);

    config_ = config;
    loaded_ = false;

#ifdef ONNXRUNTIME_FOUND
    // 步骤 1: 配置 SessionOptions
    auto status = ConfigureSessionOptions();
    if (!status.ok()) {
        LOG_ERROR("Failed to configure session options: {}", status.ToString());
        return status;
    }

    // 步骤 2: 创建 Session（加载模型）
    try {
        session_ = std::make_unique<Ort::Session>(env_, config.model_path.c_str(),
                                                    session_options_);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to create ONNX Session: {}", e.what());
        return Status::NotFound(std::string("Failed to load model: ") + e.what());
    }

    // 步骤 3: 获取模型信息
    status = GetModelInfo();
    if (!status.ok()) {
        LOG_ERROR("Failed to get model info: {}", status.ToString());
        return status;
    }

    loaded_ = true;
    LOG_INFO("Model loaded successfully: {} -> {} (inputs: {}, outputs: {})",
             config.name, config.model_path, input_names_.size(), output_names_.size());
#else
    // ONNX Runtime 未安装时的模拟实现
    LOG_WARNING("ONNX Runtime not available. Simulating model load.");

    // 模拟从配置中获取输入输出信息
    input_names_.push_back(config.input.name);
    output_names_.push_back(config.output.name);
    input_shape_ = config.input.shape;
    output_shape_ = config.output.shape;

    loaded_ = true;
    LOG_INFO("Model simulated (no ONNX Runtime): inputs={}, outputs={}",
             input_names_.size(), output_names_.size());
#endif

    return Status::Ok();
}

#ifdef ONNXRUNTIME_FOUND

Status ModelSession::ConfigureSessionOptions() {
    // 设置执行模式为并行 (可以利用多核)
    session_options_.SetExecutionMode(ORT_PARALLEL);

    // 启用图优化 (优化计算图执行顺序)
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // 设置线程数（CPU 模式下生效）
    if (config_.intra_op_num_threads > 0) {
        session_options_.SetIntraOpNumThreads(config_.intra_op_num_threads);
        LOG_INFO("  Intra-op threads: {}", config_.intra_op_num_threads);
    }

    if (config_.inter_op_num_threads > 0) {
        session_options_.SetInterOpNumThreads(config_.inter_op_num_threads);
        LOG_INFO("  Inter-op threads: {}", config_.inter_op_num_threads);
    }

    // -------------------------------------------------------
    // Execution Provider 选择
    // CUDA / TensorRT 优先执行 GPU 上的算子，
    // CPU EP 作为 fallback 处理 GPU 不支持的算子。
    // -------------------------------------------------------
    switch (config_.gpu_provider) {
        case ModelConfig::GpuProvider::CUDA: {
            LOG_INFO("  Execution Provider: CUDA (device_id={})", config_.gpu_device_id);
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config_.gpu_device_id;
            cuda_options.gpu_mem_limit = config_.gpu_mem_limit;
            cuda_options.cudnn_conv_algo_search =
                static_cast<OrtCudnnConvAlgoSearch>(config_.cudnn_conv_algo_search);
            cuda_options.arena_extend_strategy = config_.cuda_arena_extend_strategy;
            cuda_options.do_copy_in_default_stream = 1;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            break;
        }

        case ModelConfig::GpuProvider::TENSORRT: {
            LOG_INFO("  Execution Provider: TensorRT (device_id={})", config_.gpu_device_id);
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = config_.gpu_device_id;
            trt_options.has_user_compute_stream = 0;
            trt_options.user_compute_stream = nullptr;
            trt_options.trt_max_partition_iterations = 10;
            trt_options.trt_min_subgraph_size = 1;
            trt_options.trt_max_workspace_size = static_cast<size_t>(config_.gpu_mem_limit);
            trt_options.trt_fp16_enable = 1;
            trt_options.trt_int8_enable = 0;
            trt_options.trt_int8_calibration_table_name = nullptr;
            trt_options.trt_int8_use_native_calibration_table = 0;
            trt_options.trt_dla_enable = 0;
            trt_options.trt_dla_core = 0;
            trt_options.trt_dump_subgraphs = 0;
            trt_options.trt_engine_cache_enable = 0;
            trt_options.trt_engine_cache_path = nullptr;
            trt_options.trt_engine_decryption_enable = 0;
            trt_options.trt_engine_decryption_lib_path = nullptr;
            trt_options.trt_force_sequential_engine_build = 0;
            session_options_.AppendExecutionProvider_TensorRT(trt_options);
            break;
        }

        case ModelConfig::GpuProvider::NONE:
        default:
            LOG_INFO("  Execution Provider: CPU");
            break;
    }

    // CPU EP 始终追加，作为 fallback 处理 GPU 不支持的算子
    // Ort::SessionOptions 没有 C++ 封装的 CPU provider，直接调用 C API
    // Ort::SessionOptions 有隐式转换 operator OrtSessionOptions*
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CPU(
            static_cast<OrtSessionOptions*>(session_options_), 1 /* use_arena */));

    return Status::Ok();
}

Status ModelSession::GetModelInfo() {
    try {
        // 获取输入信息
        size_t num_inputs = session_->GetInputCount();
        input_names_.resize(num_inputs);
        input_shape_.clear();

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < num_inputs; ++i) {
            // ONNX Runtime 1.18: 使用 GetInputNameAllocated 获取输入名称
            // 参数：index 和 allocator 指针 (nullptr 表示使用默认分配器)
            input_names_[i] = session_->GetInputNameAllocated(i, allocator).get();

            // 获取输入形状
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();
        }

        // 获取输出信息
        size_t num_outputs = session_->GetOutputCount();
        output_names_.resize(num_outputs);
        output_shape_.clear();

        for (size_t i = 0; i < num_outputs; ++i) {
            // ONNX Runtime 1.18: 使用 GetOutputNameAllocated 获取输出名称
            // 参数：index 和 allocator 指针 (nullptr 表示使用默认分配器)
            output_names_[i] = session_->GetOutputNameAllocated(i, allocator).get();

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = tensor_info.GetShape();
        }

        LOG_INFO("  Inputs: {}, Outputs: {}", input_names_.size(), output_names_.size());
        return Status::Ok();

    } catch (const Ort::Exception& e) {
        return Status::InternalError(
            std::string("Failed to get model info: ") + e.what());
    }
}

Status ModelSession::Run(const std::vector<const char*>& input_names,
                         const std::vector<Ort::Value>& inputs,
                         const std::vector<const char*>& output_names,
                         std::vector<Ort::Value>& outputs) {
    if (!session_ || !loaded_) {
        return Status::Unavailable("Model not loaded");
    }

    try {
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), inputs.data(), inputs.size(),
                                 output_names.data(), output_names.size());

        LOG_INFO("output size: {}", outputs.size());
        return Status::Ok();

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Inference failed: {}", e.what());
        return Status::InternalError(std::string("Inference error: ") + e.what());
    } catch (...) {
        // 4. 捕获所有未知异常
        LOG_ERROR("Unknown fatal exception during inference!");
        return Status::InternalError("Unknown fatal exception");
    }
}

Status ModelSession::Warmup() {
    if (!session_ || !loaded_) {
        return Status::Unavailable("Model not loaded for warmup");
    }

    LOG_INFO("Warming up model...");

    // ONNX Runtime 1.18: 使用 CreateCpu 创建内存信息对象
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,  // OrtAllocatorType::OrtArenaAllocator
        OrtMemTypeDefault); // OrtMemType::OrtMemTypeDefault

    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }

        std::vector<float> zero_data(num_elements, 0.0f);
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, zero_data.data(), num_elements,
            shape.data(), shape.size()));
    }

    std::vector<Ort::Value> outputs;

    // 修复：准备输入输出名称数组 (转换为 const char* 格式)
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : input_names_) {
        input_name_ptrs.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
        output_name_ptrs.push_back(name.c_str());
    }

    auto status = Run(input_name_ptrs, inputs, output_name_ptrs, outputs);

    if (status.ok()) {
        LOG_INFO("Model warmup completed");
    } else {
        LOG_WARNING("Model warmup failed: {}", status.ToString());
    }

    return status;
}

#endif // ONNXRUNTIME_FOUND

} // namespace inference

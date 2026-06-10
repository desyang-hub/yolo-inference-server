/**
 * @file ModelSession.cpp
 * @brief ONNX Runtime Session 封装实现
 *
 * 这是整个推理服务的核心文件之一，展示了如何使用 ONNX Runtime C++ API。
 */

#include "inference/model/ModelSession.hpp"
#include "inference/common/Logger.hpp"

#include <system_error>

namespace inference {

ModelSession::ModelSession()
    : env_(ORT_LOGGING_LEVEL_WARNING, "InferenceService") {
    LOG_INFO("ModelSession created with ONNX Runtime Env");
}

ModelSession::~ModelSession() = default;

Status ModelSession::Load(const ModelConfig& config) {
    LOG_INFO("Loading model: {} from {}", config.name, config.model_path);

    config_ = config;

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

    LOG_INFO("Model loaded successfully: {} -> {} (inputs: {}, outputs: {})",
             config.name, config.model_path, input_names_.size(), output_names_.size());

    // 打印输入输出信息
    for (const auto& name : input_names_) {
        LOG_INFO("  Input: {}", name);
    }
    for (const auto& name : output_names_) {
        LOG_INFO("  Output: {}", name);
    }

    return Status::Ok();
}

Status ModelSession::ConfigureSessionOptions() {
    // 设置执行模式
    if (config_.intra_op_num_threads > 1) {
        session_options_.SetIntraOpNumThreads(config_.intra_op_num_threads);
        LOG_INFO("  Intra-op threads: {}", config_.intra_op_num_threads);
    }

    if (config_.inter_op_num_threads > 1) {
        session_options_.SetInterOpNumThreads(config_.inter_op_num_threads);
        LOG_INFO("  Inter-op threads: {}", config_.inter_op_num_threads);
    }

    // 设置执行模式为并行（可以利用多核）
    session_options_.SetExecutionMode(Ort::ExecutionMode::kParallel);

    // 启用内存增量分配（默认关闭，按需开启）
    // session_options_.EnableMemoryPattern(true);

    // 启用时序优化（优化计算图执行顺序）
    session_options_.SetOptimizationLevel(ORT_ENABLE_ALL);

    // GPU 支持（如果配置了 GPU 且 ONNX Runtime 支持 CUDA）
    if (config_.enable_gpu) {
        try {
            // CUDA EP 需要编译时启用
            // OrtCUDAProviderOptions cuda_options;
            // session_options_.EnableCudaProvider(config_.gpu_device_id);
            LOG_WARNING("GPU enabled in config, but CUDA EP not linked. Using CPU.");
        } catch (const Ort::Exception& e) {
            LOG_WARNING("CUDA provider not available: {}. Using CPU.", e.what());
        }
    }

    return Status::Ok();
}

Status ModelSession::GetModelInfo() {
    try {
        // 获取输入信息
        size_t num_inputs = session_->GetInputCount();
        input_names_.resize(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            input_names_[i] = session_->GetInputNameAllocated(i, Ort::MemoryInfo::DefaultCpu()).get();

            // 获取输入形状
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();

            LOG_INFO("  Input[{}] '{}' shape: [{}]",
                     i, input_names_[i],
                     fmt::join(input_shape_.begin(), input_shape_.end(), ", "));
        }

        // 获取输出信息
        size_t num_outputs = session_->GetOutputCount();
        output_names_.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            output_names_[i] = session_->GetOutputNameAllocated(i, Ort::MemoryInfo::DefaultCpu()).get();

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = tensor_info.GetShape();

            LOG_INFO("  Output[{}] '{}' shape: [{}]",
                     i, output_names_[i],
                     fmt::join(output_shape_.begin(), output_shape_.end(), ", "));
        }

        return Status::Ok();

    } catch (const Ort::Exception& e) {
        return Status::InternalError(std::string("Failed to get model info: ") + e.what());
    }
}

Status ModelSession::Run(const std::vector<const char*>& input_names,
                         const std::vector<Ort::Value>& inputs,
                         const std::vector<const char*>& output_names,
                         std::vector<Ort::Value>& outputs) {
    if (!session_) {
        return Status::Unavailable("Model not loaded");
    }

    try {
        // 运行推理
        // Run() 是线程安全的，多个线程可以同时调用
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), inputs.data(), inputs.size(),
                                 output_names.data(), output_names.size());

        return Status::Ok();

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Inference failed: {}", e.what());
        return Status::InternalError(std::string("Inference error: ") + e.what());
    }
}

Status ModelSession::Warmup() {
    if (!session_) {
        return Status::Unavailable("Model not loaded for warmup");
    }

    LOG_INFO("Warming up model...");

    // 创建零输入（全 0 张量）
    auto memory_info = Ort::MemoryInfo::DefaultCpu();

    // 根据输入形状创建零张量
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        // 获取输入类型信息
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        auto element_type = tensor_info.GetElementType();

        // 计算元素总数
        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }

        // 创建零数据
        std::vector<float> zero_data(num_elements, 0.0f);

        // 创建张量
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, zero_data.data(), num_elements, shape.data(), shape.size()));
    }

    // 运行一次推理
    std::vector<Ort::Value> outputs;
    auto status = Run(
        const_cast<std::vector<const char*>>(input_names_).data(), inputs,
        const_cast<std::vector<const char*>>(output_names_).data(), outputs);

    if (status.ok()) {
        LOG_INFO("Model warmup completed");
    } else {
        LOG_WARNING("Model warmup failed: {}", status.ToString());
    }

    return status;
}

} // namespace inference

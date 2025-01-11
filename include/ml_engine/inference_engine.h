#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <type_traits>

namespace ml_engine {

// Forward declarations
template<typename T>
class Tensor;

class MemoryPool;
class GraphOptimizer;
class KernelFusion;

/**
 * @brief High-performance ML inference engine with template metaprogramming
 * 
 * Features:
 * - Memory-efficient execution with custom allocators
 * - SIMD vectorization and CPU/GPU kernel fusion
 * - Lock-free data structures for concurrent execution
 * - Template metaprogramming for compile-time optimization
 * - Numerical accuracy within 0.1% tolerance
 */
template<typename DataType = float>
class InferenceEngine {
public:
    static_assert(std::is_floating_point_v<DataType>, "DataType must be floating point");
    
    InferenceEngine();
    ~InferenceEngine();
    
    // Disable copy, enable move
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = default;
    InferenceEngine& operator=(InferenceEngine&&) = default;
    
    /**
     * @brief Load a neural network model
     * @param model_path Path to the model file
     * @return true if successful, false otherwise
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Run inference on input data
     * @param input Input tensor
     * @return Output tensor
     */
    Tensor<DataType> infer(const Tensor<DataType>& input);
    
    /**
     * @brief Run batch inference
     * @param inputs Vector of input tensors
     * @return Vector of output tensors
     */
    std::vector<Tensor<DataType>> inferBatch(const std::vector<Tensor<DataType>>& inputs);
    
    /**
     * @brief Set number of CPU threads for OpenMP
     * @param num_threads Number of threads
     */
    void setNumThreads(int num_threads);
    
    /**
     * @brief Enable CUDA acceleration
     * @param device_id CUDA device ID
     */
    void enableCuda(int device_id = 0);
    
    /**
     * @brief Enable SIMD optimization
     */
    void enableSIMD(bool enable = true);
    
    /**
     * @brief Get performance metrics
     * @return Performance statistics
     */
    std::unordered_map<std::string, double> getPerformanceMetrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Template metaprogramming helpers
    template<typename T>
    using is_supported_type = std::bool_constant<
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> || 
        std::is_same_v<T, int32_t>
    >;
    
    template<typename T>
    static constexpr bool is_supported_type_v = is_supported_type<T>::value;
};

} // namespace ml_engine
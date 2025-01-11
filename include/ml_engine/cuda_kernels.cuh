#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>

namespace ml_engine {
namespace cuda {

/**
 * @brief CUDA kernel wrapper for tensor operations
 */
template<typename T>
class CudaKernels {
public:
    CudaKernels(int device_id = 0);
    ~CudaKernels();
    
    // Memory management
    T* allocateGPU(size_t size);
    void deallocateGPU(T* ptr);
    void copyToGPU(const T* host_data, T* gpu_data, size_t size);
    void copyToCPU(const T* gpu_data, T* host_data, size_t size);
    
    // Element-wise operations
    void add(const T* a, const T* b, T* result, size_t size);
    void multiply(const T* a, const T* b, T* result, size_t size);
    void scale(const T* a, T scalar, T* result, size_t size);
    
    // Matrix operations
    void matmul(const T* A, const T* B, T* C, size_t M, size_t N, size_t K);
    void transpose(const T* input, T* output, size_t rows, size_t cols);
    
    // Activation functions
    void relu(const T* input, T* output, size_t size);
    void sigmoid(const T* input, T* output, size_t size);
    void tanh_activation(const T* input, T* output, size_t size);
    void softmax(const T* input, T* output, size_t size);
    
    // Convolution operations
    void conv2d(const T* input, const T* kernel, T* output,
                size_t batch_size, size_t in_channels, size_t out_channels,
                size_t input_height, size_t input_width,
                size_t kernel_height, size_t kernel_width,
                size_t stride_h, size_t stride_w,
                size_t pad_h, size_t pad_w);
    
    // Pooling operations
    void max_pool2d(const T* input, T* output,
                    size_t batch_size, size_t channels,
                    size_t input_height, size_t input_width,
                    size_t pool_height, size_t pool_width,
                    size_t stride_h, size_t stride_w);
    
    void avg_pool2d(const T* input, T* output,
                    size_t batch_size, size_t channels,
                    size_t input_height, size_t input_width,
                    size_t pool_height, size_t pool_width,
                    size_t stride_h, size_t stride_w);
    
    // Batch normalization
    void batch_norm(const T* input, const T* scale, const T* bias,
                   const T* mean, const T* variance, T* output,
                   size_t batch_size, size_t channels,
                   size_t height, size_t width, T epsilon = 1e-5);
    
    // Error checking
    void checkCudaError(cudaError_t error, const char* message);
    void synchronize();

private:
    int device_id_;
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
    cudaStream_t stream_;
    
    void initializeCublas();
    void initializeCudnn();
    void cleanup();
};

/**
 * @brief Kernel fusion utilities for optimizing computation graphs
 */
class KernelFusion {
public:
    /**
     * @brief Fused convolution + batch norm + ReLU operation
     */
    template<typename T>
    static void conv_bn_relu(const T* input, const T* conv_weight, const T* conv_bias,
                            const T* bn_scale, const T* bn_bias, const T* bn_mean, const T* bn_var,
                            T* output, size_t batch_size, size_t in_channels, size_t out_channels,
                            size_t input_height, size_t input_width,
                            size_t kernel_height, size_t kernel_width,
                            size_t stride_h, size_t stride_w,
                            size_t pad_h, size_t pad_w, T epsilon = 1e-5);
    
    /**
     * @brief Fused matrix multiply + bias + activation
     */
    template<typename T>
    static void gemm_bias_activation(const T* A, const T* B, const T* bias, T* C,
                                   size_t M, size_t N, size_t K,
                                   const char* activation = "relu");
    
    /**
     * @brief Fused attention mechanism (Q*K^T, softmax, *V)
     */
    template<typename T>
    static void fused_attention(const T* Q, const T* K, const T* V, T* output,
                              size_t batch_size, size_t seq_len, size_t head_dim,
                              size_t num_heads, T scale_factor = 1.0);
};

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

} // namespace cuda
} // namespace ml_engine
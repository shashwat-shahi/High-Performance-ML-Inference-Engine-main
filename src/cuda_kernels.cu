#include "ml_engine/cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>

namespace ml_engine {
namespace cuda {

// CUDA kernel implementations
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void multiply_kernel(const T* a, const T* b, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__global__ void scale_kernel(const T* a, T scalar, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

template<typename T>
__global__ void relu_kernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > T(0) ? input[idx] : T(0);
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = T(1) / (T(1) + expf(-input[idx]));
    }
}

template<typename T>
__global__ void tanh_kernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

template<typename T>
__global__ void softmax_kernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        // Find maximum value
        T max_val = input[0];
        for (size_t i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, input[i]);
        }
        
        // Compute exponentials and sum
        T sum = T(0);
        for (size_t i = 0; i < size; ++i) {
            output[i] = expf(input[i] - max_val);
            sum += output[i];
        }
        
        // Normalize
        for (size_t i = 0; i < size; ++i) {
            output[i] /= sum;
        }
    }
}

template<typename T>
__global__ void transpose_kernel(const T* input, T* output, size_t rows, size_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// Fused convolution + batch norm + ReLU kernel
template<typename T>
__global__ void conv_bn_relu_kernel(
    const T* input, const T* conv_weight, const T* conv_bias,
    const T* bn_scale, const T* bn_bias, const T* bn_mean, const T* bn_var,
    T* output, size_t batch_size, size_t in_channels, size_t out_channels,
    size_t input_height, size_t input_width,
    size_t kernel_height, size_t kernel_width,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w, T epsilon) {
    
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int output_y = blockIdx.z / ((input_width + 2 * pad_w - kernel_width) / stride_w + 1);
    int output_x = blockIdx.z % ((input_width + 2 * pad_w - kernel_width) / stride_w + 1);
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels) return;
    
    T conv_result = 0;
    
    // Convolution
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int input_y = output_y * stride_h + ky - pad_h;
                int input_x = output_x * stride_w + kx - pad_w;
                
                if (input_y >= 0 && input_y < input_height && 
                    input_x >= 0 && input_x < input_width) {
                    
                    int input_idx = batch_idx * in_channels * input_height * input_width +
                                   in_ch * input_height * input_width +
                                   input_y * input_width + input_x;
                    
                    int weight_idx = out_ch_idx * in_channels * kernel_height * kernel_width +
                                    in_ch * kernel_height * kernel_width +
                                    ky * kernel_width + kx;
                    
                    conv_result += input[input_idx] * conv_weight[weight_idx];
                }
            }
        }
    }
    
    conv_result += conv_bias[out_ch_idx];
    
    // Batch normalization
    T normalized = (conv_result - bn_mean[out_ch_idx]) / 
                   sqrtf(bn_var[out_ch_idx] + epsilon);
    T bn_result = normalized * bn_scale[out_ch_idx] + bn_bias[out_ch_idx];
    
    // ReLU activation
    T final_result = bn_result > T(0) ? bn_result : T(0);
    
    int output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
    int output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;
    int output_idx = batch_idx * out_channels * output_height * output_width +
                     out_ch_idx * output_height * output_width +
                     output_y * output_width + output_x;
    
    output[output_idx] = final_result;
}

// CudaKernels implementation
template<typename T>
CudaKernels<T>::CudaKernels(int device_id) : device_id_(device_id) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    initializeCublas();
    initializeCudnn();
}

template<typename T>
CudaKernels<T>::~CudaKernels() {
    cleanup();
}

template<typename T>
T* CudaKernels<T>::allocateGPU(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return ptr;
}

template<typename T>
void CudaKernels<T>::deallocateGPU(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template<typename T>
void CudaKernels<T>::copyToGPU(const T* host_data, T* gpu_data, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(gpu_data, host_data, size * sizeof(T), 
                              cudaMemcpyHostToDevice, stream_));
}

template<typename T>
void CudaKernels<T>::copyToCPU(const T* gpu_data, T* host_data, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(host_data, gpu_data, size * sizeof(T), 
                              cudaMemcpyDeviceToHost, stream_));
}

template<typename T>
void CudaKernels<T>::add(const T* a, const T* b, T* result, size_t size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    add_kernel<<<grid, block, 0, stream_>>>(a, b, result, size);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void CudaKernels<T>::multiply(const T* a, const T* b, T* result, size_t size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    multiply_kernel<<<grid, block, 0, stream_>>>(a, b, result, size);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void CudaKernels<T>::scale(const T* a, T scalar, T* result, size_t size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    scale_kernel<<<grid, block, 0, stream_>>>(a, scalar, result, size);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void CudaKernels<T>::matmul(const T* A, const T* B, T* C, size_t M, size_t N, size_t K) {
    const T alpha = 1.0f, beta = 0.0f;
    
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha, B, N, A, K, &beta, C, N));
    } else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha, B, N, A, K, &beta, C, N));
    }
}

template<typename T>
void CudaKernels<T>::relu(const T* input, T* output, size_t size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    relu_kernel<<<grid, block, 0, stream_>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void CudaKernels<T>::sigmoid(const T* input, T* output, size_t size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    sigmoid_kernel<<<grid, block, 0, stream_>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void CudaKernels<T>::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template<typename T>
void CudaKernels<T>::initializeCublas() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
}

template<typename T>
void CudaKernels<T>::initializeCudnn() {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle_, stream_));
}

template<typename T>
void CudaKernels<T>::cleanup() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

// KernelFusion implementations
template<typename T>
void KernelFusion::conv_bn_relu(const T* input, const T* conv_weight, const T* conv_bias,
                               const T* bn_scale, const T* bn_bias, const T* bn_mean, const T* bn_var,
                               T* output, size_t batch_size, size_t in_channels, size_t out_channels,
                               size_t input_height, size_t input_width,
                               size_t kernel_height, size_t kernel_width,
                               size_t stride_h, size_t stride_w,
                               size_t pad_h, size_t pad_w, T epsilon) {
    
    size_t output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
    size_t output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;
    
    dim3 grid(batch_size, out_channels, output_height * output_width);
    dim3 block(1);
    
    conv_bn_relu_kernel<<<grid, block>>>(
        input, conv_weight, conv_bias, bn_scale, bn_bias, bn_mean, bn_var,
        output, batch_size, in_channels, out_channels,
        input_height, input_width, kernel_height, kernel_width,
        stride_h, stride_w, pad_h, pad_w, epsilon
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations
template class CudaKernels<float>;
template class CudaKernels<double>;

} // namespace cuda
} // namespace ml_engine
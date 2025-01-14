#include "ml_engine/tensor.h"
#include "ml_engine/memory.h"
#include "ml_engine/simd_ops.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace ml_engine {

// Global memory pool for tensor allocations
static thread_local std::unique_ptr<MemoryPool> tensor_pool = nullptr;

static MemoryPool& getTensorPool() {
    if (!tensor_pool) {
        tensor_pool = std::make_unique<MemoryPool>(512 * 1024 * 1024); // 512MB per thread
    }
    return *tensor_pool;
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape) : shape_(shape), is_on_gpu_(false), gpu_data_(nullptr) {
    size_ = computeSize(shape);
    allocate(size_);
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, const std::vector<T>& data) 
    : shape_(shape), is_on_gpu_(false), gpu_data_(nullptr) {
    size_ = computeSize(shape);
    if (data.size() != size_) {
        throw std::invalid_argument("Data size does not match tensor shape");
    }
    allocate(size_);
    std::memcpy(data_, data.data(), size_ * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(std::initializer_list<T> data) 
    : shape_{data.size()}, size_(data.size()), is_on_gpu_(false), gpu_data_(nullptr) {
    allocate(size_);
    std::copy(data.begin(), data.end(), data_);
}

template<typename T>
Tensor<T>::~Tensor() {
    deallocate();
}

template<typename T>
Tensor<T>::Tensor(const Tensor& other) 
    : shape_(other.shape_), size_(other.size_), is_on_gpu_(false), gpu_data_(nullptr) {
    allocate(size_);
    if (other.is_on_gpu_) {
        // Copy from GPU to CPU
        #ifdef CUDA_ENABLED
        cudaMemcpy(data_, other.gpu_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        #endif
    } else {
        std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape_ = other.shape_;
        size_ = other.size_;
        is_on_gpu_ = false;
        gpu_data_ = nullptr;
        allocate(size_);
        
        if (other.is_on_gpu_) {
            #ifdef CUDA_ENABLED
            cudaMemcpy(data_, other.gpu_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
            #endif
        } else {
            std::memcpy(data_, other.data_, size_ * sizeof(T));
        }
    }
    return *this;
}

template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept 
    : data_(other.data_), shape_(std::move(other.shape_)), 
      size_(other.size_), capacity_(other.capacity_),
      is_on_gpu_(other.is_on_gpu_), gpu_data_(other.gpu_data_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.is_on_gpu_ = false;
    other.gpu_data_ = nullptr;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        capacity_ = other.capacity_;
        is_on_gpu_ = other.is_on_gpu_;
        gpu_data_ = other.gpu_data_;
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        other.is_on_gpu_ = false;
        other.gpu_data_ = nullptr;
    }
    return *this;
}

template<typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimensions do not match tensor dimensions");
    }
    return data_[computeIndex(indices)];
}

template<typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimensions do not match tensor dimensions");
    }
    return data_[computeIndex(indices)];
}

template<typename T>
void Tensor<T>::reshape(const Shape& new_shape) {
    size_t new_size = computeSize(new_shape);
    if (new_size != size_) {
        throw std::invalid_argument("New shape must preserve total number of elements");
    }
    shape_ = new_shape;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    Tensor result(shape_);
    simd_add(data_, other.data_, result.data_, size_);
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    
    Tensor result(shape_);
    simd_mul(data_, other.data_, result.data_, size_);
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T scalar) const {
    Tensor result(shape_);
    simd_scale(data_, scalar, result.data_, size_);
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    
    size_t M = shape_[0];
    size_t N = other.shape_[1];
    size_t K = shape_[1];
    
    Tensor result({M, N});
    
    // Use SIMD-optimized matrix multiplication
    simd::simd_matrix_multiply(data_, other.data_, result.data_, M, N, K);
    
    return result;
}



template<typename T>
void Tensor<T>::toGPU(int device_id) {
    #ifdef CUDA_ENABLED
    if (!is_on_gpu_) {
        cudaSetDevice(device_id);
        cudaMalloc(&gpu_data_, size_ * sizeof(T));
        cudaMemcpy(gpu_data_, data_, size_ * sizeof(T), cudaMemcpyHostToDevice);
        is_on_gpu_ = true;
    }
    #else
    throw std::runtime_error("CUDA support not compiled");
    #endif
}

template<typename T>
void Tensor<T>::toCPU() {
    #ifdef CUDA_ENABLED
    if (is_on_gpu_ && gpu_data_) {
        cudaMemcpy(data_, gpu_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(gpu_data_);
        gpu_data_ = nullptr;
        is_on_gpu_ = false;
    }
    #endif
}

template<typename T>
void Tensor<T>::allocate(size_t size) {
    capacity_ = size;
    data_ = static_cast<T*>(getTensorPool().allocate(size * sizeof(T), 64));
    if (!data_) {
        throw std::bad_alloc();
    }
}

template<typename T>
void Tensor<T>::deallocate() {
    if (data_) {
        getTensorPool().deallocate(data_);
        data_ = nullptr;
    }
    #ifdef CUDA_ENABLED
    if (gpu_data_) {
        cudaFree(gpu_data_);
        gpu_data_ = nullptr;
    }
    #endif
    capacity_ = 0;
    is_on_gpu_ = false;
}

template<typename T>
size_t Tensor<T>::computeSize(const Shape& shape) const {
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

template<typename T>
size_t Tensor<T>::computeIndex(const std::vector<size_t>& indices) const {
    size_t index = 0;
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    return index;
}

template<typename T>
void Tensor<T>::simd_add(const T* a, const T* b, T* result, size_t size) const {
    if constexpr (std::is_floating_point_v<T>) {
        simd::SIMDOps<T>::add(a, b, result, size);
    } else {
        // Fallback for non-floating-point types
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

template<typename T>
void Tensor<T>::simd_mul(const T* a, const T* b, T* result, size_t size) const {
    if constexpr (std::is_floating_point_v<T>) {
        simd::SIMDOps<T>::multiply(a, b, result, size);
    } else {
        // Fallback for non-floating-point types
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
}

template<typename T>
void Tensor<T>::simd_scale(const T* a, T scalar, T* result, size_t size) const {
    if constexpr (std::is_floating_point_v<T>) {
        simd::SIMDOps<T>::scale(a, scalar, result, size);
    } else {
        // Fallback for non-floating-point types
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * scalar;
        }
    }
}

// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;

} // namespace ml_engine
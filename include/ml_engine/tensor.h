#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include <cstring>
#include <immintrin.h>

namespace ml_engine {

/**
 * @brief High-performance tensor class with SIMD support and custom memory management
 */
template<typename T>
class Tensor {
public:
    static_assert(std::is_arithmetic_v<T>, "Tensor type must be arithmetic");
    
    // Shape type for multi-dimensional tensors
    using Shape = std::vector<size_t>;
    
    /**
     * @brief Default constructor
     */
    Tensor() : data_(nullptr), size_(0), capacity_(0) {}
    
    /**
     * @brief Constructor with shape
     * @param shape Tensor dimensions
     */
    explicit Tensor(const Shape& shape);
    
    /**
     * @brief Constructor with shape and data
     * @param shape Tensor dimensions
     * @param data Initial data
     */
    Tensor(const Shape& shape, const std::vector<T>& data);
    
    /**
     * @brief Constructor with initializer list (1D)
     * @param data Initial data
     */
    Tensor(std::initializer_list<T> data);
    
    /**
     * @brief Destructor
     */
    ~Tensor();
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    /**
     * @brief Get tensor shape
     * @return Vector of dimensions
     */
    const Shape& shape() const { return shape_; }
    
    /**
     * @brief Get total number of elements
     * @return Element count
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Get raw data pointer
     * @return Pointer to data
     */
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    /**
     * @brief Access element by index (1D)
     * @param index Element index
     * @return Reference to element
     */
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    /**
     * @brief Access element by multi-dimensional index
     * @param indices Multi-dimensional indices
     * @return Reference to element
     */
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Reshape tensor (must preserve total size)
     * @param new_shape New tensor shape
     */
    void reshape(const Shape& new_shape);
    
    /**
     * @brief Element-wise addition with SIMD optimization
     * @param other Other tensor
     * @return Result tensor
     */
    Tensor operator+(const Tensor& other) const;
    
    /**
     * @brief Element-wise multiplication with SIMD optimization
     * @param other Other tensor
     * @return Result tensor
     */
    Tensor operator*(const Tensor& other) const;
    
    /**
     * @brief Scalar multiplication
     * @param scalar Scalar value
     * @return Result tensor
     */
    Tensor operator*(T scalar) const;
    
    /**
     * @brief Matrix multiplication (2D tensors)
     * @param other Other tensor
     * @return Result tensor
     */
    Tensor matmul(const Tensor& other) const;
    
    /**
     * @brief Apply function element-wise
     * @param func Function to apply
     * @return Result tensor
     */
    template<typename Func>
    Tensor apply(Func func) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = func(data_[i]);
        }
        return result;
    }
    
    /**
     * @brief Copy data to GPU
     * @param device_id CUDA device ID
     */
    void toGPU(int device_id = 0);
    
    /**
     * @brief Copy data from GPU to CPU
     */
    void toCPU();
    
    /**
     * @brief Check if tensor is on GPU
     * @return true if on GPU, false if on CPU
     */
    bool isOnGPU() const { return is_on_gpu_; }

private:
    T* data_;
    Shape shape_;
    size_t size_;
    size_t capacity_;
    bool is_on_gpu_ = false;
    void* gpu_data_ = nullptr;
    
    void allocate(size_t size);
    void deallocate();
    size_t computeSize(const Shape& shape) const;
    size_t computeIndex(const std::vector<size_t>& indices) const;
    
    // SIMD optimized operations
    void simd_add(const T* a, const T* b, T* result, size_t size) const;
    void simd_mul(const T* a, const T* b, T* result, size_t size) const;
    void simd_scale(const T* a, T scalar, T* result, size_t size) const;
};

// Type aliases for common tensor types
using Tensor32f = Tensor<float>;
using Tensor64f = Tensor<double>;

} // namespace ml_engine
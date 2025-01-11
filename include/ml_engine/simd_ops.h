#pragma once

#include <immintrin.h>
#include <type_traits>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace ml_engine {
namespace simd {

/**
 * @brief SIMD operations for different data types with template specialization
 */
template<typename T>
struct SIMDOps {
    static_assert(std::is_floating_point_v<T>, "SIMD operations only supported for floating point types");
    
    static void add(const T* a, const T* b, T* result, size_t size);
    static void multiply(const T* a, const T* b, T* result, size_t size);
    static void scale(const T* a, T scalar, T* result, size_t size);
    static void fused_multiply_add(const T* a, const T* b, const T* c, T* result, size_t size);
    static T dot_product(const T* a, const T* b, size_t size);
    static void relu(const T* input, T* output, size_t size);
    static void sigmoid(const T* input, T* output, size_t size);
    static void tanh_activation(const T* input, T* output, size_t size);
};

/**
 * @brief Specialization for float (32-bit) using AVX2/AVX-512
 */
template<>
struct SIMDOps<float> {
    static constexpr size_t VECTOR_SIZE = 8; // AVX2: 8 floats per vector
    
    static void add(const float* a, const float* b, float* result, size_t size) {
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_store_ps(&result[i], vr);
        }
        
        // Handle remaining elements
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    static void multiply(const float* a, const float* b, float* result, size_t size) {
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vr = _mm256_mul_ps(va, vb);
            _mm256_store_ps(&result[i], vr);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
    
    static void scale(const float* a, float scalar, float* result, size_t size) {
        __m256 vs = _mm256_set1_ps(scalar);
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vr = _mm256_mul_ps(va, vs);
            _mm256_store_ps(&result[i], vr);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] * scalar;
        }
    }
    
    static void fused_multiply_add(const float* a, const float* b, const float* c, float* result, size_t size) {
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vc = _mm256_load_ps(&c[i]);
            __m256 vr = _mm256_fmadd_ps(va, vb, vc);
            _mm256_store_ps(&result[i], vr);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    }
    
    static float dot_product(const float* a, const float* b, size_t size) {
        __m256 sum = _mm256_setzero_ps();
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        
        // Horizontal sum
        float result[8];
        _mm256_store_ps(result, sum);
        float final_sum = result[0] + result[1] + result[2] + result[3] + 
                         result[4] + result[5] + result[6] + result[7];
        
        // Handle remaining elements
        for (size_t i = vectorized_size; i < size; ++i) {
            final_sum += a[i] * b[i];
        }
        
        return final_sum;
    }
    
    static void relu(const float* input, float* output, size_t size) {
        __m256 zero = _mm256_setzero_ps();
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 v = _mm256_load_ps(&input[i]);
            __m256 result = _mm256_max_ps(v, zero);
            _mm256_store_ps(&output[i], result);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            output[i] = input[i] > 0.0f ? input[i] : 0.0f;
        }
    }
    
    static void sigmoid(const float* input, float* output, size_t size) {
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_one = _mm256_set1_ps(-1.0f);
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256 x = _mm256_load_ps(&input[i]);
            __m256 neg_x = _mm256_mul_ps(x, neg_one);
            
            // Approximate exp using polynomial approximation for performance
            __m256 exp_neg_x = exp_approx_avx2(neg_x);
            __m256 result = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
            _mm256_store_ps(&output[i], result);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));
        }
    }
    
    static void tanh_activation(const float* input, float* output, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::tanh(input[i]);
        }
    }

private:
    // Fast exponential approximation for SIMD
    static __m256 exp_approx_avx2(__m256 x) {
        // Polynomial approximation of exp(x)
        __m256 c1 = _mm256_set1_ps(1.0f);
        __m256 c2 = _mm256_set1_ps(1.0f);
        __m256 c3 = _mm256_set1_ps(0.5f);
        __m256 c4 = _mm256_set1_ps(0.16666667f);
        
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        
        __m256 result = _mm256_add_ps(c1, x);
        result = _mm256_fmadd_ps(c3, x2, result);
        result = _mm256_fmadd_ps(c4, x3, result);
        
        return result;
    }
};

/**
 * @brief Specialization for double (64-bit) using AVX2
 */
template<>
struct SIMDOps<double> {
    static constexpr size_t VECTOR_SIZE = 4; // AVX2: 4 doubles per vector
    
    static void add(const double* a, const double* b, double* result, size_t size) {
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vb = _mm256_load_pd(&b[i]);
            __m256d vr = _mm256_add_pd(va, vb);
            _mm256_store_pd(&result[i], vr);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    static void multiply(const double* a, const double* b, double* result, size_t size) {
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vb = _mm256_load_pd(&b[i]);
            __m256d vr = _mm256_mul_pd(va, vb);
            _mm256_store_pd(&result[i], vr);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
    
    static void scale(const double* a, double scalar, double* result, size_t size) {
        __m256d vs = _mm256_set1_pd(scalar);
        size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
        
        for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vr = _mm256_mul_pd(va, vs);
            _mm256_store_pd(&result[i], vr);
        }
        
        for (size_t i = vectorized_size; i < size; ++i) {
            result[i] = a[i] * scalar;
        }
    }
    
    // Additional methods similar to float specialization...
    static void fused_multiply_add(const double* a, const double* b, const double* c, double* result, size_t size);
    static double dot_product(const double* a, const double* b, size_t size);
    static void relu(const double* input, double* output, size_t size);
    static void sigmoid(const double* input, double* output, size_t size);
    static void tanh_activation(const double* input, double* output, size_t size);
};

/**
 * @brief Matrix multiplication with SIMD optimization
 */
template<typename T>
void simd_matrix_multiply(const T* A, const T* B, T* C, 
                         size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

} // namespace simd
} // namespace ml_engine
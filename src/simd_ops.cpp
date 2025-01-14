#include "ml_engine/simd_ops.h"
#include <cmath>

namespace ml_engine {
namespace simd {

// Implementation for double specialization methods
void SIMDOps<double>::fused_multiply_add(const double* a, const double* b, const double* c, double* result, size_t size) {
    size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
    
    for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vc = _mm256_load_pd(&c[i]);
        __m256d vr = _mm256_fmadd_pd(va, vb, vc);
        _mm256_store_pd(&result[i], vr);
    }
    
    for (size_t i = vectorized_size; i < size; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

double SIMDOps<double>::dot_product(const double* a, const double* b, size_t size) {
    __m256d sum = _mm256_setzero_pd();
    size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
    
    for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        sum = _mm256_fmadd_pd(va, vb, sum);
    }
    
    // Horizontal sum
    double result[4];
    _mm256_store_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3];
    
    // Handle remaining elements
    for (size_t i = vectorized_size; i < size; ++i) {
        final_sum += a[i] * b[i];
    }
    
    return final_sum;
}

void SIMDOps<double>::relu(const double* input, double* output, size_t size) {
    __m256d zero = _mm256_setzero_pd();
    size_t vectorized_size = (size / VECTOR_SIZE) * VECTOR_SIZE;
    
    for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
        __m256d v = _mm256_load_pd(&input[i]);
        __m256d result = _mm256_max_pd(v, zero);
        _mm256_store_pd(&output[i], result);
    }
    
    for (size_t i = vectorized_size; i < size; ++i) {
        output[i] = input[i] > 0.0 ? input[i] : 0.0;
    }
}

void SIMDOps<double>::sigmoid(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

void SIMDOps<double>::tanh_activation(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = tanh(input[i]);
    }
}

// Optimized matrix multiplication for different scenarios
template<>
void simd_matrix_multiply<float>(const float* A, const float* B, float* C, 
                                size_t M, size_t N, size_t K) {
    // Cache-friendly matrix multiplication with SIMD
    constexpr size_t BLOCK_SIZE = 64;
    
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                // Process block
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; jj += 8) { // Process 8 elements at once with AVX
                        __m256 sum = _mm256_setzero_ps();
                        
                        for (size_t kk = k; kk < k_end; ++kk) {
                            __m256 a_val = _mm256_set1_ps(A[ii * K + kk]);
                            __m256 b_vals = _mm256_load_ps(&B[kk * N + jj]);
                            sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                        }
                        
                        if (k == 0) {
                            _mm256_store_ps(&C[ii * N + jj], sum);
                        } else {
                            __m256 existing = _mm256_load_ps(&C[ii * N + jj]);
                            sum = _mm256_add_ps(sum, existing);
                            _mm256_store_ps(&C[ii * N + jj], sum);
                        }
                    }
                }
            }
        }
    }
}

template<>
void simd_matrix_multiply<double>(const double* A, const double* B, double* C, 
                                 size_t M, size_t N, size_t K) {
    // Similar implementation for double precision
    constexpr size_t BLOCK_SIZE = 64;
    
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; jj += 4) { // Process 4 doubles at once with AVX
                        __m256d sum = _mm256_setzero_pd();
                        
                        for (size_t kk = k; kk < k_end; ++kk) {
                            __m256d a_val = _mm256_set1_pd(A[ii * K + kk]);
                            __m256d b_vals = _mm256_load_pd(&B[kk * N + jj]);
                            sum = _mm256_fmadd_pd(a_val, b_vals, sum);
                        }
                        
                        if (k == 0) {
                            _mm256_store_pd(&C[ii * N + jj], sum);
                        } else {
                            __m256d existing = _mm256_load_pd(&C[ii * N + jj]);
                            sum = _mm256_add_pd(sum, existing);
                            _mm256_store_pd(&C[ii * N + jj], sum);
                        }
                    }
                }
            }
        }
    }
}

} // namespace simd
} // namespace ml_engine
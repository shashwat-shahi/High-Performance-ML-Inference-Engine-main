#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <functional>
#include <chrono>
#include <mutex>

namespace ml_engine {

/**
 * @brief Neural network graph optimization passes
 */
class GraphOptimizer {
public:
    struct Node {
        std::string op_type;
        std::vector<size_t> inputs;
        std::vector<size_t> outputs;
        std::unordered_map<std::string, float> attributes;
        
        Node(const std::string& type) : op_type(type) {}
    };
    
    struct Graph {
        std::vector<std::unique_ptr<Node>> nodes;
        std::vector<size_t> input_nodes;
        std::vector<size_t> output_nodes;
        
        // Delete copy constructor and assignment to prevent issues with unique_ptr
        Graph() = default;
        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;
        Graph(Graph&&) = default;
        Graph& operator=(Graph&&) = default;
    };
    
    GraphOptimizer();
    ~GraphOptimizer();
    
    /**
     * @brief Optimize the computation graph
     * @param graph Input graph to optimize
     * @return Optimized graph
     */
    Graph optimize(const Graph& graph);
    
    /**
     * @brief Enable specific optimization passes
     */
    void enableConstantFolding(bool enable = true) { constant_folding_ = enable; }
    void enableDeadCodeElimination(bool enable = true) { dead_code_elimination_ = enable; }
    void enableOperatorFusion(bool enable = true) { operator_fusion_ = enable; }
    void enableMemoryOptimization(bool enable = true) { memory_optimization_ = enable; }
    void enableLayoutOptimization(bool enable = true) { layout_optimization_ = enable; }

private:
    bool constant_folding_;
    bool dead_code_elimination_;
    bool operator_fusion_;
    bool memory_optimization_;
    bool layout_optimization_;
    
    // Optimization pass implementations
    Graph constantFolding(const Graph& graph);
    Graph deadCodeElimination(const Graph& graph);
    Graph operatorFusion(const Graph& graph);
    Graph memoryOptimization(const Graph& graph);
    Graph layoutOptimization(const Graph& graph);
    
    // Helper functions
    bool isConstantNode(const Node& node);
    bool canFuseNodes(const Node& node1, const Node& node2);
    std::vector<size_t> getUnusedNodes(const Graph& graph);
    void topologicalSort(Graph& graph);
};

/**
 * @brief Performance profiler for measuring execution time and memory usage
 */
class PerformanceProfiler {
public:
    PerformanceProfiler();
    ~PerformanceProfiler();
    
    /**
     * @brief Start timing a section
     * @param name Section name
     */
    void startTimer(const std::string& name);
    
    /**
     * @brief End timing a section
     * @param name Section name
     */
    void endTimer(const std::string& name);
    
    /**
     * @brief Record memory usage
     * @param name Memory section name
     * @param bytes Number of bytes used
     */
    void recordMemoryUsage(const std::string& name, size_t bytes);
    
    /**
     * @brief Get timing results
     * @return Map of section names to elapsed time in milliseconds
     */
    std::unordered_map<std::string, double> getTimingResults() const;
    
    /**
     * @brief Get memory usage results
     * @return Map of section names to memory usage in bytes
     */
    std::unordered_map<std::string, size_t> getMemoryResults() const;
    
    /**
     * @brief Print performance report
     */
    void printReport() const;
    
    /**
     * @brief Reset all measurements
     */
    void reset();

private:
    struct TimingInfo {
        std::chrono::high_resolution_clock::time_point start_time;
        double total_time_ms;
        size_t call_count;
    };
    
    std::unordered_map<std::string, TimingInfo> timing_data_;
    std::unordered_map<std::string, size_t> memory_data_;
    mutable std::mutex profiler_mutex_;
};

/**
 * @brief RAII timer for automatic profiling
 */
class ScopedTimer {
public:
    ScopedTimer(PerformanceProfiler& profiler, const std::string& name)
        : profiler_(profiler), name_(name) {
        profiler_.startTimer(name_);
    }
    
    ~ScopedTimer() {
        profiler_.endTimer(name_);
    }

private:
    PerformanceProfiler& profiler_;
    std::string name_;
};

#define PROFILE_SCOPE(profiler, name) \
    ScopedTimer timer(profiler, name)

/**
 * @brief Numerical accuracy checker
 */
class AccuracyChecker {
public:
    /**
     * @brief Check if two tensors are numerically equivalent within tolerance
     * @param expected Expected values
     * @param actual Actual values
     * @param size Number of elements
     * @param tolerance Relative tolerance (default 0.001 for 0.1%)
     * @return true if within tolerance, false otherwise
     */
    template<typename T>
    static bool checkAccuracy(const T* expected, const T* actual, size_t size, 
                             T tolerance = static_cast<T>(0.001)) {
        for (size_t i = 0; i < size; ++i) {
            T relative_error = std::abs((actual[i] - expected[i]) / (expected[i] + 1e-8));
            if (relative_error > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Calculate maximum relative error
     * @param expected Expected values
     * @param actual Actual values
     * @param size Number of elements
     * @return Maximum relative error
     */
    template<typename T>
    static T calculateMaxError(const T* expected, const T* actual, size_t size) {
        T max_error = 0;
        for (size_t i = 0; i < size; ++i) {
            T relative_error = std::abs((actual[i] - expected[i]) / (expected[i] + 1e-8));
            max_error = std::max(max_error, relative_error);
        }
        return max_error;
    }
    
    /**
     * @brief Calculate mean absolute error
     * @param expected Expected values
     * @param actual Actual values
     * @param size Number of elements
     * @return Mean absolute error
     */
    template<typename T>
    static T calculateMeanAbsoluteError(const T* expected, const T* actual, size_t size) {
        T sum_error = 0;
        for (size_t i = 0; i < size; ++i) {
            sum_error += std::abs(actual[i] - expected[i]);
        }
        return sum_error / size;
    }
};

} // namespace ml_engine
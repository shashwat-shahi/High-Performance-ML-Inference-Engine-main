#include "ml_engine/optimization.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <queue>

namespace ml_engine {

// GraphOptimizer implementation
GraphOptimizer::GraphOptimizer() 
    : constant_folding_(true), dead_code_elimination_(true), 
      operator_fusion_(true), memory_optimization_(true),
      layout_optimization_(true) {}

GraphOptimizer::~GraphOptimizer() = default;

GraphOptimizer::Graph GraphOptimizer::optimize(const Graph& graph) {
    // Create a new graph by moving from the input
    Graph optimized_graph;
    
    // Copy nodes using deep copy
    for (const auto& node : graph.nodes) {
        optimized_graph.nodes.push_back(std::make_unique<Node>(*node));
    }
    optimized_graph.input_nodes = graph.input_nodes;
    optimized_graph.output_nodes = graph.output_nodes;
    
    // Apply optimization passes in order
    if (constant_folding_) {
        optimized_graph = constantFolding(optimized_graph);
    }
    
    if (dead_code_elimination_) {
        optimized_graph = deadCodeElimination(optimized_graph);
    }
    
    if (operator_fusion_) {
        optimized_graph = operatorFusion(optimized_graph);
    }
    
    if (memory_optimization_) {
        optimized_graph = memoryOptimization(optimized_graph);
    }
    
    if (layout_optimization_) {
        optimized_graph = layoutOptimization(optimized_graph);
    }
    
    // Ensure topological ordering
    topologicalSort(optimized_graph);
    
    return optimized_graph;
}

GraphOptimizer::Graph GraphOptimizer::constantFolding(const Graph& graph) {
    Graph folded_graph;
    
    // Deep copy nodes
    for (const auto& node : graph.nodes) {
        auto new_node = std::make_unique<Node>(*node);
        
        if (isConstantNode(*new_node)) {
            // Mark node as constant for later optimization
            new_node->attributes["is_constant"] = 1.0f;
        }
        
        folded_graph.nodes.push_back(std::move(new_node));
    }
    
    folded_graph.input_nodes = graph.input_nodes;
    folded_graph.output_nodes = graph.output_nodes;
    
    return folded_graph;
}

GraphOptimizer::Graph GraphOptimizer::deadCodeElimination(const Graph& graph) {
    Graph pruned_graph;
    
    // Mark reachable nodes from outputs
    std::vector<bool> reachable(graph.nodes.size(), false);
    std::queue<size_t> to_visit;
    
    // Start from output nodes
    for (size_t output_idx : graph.output_nodes) {
        to_visit.push(output_idx);
        reachable[output_idx] = true;
    }
    
    // Backward traversal to mark all reachable nodes
    while (!to_visit.empty()) {
        size_t current = to_visit.front();
        to_visit.pop();
        
        if (current < graph.nodes.size()) {
            const auto& node = graph.nodes[current];
            for (size_t input_idx : node->inputs) {
                if (!reachable[input_idx]) {
                    reachable[input_idx] = true;
                    to_visit.push(input_idx);
                }
            }
        }
    }
    
    // Create mapping from old indices to new indices
    std::vector<size_t> index_mapping(graph.nodes.size());
    size_t new_idx = 0;
    
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (reachable[i]) {
            index_mapping[i] = new_idx++;
            pruned_graph.nodes.push_back(std::make_unique<Node>(*graph.nodes[i]));
        }
    }
    
    // Update input/output references
    for (auto& node : pruned_graph.nodes) {
        for (auto& input_idx : node->inputs) {
            input_idx = index_mapping[input_idx];
        }
        for (auto& output_idx : node->outputs) {
            output_idx = index_mapping[output_idx];
        }
    }
    
    // Update graph input/output nodes
    for (size_t input_idx : graph.input_nodes) {
        if (reachable[input_idx]) {
            pruned_graph.input_nodes.push_back(index_mapping[input_idx]);
        }
    }
    
    for (size_t output_idx : graph.output_nodes) {
        if (reachable[output_idx]) {
            pruned_graph.output_nodes.push_back(index_mapping[output_idx]);
        }
    }
    
    return pruned_graph;
}

GraphOptimizer::Graph GraphOptimizer::operatorFusion(const Graph& graph) {
    Graph fused_graph;
    
    // Deep copy nodes first
    for (const auto& node : graph.nodes) {
        fused_graph.nodes.push_back(std::make_unique<Node>(*node));
    }
    fused_graph.input_nodes = graph.input_nodes;
    fused_graph.output_nodes = graph.output_nodes;
    
    // Look for fusible operator patterns
    for (size_t i = 0; i < fused_graph.nodes.size() - 1; ++i) {
        auto& current_node = fused_graph.nodes[i];
        auto& next_node = fused_graph.nodes[i + 1];
        
        if (canFuseNodes(*current_node, *next_node)) {
            // Fuse Conv2D + BatchNorm + ReLU pattern
            if (current_node->op_type == "Conv2D" && 
                next_node->op_type == "BatchNorm") {
                
                // Look for ReLU after BatchNorm
                if (i + 2 < fused_graph.nodes.size() && 
                    fused_graph.nodes[i + 2]->op_type == "ReLU") {
                    
                    // Create fused node
                    auto fused_node = std::make_unique<Node>("FusedConvBnRelu");
                    fused_node->inputs = current_node->inputs;
                    fused_node->outputs = fused_graph.nodes[i + 2]->outputs;
                    
                    // Copy attributes from all three nodes
                    for (const auto& [key, value] : current_node->attributes) {
                        fused_node->attributes["conv_" + key] = value;
                    }
                    for (const auto& [key, value] : next_node->attributes) {
                        fused_node->attributes["bn_" + key] = value;
                    }
                    
                    // Replace the three nodes with the fused node
                    fused_graph.nodes[i] = std::move(fused_node);
                    fused_graph.nodes.erase(fused_graph.nodes.begin() + i + 1, 
                                           fused_graph.nodes.begin() + i + 3);
                }
            }
            
            // Fuse MatMul + Bias + Activation pattern
            else if (current_node->op_type == "MatMul" && 
                     next_node->op_type == "Add") {
                
                auto fused_node = std::make_unique<Node>("FusedLinear");
                fused_node->inputs = current_node->inputs;
                fused_node->outputs = next_node->outputs;
                
                // Copy attributes
                for (const auto& [key, value] : current_node->attributes) {
                    fused_node->attributes[key] = value;
                }
                
                // Replace nodes
                fused_graph.nodes[i] = std::move(fused_node);
                fused_graph.nodes.erase(fused_graph.nodes.begin() + i + 1);
            }
        }
    }
    
    return fused_graph;
}

GraphOptimizer::Graph GraphOptimizer::memoryOptimization(const Graph& graph) {
    Graph optimized_graph;
    
    // Deep copy nodes
    for (const auto& node : graph.nodes) {
        auto new_node = std::make_unique<Node>(*node);
        
        // Add memory optimization hints
        if (new_node->op_type == "Conv2D" || new_node->op_type == "MatMul") {
            new_node->attributes["memory_efficient"] = 1.0f;
        }
        
        // Prefer in-place operations where possible
        if (new_node->op_type == "ReLU" || new_node->op_type == "Add") {
            new_node->attributes["in_place"] = 1.0f;
        }
        
        optimized_graph.nodes.push_back(std::move(new_node));
    }
    
    optimized_graph.input_nodes = graph.input_nodes;
    optimized_graph.output_nodes = graph.output_nodes;
    
    return optimized_graph;
}

GraphOptimizer::Graph GraphOptimizer::layoutOptimization(const Graph& graph) {
    Graph optimized_graph;
    
    // Deep copy nodes
    for (const auto& node : graph.nodes) {
        auto new_node = std::make_unique<Node>(*node);
        
        if (new_node->op_type == "Conv2D") {
            // Prefer NCHW layout for convolutions
            new_node->attributes["data_layout"] = 0.0f; // 0 = NCHW, 1 = NHWC
        } else if (new_node->op_type == "MatMul") {
            // Prefer row-major layout for matrix operations
            new_node->attributes["weight_layout"] = 0.0f; // 0 = row-major, 1 = col-major
        }
        
        optimized_graph.nodes.push_back(std::move(new_node));
    }
    
    optimized_graph.input_nodes = graph.input_nodes;
    optimized_graph.output_nodes = graph.output_nodes;
    
    return optimized_graph;
}

bool GraphOptimizer::isConstantNode(const Node& node) {
    return node.op_type == "Constant" || 
           node.attributes.find("is_constant") != node.attributes.end();
}

bool GraphOptimizer::canFuseNodes(const Node& node1, const Node& node2) {
    // Check if nodes can be fused based on their types and dependencies
    
    // Conv2D can be fused with BatchNorm
    if (node1.op_type == "Conv2D" && node2.op_type == "BatchNorm") {
        return true;
    }
    
    // MatMul can be fused with Add (bias)
    if (node1.op_type == "MatMul" && node2.op_type == "Add") {
        return true;
    }
    
    // Any operation can be fused with element-wise activations
    if (node2.op_type == "ReLU" || node2.op_type == "Sigmoid" || node2.op_type == "Tanh") {
        return true;
    }
    
    return false;
}

void GraphOptimizer::topologicalSort(Graph& graph) {
    // Simple topological sort implementation
    std::vector<size_t> in_degree(graph.nodes.size(), 0);
    
    // Calculate in-degrees
    for (const auto& node : graph.nodes) {
        for (size_t input_idx : node->inputs) {
            if (input_idx < in_degree.size()) {
                in_degree[input_idx]++;
            }
        }
    }
    
    // Process nodes with zero in-degree
    std::queue<size_t> zero_in_degree;
    for (size_t i = 0; i < in_degree.size(); ++i) {
        if (in_degree[i] == 0) {
            zero_in_degree.push(i);
        }
    }
    
    std::vector<std::unique_ptr<Node>> sorted_nodes;
    while (!zero_in_degree.empty()) {
        size_t current = zero_in_degree.front();
        zero_in_degree.pop();
        
        if (current < graph.nodes.size()) {
            sorted_nodes.push_back(std::move(graph.nodes[current]));
            
            // Update in-degrees of dependent nodes
            for (size_t output_idx : sorted_nodes.back()->outputs) {
                if (output_idx < in_degree.size()) {
                    in_degree[output_idx]--;
                    if (in_degree[output_idx] == 0) {
                        zero_in_degree.push(output_idx);
                    }
                }
            }
        }
    }
    
    graph.nodes = std::move(sorted_nodes);
}

// PerformanceProfiler implementation
PerformanceProfiler::PerformanceProfiler() = default;
PerformanceProfiler::~PerformanceProfiler() = default;

void PerformanceProfiler::startTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    
    auto& timing_info = timing_data_[name];
    timing_info.start_time = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::endTimer(const std::string& name) {
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    
    auto it = timing_data_.find(name);
    if (it != timing_data_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second.start_time).count();
        
        it->second.total_time_ms += duration / 1000.0;
        it->second.call_count++;
    }
}

void PerformanceProfiler::recordMemoryUsage(const std::string& name, size_t bytes) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    memory_data_[name] = std::max(memory_data_[name], bytes);
}

std::unordered_map<std::string, double> PerformanceProfiler::getTimingResults() const {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    
    std::unordered_map<std::string, double> results;
    for (const auto& [name, timing_info] : timing_data_) {
        results[name] = timing_info.total_time_ms;
    }
    return results;
}

std::unordered_map<std::string, size_t> PerformanceProfiler::getMemoryResults() const {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    return memory_data_;
}

void PerformanceProfiler::printReport() const {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    
    std::cout << "\n=== Performance Report ===\n";
    std::cout << std::left << std::setw(25) << "Operation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(10) << "Calls" 
              << std::setw(15) << "Avg (ms)" << "\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (const auto& [name, timing_info] : timing_data_) {
        double avg_time = timing_info.call_count > 0 ? 
                         timing_info.total_time_ms / timing_info.call_count : 0.0;
        
        std::cout << std::left << std::setw(25) << name
                  << std::setw(15) << std::fixed << std::setprecision(3) << timing_info.total_time_ms
                  << std::setw(10) << timing_info.call_count
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time << "\n";
    }
    
    std::cout << "\n=== Memory Usage ===\n";
    std::cout << std::left << std::setw(25) << "Section" 
              << std::setw(15) << "Bytes" 
              << std::setw(10) << "MB" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (const auto& [name, bytes] : memory_data_) {
        double mb = bytes / (1024.0 * 1024.0);
        std::cout << std::left << std::setw(25) << name
                  << std::setw(15) << bytes
                  << std::setw(10) << std::fixed << std::setprecision(2) << mb << "\n";
    }
    
    std::cout << "\n";
}

void PerformanceProfiler::reset() {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    timing_data_.clear();
    memory_data_.clear();
}

} // namespace ml_engine
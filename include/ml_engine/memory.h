#pragma once

#include <memory>
#include <atomic>
#include <vector>
#include <cstddef>
#include <mutex>
#include <cassert>

namespace ml_engine {

/**
 * @brief Custom memory allocator with pool management for high-performance allocation
 */
class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size = 1024 * 1024 * 1024); // 1GB default
    ~MemoryPool();
    
    // Disable copy, enable move
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = default;
    MemoryPool& operator=(MemoryPool&&) = default;
    
    /**
     * @brief Allocate aligned memory
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (default 64 for SIMD)
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size, size_t alignment = 64);
    
    /**
     * @brief Deallocate memory
     * @param ptr Pointer to memory to deallocate
     */
    void deallocate(void* ptr);
    
    /**
     * @brief Get total pool size
     * @return Pool size in bytes
     */
    size_t getPoolSize() const { return pool_size_; }
    
    /**
     * @brief Get used memory
     * @return Used memory in bytes
     */
    size_t getUsedMemory() const { return used_memory_.load(); }
    
    /**
     * @brief Reset the memory pool (free all allocations)
     */
    void reset();

private:
    struct Block {
        void* ptr;
        size_t size;
        bool is_free;
        Block* next;
        
        Block(void* p, size_t s, bool free = true) 
            : ptr(p), size(s), is_free(free), next(nullptr) {}
    };
    
    void* pool_memory_;
    size_t pool_size_;
    std::atomic<size_t> used_memory_;
    Block* free_list_;
    std::mutex allocation_mutex_;
    
    Block* findFreeBlock(size_t size, size_t alignment);
    void mergeBlocks();
    void* alignPointer(void* ptr, size_t alignment);
};

/**
 * @brief RAII wrapper for memory pool allocations
 */
template<typename T>
class PoolAllocator {
public:
    explicit PoolAllocator(MemoryPool& pool) : pool_(pool) {}
    
    T* allocate(size_t n) {
        return static_cast<T*>(pool_.allocate(n * sizeof(T), alignof(T)));
    }
    
    void deallocate(T* ptr, size_t) {
        pool_.deallocate(ptr);
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return &pool_ == &other.pool_;
    }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }

private:
    MemoryPool& pool_;
};

/**
 * @brief Lock-free stack for high-performance concurrent operations
 */
template<typename T>
class LockFreeStack {
public:
    LockFreeStack() : head_(nullptr) {}
    
    ~LockFreeStack() {
        while (Node* old_head = head_.load()) {
            head_ = old_head->next;
            delete old_head;
        }
    }
    
    void push(T item) {
        Node* new_node = new Node(std::move(item));
        new_node->next = head_.load();
        while (!head_.compare_exchange_weak(new_node->next, new_node));
    }
    
    bool pop(T& result) {
        Node* old_head = head_.load();
        while (old_head && !head_.compare_exchange_weak(old_head, old_head->next));
        
        if (old_head) {
            result = std::move(old_head->data);
            delete old_head;
            return true;
        }
        return false;
    }
    
    bool empty() const {
        return head_.load() == nullptr;
    }

private:
    struct Node {
        T data;
        Node* next;
        explicit Node(T&& item) : data(std::move(item)), next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
};

/**
 * @brief Lock-free queue for producer-consumer scenarios
 */
template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue() {
        Node* dummy = new Node;
        head_.store(dummy);
        tail_.store(dummy);
    }
    
    ~LockFreeQueue() {
        while (Node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }
    
    void enqueue(T item) {
        Node* new_node = new Node;
        new_node->data = std::move(item);
        new_node->next.store(nullptr);
        
        Node* prev_tail = tail_.exchange(new_node);
        prev_tail->next.store(new_node);
    }
    
    bool dequeue(T& result) {
        Node* head = head_.load();
        Node* next = head->next.load();
        
        if (next == nullptr) {
            return false;
        }
        
        result = std::move(next->data);
        head_.store(next);
        delete head;
        return true;
    }
    
    bool empty() const {
        Node* head = head_.load();
        Node* next = head->next.load();
        return next == nullptr;
    }

private:
    struct Node {
        std::atomic<Node*> next;
        T data;
        Node() : next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
};

} // namespace ml_engine
#include "ml_engine/memory.h"
#include <cstdlib>
#include <algorithm>
#include <stdexcept>

namespace ml_engine {

MemoryPool::MemoryPool(size_t pool_size) 
    : pool_size_(pool_size), used_memory_(0), free_list_(nullptr) {
    // Allocate aligned memory pool
    pool_memory_ = std::aligned_alloc(64, pool_size);
    if (!pool_memory_) {
        throw std::bad_alloc();
    }
    
    // Initialize free list with entire pool
    free_list_ = new Block(pool_memory_, pool_size, true);
}

MemoryPool::~MemoryPool() {
    if (pool_memory_) {
        std::free(pool_memory_);
    }
    
    // Clean up free list
    Block* current = free_list_;
    while (current) {
        Block* next = current->next;
        delete current;
        current = next;
    }
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Find suitable free block
    Block* block = findFreeBlock(size, alignment);
    if (!block) {
        return nullptr; // Out of memory
    }
    
    // Calculate aligned pointer
    void* aligned_ptr = alignPointer(block->ptr, alignment);
    size_t aligned_offset = static_cast<char*>(aligned_ptr) - static_cast<char*>(block->ptr);
    size_t total_size = size + aligned_offset;
    
    if (block->size < total_size) {
        return nullptr; // Not enough space after alignment
    }
    
    // Split block if necessary
    if (block->size > total_size + sizeof(Block)) {
        Block* new_block = new Block(
            static_cast<char*>(block->ptr) + total_size,
            block->size - total_size,
            true
        );
        new_block->next = block->next;
        block->next = new_block;
        block->size = total_size;
    }
    
    block->is_free = false;
    used_memory_.fetch_add(total_size);
    
    return aligned_ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Find the block containing this pointer
    Block* current = free_list_;
    while (current) {
        char* block_start = static_cast<char*>(current->ptr);
        char* block_end = block_start + current->size;
        char* ptr_char = static_cast<char*>(ptr);
        
        if (ptr_char >= block_start && ptr_char < block_end) {
            current->is_free = true;
            used_memory_.fetch_sub(current->size);
            mergeBlocks();
            return;
        }
        current = current->next;
    }
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Clean up existing free list
    Block* current = free_list_;
    while (current) {
        Block* next = current->next;
        delete current;
        current = next;
    }
    
    // Recreate single free block for entire pool
    free_list_ = new Block(pool_memory_, pool_size_, true);
    used_memory_.store(0);
}

MemoryPool::Block* MemoryPool::findFreeBlock(size_t size, size_t alignment) {
    Block* current = free_list_;
    while (current) {
        if (current->is_free) {
            void* aligned_ptr = alignPointer(current->ptr, alignment);
            size_t aligned_offset = static_cast<char*>(aligned_ptr) - static_cast<char*>(current->ptr);
            if (current->size >= size + aligned_offset) {
                return current;
            }
        }
        current = current->next;
    }
    return nullptr;
}

void MemoryPool::mergeBlocks() {
    Block* current = free_list_;
    while (current && current->next) {
        if (current->is_free && current->next->is_free) {
            char* current_end = static_cast<char*>(current->ptr) + current->size;
            char* next_start = static_cast<char*>(current->next->ptr);
            
            // Check if blocks are adjacent
            if (current_end == next_start) {
                Block* next_block = current->next;
                current->size += next_block->size;
                current->next = next_block->next;
                delete next_block;
                continue; // Don't advance current, check for more merges
            }
        }
        current = current->next;
    }
}

void* MemoryPool::alignPointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

} // namespace ml_engine
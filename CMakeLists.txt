cmake_minimum_required(VERSION 3.18)
project(HighPerformanceMLInferenceEngine LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -fsanitize=undefined")

# Find required packages
find_package(OpenMP REQUIRED)

# Optional CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        add_definitions(-DCUDA_ENABLED)
    else()
        message(WARNING "CUDA requested but not found. Building CPU-only version.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Source files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp")

# Conditionally add CUDA sources
if(ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
    file(GLOB_RECURSE CUDA_HEADERS "include/*.cuh")
    list(APPEND HEADERS ${CUDA_HEADERS})
    set(ALL_SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})
else()
    set(ALL_SOURCES ${CPP_SOURCES})
endif()

# Create library
add_library(ml_inference_engine STATIC ${ALL_SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(ml_inference_engine
    OpenMP::OpenMP_CXX
)

if(ENABLE_CUDA)
    target_link_libraries(ml_inference_engine
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
    )
    set_target_properties(ml_inference_engine PROPERTIES
        CUDA_ARCHITECTURES "70;75;80;86"
    )
endif()

# Set target properties
set_target_properties(ml_inference_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Create executable for testing
add_executable(inference_test tests/main.cpp)
target_link_libraries(inference_test ml_inference_engine)

# Enable testing
enable_testing()
add_test(NAME inference_test COMMAND inference_test)

# Benchmarking executable
add_executable(benchmark benchmarks/benchmark.cpp)
target_link_libraries(benchmark ml_inference_engine)
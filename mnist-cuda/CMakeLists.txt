cmake_minimum_required(VERSION 3.18)
project(mnist_cuda LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA and cuBLAS
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
# SM 7.5 = Turing (T4, RTX 20xx), SM 8.0 = Ampere (A100, RTX 30xx)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# v3: C implementation
add_executable(v3 src/v3.c)
target_link_libraries(v3 PRIVATE m)

# v4: Naive CUDA kernels
add_executable(v4 src/v4.cu)

# v5: cuBLAS optimized
add_executable(v5 src/v5.cu)
target_link_libraries(v5 PRIVATE CUDA::cublas)

# v6: Streams + Pinned Memory + Fusion
add_executable(v6 src/v6.cu)
target_link_libraries(v6 PRIVATE CUDA::cublas)

# v7: Custom Tiled GEMM
add_executable(v7 src/v7.cu)
target_link_libraries(v7 PRIVATE CUDA::cublas)

# v8: Pure FP16
add_executable(v8 src/v8.cu)
target_link_libraries(v8 PRIVATE CUDA::cublas)

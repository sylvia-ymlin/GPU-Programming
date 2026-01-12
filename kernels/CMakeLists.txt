cmake_minimum_required(VERSION 3.18)
project(cuda_kernels_from_scratch LANGUAGES CUDA CXX)

# Build settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If you want to control GPU arch, uncomment and set accordingly (e.g., 80 for A100)
# set(CMAKE_CUDA_ARCHITECTURES 75)

# Elementwise test binary
add_executable(test_elementwise
    tests/test_elementwise.cu
    elementwise/elementwise_add.cu
    elementwise/activations.cu
)

target_include_directories(test_elementwise PRIVATE
    elementwise
)

# Enable separable compilation for CUDA translation units
set_target_properties(test_elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Reduction test binary
add_executable(test_reduction
    tests/test_reduction.cu
    reduce/reduce_sum.cu
    reduce/reduce_max.cu
    reduce/reduce_softmax.cu
    reduce/reduce_softmax_matrix.cu
)

target_include_directories(test_reduction PRIVATE
    reduce
)

# Keep files separate with separable compilation
set_target_properties(test_reduction PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# SGEMM test binary
add_executable(test_sgemm
    tests/test_sgemm.cu
    sgemm/sgemm_v1_naive.cu
    sgemm/sgemm_v2.cu
    sgemm/sgemm_v3.cu
    sgemm/sgemm_v4.cu
    sgemm/sgemm_v5.cu
    sgemm/sgemm_v6.cu
    sgemm/sgemm_v7.cu
)

target_include_directories(test_sgemm PRIVATE
    sgemm
)

target_link_libraries(test_sgemm cublas)
set_target_properties(test_sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Transpose test binary
add_executable(test_transpose
    tests/test_transpose.cu
    transpose/transpose.cu
)

target_include_directories(test_transpose PRIVATE
    transpose
)

set_target_properties(test_transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

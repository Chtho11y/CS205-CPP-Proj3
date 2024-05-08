cmake_minimum_required(VERSION 3.18)

add_definitions(-O3)
add_definitions(-mavx2)
add_definitions(-mavx512f)
add_definitions(-fopenmp)
add_definitions(-mfma)

project(mat)

enable_language(CUDA)

include_directories(inc)
file(GLOB SRCS "./src/*.c" "./src/*.cu")

add_executable(mat ${SRCS})

set_property(TARGET mat PROPERTY CUDA_ARCHITECTURES "75")
set(CUDA_NVCC_FLAGS -O3;-Xptxas -O3,-v)

include_directories(./openblas)
link_directories(./openblas/lib)
target_link_libraries(mat PUBLIC openblas)


find_package(OpenMP)
if(OpenMP_C_FOUND)
    message( "openMP found. ")
    target_link_libraries(mat PUBLIC OpenMP::OpenMP_C)
endif()
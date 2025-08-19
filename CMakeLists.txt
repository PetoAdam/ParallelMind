cmake_minimum_required(VERSION 3.18)
project(ParallelMind LANGUAGES CXX CUDA)

# Enable CUDA
find_package(CUDAToolkit REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src)

# Add the core library
add_library(core
    src/core/Matrix.cu
    src/core/Node.cpp
    src/core/Layer.cpp
    src/core/Network.cpp
)

add_library(utils
    src/utils/NpyReader.cpp
)

target_link_libraries(core PRIVATE CUDA::cudart)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(examples)

# Enable testing
enable_testing()
add_subdirectory(tests)

# Hello World executable for testing CUDA
add_executable(hello_world src/gpu/hello_world.cu)
target_link_libraries(hello_world PRIVATE CUDA::cudart)

# (no core executable; examples define their own mains)


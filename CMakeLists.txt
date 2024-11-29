cmake_minimum_required(VERSION 3.18)
project(ParallelMind LANGUAGES CXX CUDA)

# Enable CUDA
find_package(CUDAToolkit REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(examples)

# Enable testing
enable_testing()
add_subdirectory(tests)

add_executable(hello_world src/gpu/hello_world.cu)
target_link_libraries(hello_world PRIVATE CUDA::cudart)


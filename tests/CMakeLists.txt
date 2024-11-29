# Enable testing
enable_testing()

# Add test executable
add_executable(test1 test1.cpp)

# Link the test executable to the core library and CUDA libraries
target_link_libraries(test1 PRIVATE core CUDA::cudart)

# Ensure CUDA include paths are available to the test target
target_include_directories(test1 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Register the test with CTest
add_test(NAME Test1 COMMAND test1)

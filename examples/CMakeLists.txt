add_executable(dummy_example main.cpp)
add_executable(mnist_example mnist_main.cpp)
target_link_libraries(mnist_example PRIVATE CUDA::cudart core utils)

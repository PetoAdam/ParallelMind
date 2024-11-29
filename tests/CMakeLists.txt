# Enable testing
enable_testing()

# Find all test source files (e.g., all .cpp files in the tests folder)
file(GLOB TEST_SOURCES "test*.cpp")

# Loop over each test source and create a corresponding executable and test
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the name of the test executable by removing the path and extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Add test executable for each source file
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Link the test executable to the core library and CUDA libraries
    target_link_libraries(${TEST_NAME} PRIVATE core CUDA::cudart)

    # Ensure CUDA include paths are available to the test target
    target_include_directories(${TEST_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Register the test with CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

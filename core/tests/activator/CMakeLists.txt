# CMakeLists.txt for activation component tests
# Located at: core/tests/activation/CMakeLists.txt

set(TEST_EXECUTABLE_NAME runActivationComponentTests)

# Test Source Files
add_executable(${TEST_EXECUTABLE_NAME}
    sigmoid/test_sigmoid.cc
    # test_matrix_operations.cpp # Example: if you have multiple test files for matrix
)

# Core Library, gtest_main, armadillo links
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    DenkCore                     # Your main library
    gtest_main                   # Google Test
    ${ARMADILLO_LIBRARIES}       # CHANGED: Use the variable here
)

# Add Armadillo include directories if your test code directly includes <armadillo>
# or if it's needed for compiling against DenkCore's headers that might expose Armadillo types.
target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${ARMADILLO_INCLUDE_DIRS}    # ADDED/CHANGED: Use the variable here
)

include(GoogleTest)
gtest_discover_tests(${TEST_EXECUTABLE_NAME})
# CMakeLists.txt for matrix component tests
# Located at: core/tests/matrix/CMakeLists.txt

set(TEST_EXECUTABLE_NAME runMatrixComponentTests)

# Add the source file(s) for this component's tests
add_executable(${TEST_EXECUTABLE_NAME}
    test_matrix_constructor.cc
    # test_matrix_operations.cpp # Example: if you have multiple test files for matrix
)

# Link against your core library, gtest_main, and Armadillo (using the variables)
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
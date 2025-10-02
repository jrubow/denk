cmake_minimum_required(VERSION 3.14)
project(denk LANGUAGES CXX)

# Armadillo Initialization
find_package(Armadillo REQUIRED)

message(STATUS "Armadillo found: ${ARMADILLO_FOUND}")
message(STATUS "Armadillo include_dirs: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")

if(TARGET Armadillo::armadillo)
    message(STATUS "SUCCESS: Armadillo::armadillo target IS defined.")
    # Use modern target-based linking
    set(ARMADILLO_LINK_TARGET Armadillo::armadillo)
else()
    message(WARNING "FAILURE: Armadillo::armadillo target IS NOT defined after find_package.")
    message(STATUS "This means linking with 'Armadillo::armadillo' will fail.")
    message(STATUS "Linking with '${ARMADILLO_LIBRARIES}' (and using include_dirs '${ARMADILLO_INCLUDE_DIRS}') instead.")

    # Fallback: manually add include dirs + libs
    include_directories(${ARMADILLO_INCLUDE_DIRS})
    set(ARMADILLO_LINK_TARGET ${ARMADILLO_LIBRARIES})
endif()


#################################
#       DenkCore Library
#################################

# Library Files for DenkCore
set(CORE_SOURCES
    core/matrix/matrix.cc
    core/activation/sigmoid.cc
    core/layer/layer.cc
    core/loss/mse.cc
)
set(CORE_HEADERS
    core/core.hh
    core/matrix/matrix.hh
    
    # Activation Headers
    core/activation/activation.hh
    core/activation/sigmoid.hh
    
    core/layer/layer.hh
    
    # Loss Headers
    core/loss/loss.hh
    core/loss/mse.hh
)

add_library(DenkCore ${CORE_SOURCES} ${CORE_HEADERS})

# Link Armadillo using the variables
target_link_libraries(DenkCore PRIVATE ${ARMADILLO_LIBRARIES})
# Add Armadillo include directories
target_include_directories(DenkCore PRIVATE ${ARMADILLO_INCLUDE_DIRS})

# Include directories for DenkCore Headers
target_include_directories(DenkCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/matrix"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/activation"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/layer"
)


#################################
#       DenkModels Library
#################################

# Library Files for DenkCore
set(MODEL_SOURCES
    # Model Sources
    model/bnn/bnn.cc
)

set(MODEL_HEADERS
    model/model.hh
    model/bnn/bnn.hh
)

add_library(DenkModels ${MODEL_SOURCES} ${MODEL_HEADERS})

# Include directories for DenkCore Headers
target_include_directories(DenkModels PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/model/bnn"
)
target_link_libraries(DenkModels PRIVATE DenkCore)

# Main Executable
add_executable(DenkApp main.cc)
target_link_libraries(DenkApp PRIVATE DenkCore)

# Google Test Integration
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test Subdirectories
add_subdirectory(core/tests/matrix)
add_subdirectory(core/tests/activation)
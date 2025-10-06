cmake_minimum_required(VERSION 3.14)
project(denk LANGUAGES CXX C)

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
    
    # Activators
    core/activator/sigmoid.cc
    
    # Layers
    core/layer/layer.cc

    # Loss
    core/loss/mse.cc

    # Optimizers
    core/optimizer/sgd.cc
    core/optimizer/newtonsmethod.cc

    # Logger
    core/logger/logger.c

)
#set(CORE_HEADERS
#    core/core.hh
#    core/matrix/matrix.hh
#    
#    # Activators
#    core/activator/activator.hh
#    core/activator/sigmoid.hh
#    
#    # Layers
#    core/layer/layer.hh
#    
#    # Loss
#    core/loss/loss.hh
#    core/loss/mse.hh
#
#    # Optimizers
#    core/optimizer/optimizer.hh
#    core/optimizer/sgd.hh
#    core/optimizer/newtonsmethod.hh
#   
#    # Loggers
#    core/logger/logger.h
#
#    # Status
#    core/status/status.h
#
#)

add_library(DenkCore ${CORE_SOURCES} ${CORE_HEADERS})

# Link Armadillo using the variables
target_link_libraries(DenkCore PRIVATE ${ARMADILLO_LIBRARIES})
# Add Armadillo include directories
target_include_directories(DenkCore PRIVATE ${ARMADILLO_INCLUDE_DIRS})

# Include directories for DenkCore Headers
target_include_directories(DenkCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/matrix"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/activator"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/layer"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/loss"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/optimizer"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/logger"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/status"
)


#################################
#       DenkModel Library
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

add_library(DenkModel ${MODEL_SOURCES} ${MODEL_HEADERS})

# Include directories for DenkCore Headers
target_include_directories(DenkModel PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/model/bnn"
)
target_link_libraries(DenkModel PRIVATE DenkCore)

# Main Executable
add_executable(DenkApp main.cc)
target_link_libraries(DenkApp PRIVATE DenkCore)
target_link_libraries(DenkApp PRIVATE DenkModel)

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
add_subdirectory(core/tests/activator)

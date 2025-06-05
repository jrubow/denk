cmake_minimum_required(VERSION 3.14)
project(denk LANGUAGES CXX)

# --- Armadillo Setup ---
find_package(Armadillo REQUIRED)
message(STATUS "Armadillo found: ${ARMADILLO_FOUND}")
message(STATUS "Armadillo include_dirs: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")

if(TARGET Armadillo::armadillo)
    message(STATUS "SUCCESS: Armadillo::armadillo target IS defined.")
else()
    message(WARNING "FAILURE: Armadillo::armadillo target IS NOT defined after find_package.")
    message(STATUS "This means linking with 'Armadillo::armadillo' will fail.")
    message(STATUS "Linking with '${ARMADILLO_LIBRARIES}' (and using include_dirs '${ARMADILLO_INCLUDE_DIRS}') instead.")
endif()

# --- Define a Library for your Core Code (e.g., Matrix class) ---
set(CORE_SOURCES
    core/matrix/matrix.cc
)
set(CORE_HEADERS
    core/matrix/matrix.hh
)

add_library(DenkCore ${CORE_SOURCES} ${CORE_HEADERS})

# Link Armadillo using the variables
target_link_libraries(DenkCore PRIVATE ${ARMADILLO_LIBRARIES}) # CHANGED
# Add Armadillo include directories
target_include_directories(DenkCore PRIVATE ${ARMADILLO_INCLUDE_DIRS}) # ADDED

# Set include directories for DenkCore's own headers
target_include_directories(DenkCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/core/matrix"
)

# --- Define Your Main Executable ---
add_executable(DenkApp main.cc)
target_link_libraries(DenkApp PRIVATE DenkCore)

# --- Google Test Integration using FetchContent ---
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        main # Or main / release-1.13.0 if 1.14.0 still has checkout issues
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Add Tests Subdirectories ---
add_subdirectory(core/tests/matrix)
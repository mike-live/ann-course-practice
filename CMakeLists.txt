cmake_minimum_required(VERSION 2.8.2)

set(PROJECT_NAME ann-course-practice)
project(${PROJECT_NAME})

include(CTest)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(DownloadProject/DownloadProject.cmake)
download_project(PROJ                googletest
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             master
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

# When using CMake 2.8.11 or later, header path dependencies
# are automatically added to the gtest and gmock targets.
# For earlier CMake versions, we have to explicitly add the
# required directories to the header search path ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()

# Set compiler flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE)

# ENVIROMENT:

set(ANN_LIBRARY ann-library)
set(ANN_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/code/ann-library/src")
set(ANN_TESTS "test-${ANN_LIBRARY}")
set(LAB1_MNIST lab1-mnist)

include_directories("${ANN_INCLUDE}")

# Enables 'test' target
enable_testing()

# BUILD:
# - lab1-mnist
add_subdirectory("code/lab1-mnist/src")

# - ann-library
add_subdirectory("code/ann-library/src")

# - tests
add_subdirectory("code/ann-library/tests")

# DATA FILES

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/download")
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz "${CMAKE_BINARY_DIR}/download/train-images-idx3-ubyte.gz"
	 SHOW_PROGRESS)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz "${CMAKE_BINARY_DIR}/download/train-labels-idx1-ubyte.gz"
	 SHOW_PROGRESS)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz "${CMAKE_BINARY_DIR}/download/t10k-images-idx3-ubyte.gz"
	 SHOW_PROGRESS)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz "${CMAKE_BINARY_DIR}/download/t10k-labels-idx1-ubyte.gz"
	 SHOW_PROGRESS)

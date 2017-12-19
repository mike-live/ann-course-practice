set(target ${ANN_TESTS})

FILE(GLOB hdrs "*.h")
FILE(GLOB hdrs "*.cpp")

include_directories("${ANN_INCLUDE}")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} gtest gmock_main ${ANN_LIBRARY})
# Register the test
add_test(NAME ${target} COMMAND "${target}" --gtest_color=yes)

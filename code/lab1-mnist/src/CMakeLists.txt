set(target ${LAB1_MNIST})

FILE(GLOB hdrs "*.h")
FILE(GLOB hdrs "*.cpp")

add_executable(${target} ${srcs} ${hdrs})

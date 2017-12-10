set(target ${ANN_LIBRARY})

FILE(GLOB hdrs "*.h")
FILE(GLOB hdrs "*.cpp")

add_library(${target} STATIC ${srcs} ${hdrs})
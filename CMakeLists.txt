cmake_minimum_required(VERSION 3.17)
project(darknet-mini)

set(CMAKE_C_STANDARD 99)

include_directories(include)

add_executable(darknet-mini
        cfg/data/cifar.data
        cfg/networks/cls_cifar_small.cfg
        main.c
        include/darknet.h
        src/classifier.c
        src/utils/utils.h
        src/utils/utils.c
        src/utils/parse.h
        src/utils/parse.c
        src/data_struct/list.h
        src/data_struct/list.c src/layers/network.h src/layers/network.c)

# 在Clion中生成的CMakeList.txt, 添加pthread库.
find_package(Threads)
target_link_libraries(darknet-mini ${CMAKE_THREAD_LIBS_INIT})
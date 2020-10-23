cmake_minimum_required(VERSION 3.12)
project(darknet-mini)

set(CMAKE_C_STANDARD 99)

include_directories(include)

add_executable(darknet-mini
        cfg/data/cifar.data
        cfg/net/cls_cifar_small.cfg
        main.c
        include/darknet.h
        src/classifier.c
        src/utils/utils.h
        src/utils/utils.c
        src/utils/parse.h
        src/utils/parse.c
        src/data_struct/list.h
        src/data_struct/list.c
        src/layers/network.h
        src/layers/network.c
        src/layers/convolutional_layer.h
        src/layers/convolutional_layer.c
        src/layers/activation_layer.h
        src/layers/activation_layer.c
        src/utils/blas.h
        src/utils/blas.c
        src/layers/activations.h
        src/layers/activations.c
        src/utils/im2col.h
        src/utils/im2col.c
        src/utils/gemm.h
        src/utils/gemm.c
        src/utils/col2im.h
        src/utils/col2im.c
        src/layers/batchnorm_layer.h
        src/layers/batchnorm_layer.c
        src/layers/maxpool_layer.h
        src/layers/maxpool_layer.c
        src/layers/avgpool_layer.h
        src/layers/avgpool_layer.c
        src/layers/softmax_layer.h
        src/layers/softmax_layer.c
        src/data_struct/sections.h
        src/data_struct/sections.c
        src/utils/data.h
        src/utils/data.c
        src/layers/layer.h
        src/layers/layer.c
        src/data_struct/matrix.h
        src/data_struct/matrix.c
        src/data_struct/image.h
        src/data_struct/image.c
        src/data_struct/image_opencv.cpp)

# 在Clion中生成的CMakeList.txt, 添加pthread库.
find_package(Threads)
target_link_libraries(darknet-mini ${CMAKE_THREAD_LIBS_INIT})

# cmake中添加math.h标准库
target_link_libraries(darknet-mini m)

# 添加opencv库
 set(OpenCV_DIR /home/cxj/Documents/opencv-3.4.2/build/CMakeFiles)
find_package(OpenCV)
target_link_libraries(darknet-mini ${OpenCV_LIBS})
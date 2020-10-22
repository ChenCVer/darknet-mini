#include "darknet.h"
#include <stdio.h>

// 函数的声明中带有关键字extern, 仅仅是暗示这个函数可能在别的源文档里定义.
extern void run_classifier(int argc, char **argv);


int main(int argc, char **argv)
{

//    /************下面配置用于分类网络的代码分析****************************
    // 参考: https://pjreddie.com/darknet/train-cifar/
    argc = 5;
    char* parameters[] = {"./darknet",
                          "classifier",
                          "train",
                          "/home/georg/Desktop/darknet-mini/cfg/data/cifar.data",
                          "/home/georg/Desktop/darknet-mini/cfg/net/cls_cifar_small.cfg"};
    argv = parameters;
//    *****************************************************************/

    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }

    if (0 == strcmp(argv[1], "classifier")){
         run_classifier(argc, argv);
    }
    else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }


    return 0;
}
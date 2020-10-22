#include "darknet.h"
#include "utils/utils.h"
#include "utils/parse.h"

void train_classifier(char* datacfg, char* cfgfile, char* weightfile){

    int i;
    float avg_loss = -1;

    char *base = basecfg(cfgfile);  // 分析(网络结构)配置文件
    printf("network cfg: %s\n", base);

    // nets指向大小为1*sizeof(network)的内存空间
    network* nets = (network*)xcalloc(1, sizeof(network));

    // 解析网络配置文件, 为nets中的每个net进行初始化
    nets[0] = parse_network_cfg(cfgfile);

    // 加载预训练权重
    if(weightfile){
        // TODO: 待办.
        // load_weights(&nets[i], weightfile);
    }

    // 设置随机种子
    srand(time(0));
    int seed = rand();

    printf("trian_classifier is running...\n");
}



void run_classifier(int argc, char** argv){

    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *data = argv[3];  // 获取数据文件路径
    char *cfg = argv[4];   // 获取配置文件路径
    char *weights = (argc > 5) ? argv[5] : 0;  // 预训练权重路径

    if (0 == strcmp(argv[2], "train"))
        train_classifier(data, cfg, weights);
}
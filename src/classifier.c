#include "darknet.h"
#include "utils/utils.h"
#include "utils/parse.h"
#include "data_struct/list.h"
#include "utils/data.h"

void train_classifier(char* datacfg, char* cfgfile, char* weightfile){

    char *base = basecfg(cfgfile);  // 分析(网络结构)配置文件
    printf("network cfg: %s\n", base);

    /** phase1: 根据cfgfile, 构建网络结构, 同时对网络结构参数进行初始化.*/
    // nets指向大小为1*sizeof(network)的内存空间
    network* nets = (network*)xcalloc(1, sizeof(network));
    // 解析网络配置文件, 为nets中的每个net进行初始化
    nets[0] = parse_network_cfg(cfgfile);
    // TODO: 加载预训练权重
    // 设置随机种子
    srand(time(0));
    network net = nets[0];
    int imgs = net.batch * net.subdivisions * 1;  // ngpus=1
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    /** phase2: 根据datacfg, 生成paths和其他相关的data参数*/
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk_data = option_find_int(options, "top", 5);    // top-k正确率
    char topk_buff[10];
    sprintf(topk_buff, "top%d", topk_data);  // 将"top%d"格式化输出到topk_buff中
    layer l = net.layers[net.n - 1];  // 获取网络的最后一层
    if (classes != l.outputs && (l.type == SOFTMAX || l.type == COST)) {
        printf("\n Error: num of filters = %d in the last conv-layer in cfg-file doesn't match to classes = %d in data-file \n",
               l.outputs, classes);
        getchar();
    }

    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    /** phase3: 多线程load数据, 这里通过buffer将最终的数据传递给args.d*/
    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.threads = 32;

    args.min = net.min_ratio*net.w;
    args.max = net.max_ratio*net.w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);  // 数据加载过程的核心操作.


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
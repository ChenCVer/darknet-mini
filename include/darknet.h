#ifndef DARKNET_API
#define DARKNET_API

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

/** Note: 该文件用来放各种定义的数据结构和定义的枚举类型*/

// ------------------------以下各种枚举------------------------------
// 1.网络中的各种层类型
typedef enum {
    CONVOLUTIONAL,
    MAXPOOL,
    SOFTMAX,
    NETWORK,
    AVGPOOL,
    ACTIVE,
    BATCHNORM,
    BLANK,
    COST
} LAYER_TYPE;

// 2. 各种激活函数类型
typedef enum{
    RELU,
    LEAKY,
} ACTIVATION;

// 3. 各种损失函数类型
typedef enum{
    SSE,
    MASKED,
    L1,
    SEG,
    SMOOTH,
    WGAN
} COST_TYPE;


// 4. 学习率调整策略类型
typedef enum {
    CONSTANT,
    STEP,
    EXP,
    POLY,
    STEPS,
    SIG,
    RANDOM,
    SGDR
} learning_rate_policy;

// 5. 各种数据类型: 分类, 分割, 检测
typedef enum {
    CLASSIFICATION_DATA,
    DETECTION_DATA,
    SEGMENTATION_DATA,
} data_type;

// 6. 图片格式
typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;


// ------------------------以下各种数据结构----------------------------
// 1. 用于更新网络参数时所用到
typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

typedef struct network network;
typedef struct layer layer;

// 2. 网络中的每一层的层定义
typedef struct layer{
    LAYER_TYPE type;        // 网络层类型
    ACTIVATION activation;  // 激活层类型
    COST_TYPE cost_type;    // 损失函数类型
    void (*forward)   (struct layer, struct network);  // 函数指针, 多态实现, forward操作
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    int batch_normalize;  // 是否进行BN,如果进行BN,则值为1
    int shortcut;  // 残差链接
    int batch;     // batchsize
    int forced;
    int flipped;
    int inputs;   // 一张输入图片所含的元素个数(一般在各网络层构建函数中赋值, 比如make_connected_layer()),
    // 第一层的值等于l.h*l.w*l.c,之后的每一层都是由上一层的输出自动推算得到的(参见parse_network_cfg(),
    // 在构建每一层后,会更新params.inputs为上一层的l.outputs).
    int outputs;  // 该层对应一张输入图片的输出元素个数(一般在各网络层构建函数中赋值,比如make_connected_layer())
    // 对于一些网络,可由输入图片的尺寸及相关参数计算出,比如卷积层,可以通过输入尺寸以及步长、核大小计算出；
    // 对于另一些尺寸,则需要通过网络配置文件指定,如未指定,取默认值1,比如全连接层(见parse_connected()函数)
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu;

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;
    size_t workspace_size;

}layer;

// 3. 网络结构定义
typedef struct network{
    int n;          // 网络的层数, 调用make_network(int n)时赋值
    int batch;      //一批训练中的图片参数, 和subdivsions参数相关
    size_t *seen;   //目前已经读入的图片张数(网络已经处理的图片张数)
    int *t;
    float epoch;   //到目前为止训练了整个数据集的次数
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;  // 训练过程中是否更改输入图像分辨率

    int gpu_index;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

} network;

// ---------------------------读取配置文件时相关数据结构
// 4. 解析.cfg文件时用到
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

// 5. 解析.cfg文件时用到
typedef struct list{
    int size;
    node *front;
    node *back;
} list;

// 5. 解析.cfg文件会用到
typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

// 6. 解析.cfg文件会用到
typedef struct{
    char *type;
    list *options;
}section;

// 7. parse网络层时用到.
typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;


// 数据加载, 存放相关数据结构

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;

} load_args;

#endif
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "network.h"

typedef layer convolutional_layer;

int convolutional_out_height(convolutional_layer l);
int convolutional_out_width(convolutional_layer l);
void add_bias(float *output, float *biases, int batch, int n, int size);
void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void forward_convolutional_layer(const convolutional_layer layer, network net);
void backward_convolutional_layer(convolutional_layer layer, network net);
void update_convolutional_layer(convolutional_layer layer, update_args a);

#endif

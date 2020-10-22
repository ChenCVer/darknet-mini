#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "network.h"
#include <float.h>

typedef layer maxpool_layer;


maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);
#endif

#endif


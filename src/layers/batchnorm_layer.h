#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c);
void forward_batchnorm_layer(layer l, network net);
void backward_batchnorm_layer(layer l, network net);

#endif

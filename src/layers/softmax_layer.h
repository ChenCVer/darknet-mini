#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "network.h"

typedef layer softmax_layer;


softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);

#endif

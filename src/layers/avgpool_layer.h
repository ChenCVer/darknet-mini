#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "network.h"

typedef layer avgpool_layer;

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void forward_avgpool_layer(const avgpool_layer l, network net);
void backward_avgpool_layer(const avgpool_layer l, network net);

#endif
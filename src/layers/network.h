#ifndef NETWORK_H
#define NETWORK_H

#include "darknet.h"
#include <assert.h>
#include <math.h>
#include "../utils/utils.h"
#include "../utils/blas.h"

void forward_network(network *netp);
void backward_network(network *netp);
void update_network(network *netp);
void calc_network_cost(network *netp);
float train_network_datum(network *net);
size_t get_current_batch(network *net);
float get_current_rate(network *net);
network *make_network(int n);
layer get_network_output_layer(network *net);
float train_network(network *net, data d);
void free_network(network *net);

#endif

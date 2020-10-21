#ifndef NETWORK_H
#define NETWORK_H

#include "darknet.h"


network *make_network(int n);
layer get_network_output_layer(network *net);

#endif

#ifndef PARSE_H
#define PARSE_H

#include "darknet.h"
#include "../layers/network.h"

void save_weights(network *net, char *filename);
network parse_network_cfg(char *filename);
list *read_cfg(char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void save_convolutional_weights(layer l, FILE *fp);
void save_batchnorm_weights(layer l, FILE *fp);

#endif

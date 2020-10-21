#ifndef PARSE_H
#define PARSE_H

#include "darknet.h"
#include "../layers/network.h"

network parse_network_cfg(char *filename);
list *read_cfg(char *filename);

#endif

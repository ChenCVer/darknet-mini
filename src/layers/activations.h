#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "darknet.h"

ACTIVATION get_activation(char *s);

static inline float relu_activate(float x){return x*(x>0);}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}

void activate_array(float *x, const int n, const ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);

static inline float relu_gradient(float x){return (x>0);}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}

#endif
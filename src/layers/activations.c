#include "activations.h"


ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "leaky")==0) return LEAKY;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}


float activate(float x, ACTIVATION a)
{
    switch(a){
        case RELU:
            return relu_activate(x);
        case LEAKY:
            return leaky_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case RELU:
            return relu_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}
#ifndef DATA_H
#define DATA_H

#include "darknet.h"
#include <pthread.h>

char **get_labels(char *filename);
list *get_paths(char *filename);
void fill_truth(char *path, char **labels, int k, float *truth);
pthread_t load_data(load_args args);
void *load_threads(void *ptr);
pthread_t load_data_in_thread(load_args args);
void *load_thread(void *ptr);

#endif

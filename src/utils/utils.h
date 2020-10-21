#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>


void calloc_error();
char *copy_string(char *s);
void *xcalloc(size_t nmemb, size_t size);
void file_error(char *s);
void malloc_error();
char *fgetl(FILE *fp);
void error(const char *s);
void strip(char *s);

#endif
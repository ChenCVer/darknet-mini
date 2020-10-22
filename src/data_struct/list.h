#ifndef LIST_H
#define LIST_H
#include "darknet.h"


list *make_list();
void free_list(list *l);
void option_insert(list *l, char *key, char *val);
int read_option(char *s, list *options);
int list_find(list *l, void *val);
void list_insert(list *, void *);
void free_list_contents(list *l);
int option_find_int(list *l, char *key, int def);
char *option_find(list *l, char *key);
float option_find_float(list *l, char *key, float def);
int option_find_int_quiet(list *l, char *key, int def);
float option_find_float_quiet(list *l, char *key, float def);
char *option_find_str(list *l, char *key, char *def);
void option_unused(list *l);
list *read_cfg(char *filename);
list *read_data_cfg(char *filename);
void **list_to_array(list *l);
#endif

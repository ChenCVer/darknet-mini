#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

void calloc_error()
{
    fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}


void *xcalloc(size_t nmemb, size_t size) {
    // malloc()和calloc()函数区别: https://www.cnblogs.com/stevenwuzheng/p/5484986.html
    void *ptr=calloc(nmemb,size);
    if(!ptr) {
        calloc_error();
    }
    // memset用来对一段内存空间全部设置为某个字符, 一般用在对定义的字符串进行初始化为‘ ’或‘/0’
    // memset可以方便的清空一个结构类型的变量或数组。
    memset(ptr, 0, nmemb * size);
    return ptr;
}

char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}

char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
    c = copy_string(c);
    next = strchr(c, '.');
    if (next) *next = 0;
    return c;
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}
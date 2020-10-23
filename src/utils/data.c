#include "data.h"
#include "../data_struct/list.h"
#include "utils.h"
#include "../data_struct/matrix.h"
#include "../data_struct/image.h"


pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        //if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle,
                                float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);  // load_data. if opencv is compiled, ther use opencv
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);  // 数据增强

        // debug:
        // show_image(im, "orig", 100);
        // show_image(crop, "crop", 100);
         // cvWaitKey(0);

        //grayscale_image_3c(crop);
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
            //printf("%s %s %d\n", path, labels[i], i);
        }
    }
    if(count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}

matrix load_labels_paths(char **paths, int n, char **labels, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
    }
    return y;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k,
                       int min, int max, int size, float angle, float aspect, float hue,
                       float saturation, float exposure, int center)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.w=size;
    d.h=size;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
    d.y = load_labels_paths(paths, n, labels, k);
    if(m) free(paths);
    return d;
}

void *load_thread(void *ptr)
{
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes,  a.min, a.max, a.size,
                                 a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
    }
    /*
    else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter,
                                   a.hue, a.saturation, a.exposure);
    } else if (a.type == SEGMENTATION_DATA){
        *a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect,
                             a.hue, a.saturation, a.exposure, a.scale);
    }*/

    free(ptr);

    return 0;
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.w = d1.w;
    d.h = d1.h;
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);

    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));

    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }

    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }

    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;

    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }

    free(buffers);
    free(threads);

    return 0;
}


pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}
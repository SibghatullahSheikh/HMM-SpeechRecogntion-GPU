#ifndef UTILS_H
#define UTILS_H 


#ifdef __cplusplus
extern "C"{
#endif


unsigned int getLineNumber(char *fileName);

void read_frames(char *file, float **frames, unsigned int col, unsigned int row);

void init_2d_f(float **frames, unsigned int row, unsigned int col, float val);

void init_1d_f(float *array, unsigned int N, float val);

void check_3d_f(float ***array, unsigned int row, unsigned int col, unsigned int N);

void check_2d_f(float **array, unsigned int row, unsigned int col);

void check_1d_f(float *array, unsigned int len);

void rand_1d_f(float *a, unsigned int N);

void rand_2d_f(float **a, unsigned int row, unsigned int col);

void randperm_1d(int *a, unsigned int N);

void transpose(float **in, float **out, unsigned int row, unsigned int col);

void symmetrize_f(float **in, unsigned int N);

void normalise_2d_f(float **x, unsigned int row, unsigned int col);

void normalise_1d_f(float *x, unsigned int len);

void mk_stochastic_2d_f(float **x, unsigned int row, unsigned int col, float **out);

float norm_square(float **U, unsigned int j, unsigned int rows);



#ifdef __cplusplus
}
#endif

#endif

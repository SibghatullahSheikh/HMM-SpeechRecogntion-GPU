#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C"{
#endif

unsigned int getLineNumber(char *fileName);

void read_frames(char *file, float **frames, unsigned int col, unsigned int row);

void init_2d_f(float **frames, unsigned int row, unsigned int col, float val);

void init_1d_f(float *array, unsigned int N, float val);

void buffer(float *sound, float**frames, unsigned int N, unsigned int framesize, unsigned int overlap, unsigned int frameNum);

void hamming_f(float *window, unsigned int N);




#ifdef __cplusplus
}
#endif

#endif

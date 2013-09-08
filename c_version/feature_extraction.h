#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef struct{
	float real,imag;
} COMPLEX_F;


unsigned int getLineNumber(char *fileName);

void read_frames(char *file, float **frames, unsigned int col, unsigned int row);

void init_2d_f(float **frames, unsigned int row, unsigned int col, float val);

void init_1d_f(float *array, unsigned int N, float val);

void buffer(float *sound, float**frames, unsigned int N, unsigned int framesize, unsigned int overlap, unsigned int frameNum);

void hamming_f(float *window, unsigned int N);

void FFT_F(float *x, unsigned int NFFT, COMPLEX_F *X);

void BubbleSort(float *arr, unsigned int len);

unsigned int *findpeaks(float *x, unsigned int N, unsigned int *peakNumber);

void FeatureExtraction_freqpeaks(float **frames, float **framefrequencies, unsigned int featureDim, unsigned int framesize, unsigned int frameNum, float *window, unsigned int NFFT);




#ifdef __cplusplus
}
#endif

#endif

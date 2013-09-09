#ifndef TRAIN_H
#define TRAIN_H

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C"{
#endif

void Train(float **observations, unsigned int featureDim, unsigned int frameNum, unsigned int hiddenstates);

void transpose(float **in, float **out, unsigned int row, unsigned int col);

void cov(float **input, unsigned int R, unsigned int C, float **result);

void chol(float **sigma, unsigned int row, unsigned int col, float **C);

void mvnpdf(double **B, float **obs, float **mean, float ***cov, unsigned int N, unsigned int T, unsigned int D);

void LU_pivot(float **R, float *P, unsigned int n);

void LUdcmp(float **sigma, float *LowerTri, float *UpperTri, float *permM, unsigned int n);

void pinv_main(float **R, float **R_pinv, unsigned int D);

void svd(float **input, float **U, float **S, float **V, unsigned int n);

float norm_square(float **U, unsigned int j, unsigned int rows); 

int sign_d(double x);

void pinv(float **U, float **S, float **V, unsigned int n, float **X);

#ifdef __cplusplus
}
#endif

#endif

#ifndef TRAIN_H
#define TRAIN_H

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef struct
{
	char *name;
	float *prior; // Nx1
	float **A; // NxN
	float **mu; //DxN
	float ***sigma; // DxDxN
	unsigned int N; // hidden states
	unsigned int D; // feature dimension
	unsigned int T; // frames
} HMM;


void TRAIN(float **observations, unsigned int featureDim, unsigned int frameNum, unsigned int hiddenstates);

void train_sample(float **observations, float **observations_t, float*prior, float **A, float ***Sigma,float**mu, unsigned int N, unsigned int D, unsigned int T, unsigned int loopID);

void transpose(float **in, float **out, unsigned int row, unsigned int col);

void cov(float **input, unsigned int R, unsigned int C, float **result);

void cholcov(float **sigma, unsigned int row, unsigned int col, float **C, unsigned int loopID, unsigned int state);

void chol(float **sigma, unsigned int n, float **T, int *perr);

//void mvnpdf(float **B, float **obs, float **mean, float ***cov, unsigned int N, unsigned int T, unsigned int D, unsigned int loopID);
void mvnpdf(float **B, float **observations_t, float **mu_t, float ***Sigma, unsigned int N, unsigned int T, unsigned int D, unsigned int loopID);

void LU_pivot(float **R, float *P, unsigned int n);

void LUdcmp(float **sigma, float *LowerTri, float *UpperTri, float *permM, unsigned int n);

void pinv_main(float **R, float **R_pinv, unsigned int D);

void svd(float **input, float **U, float **S, float **V, unsigned int n);

float norm_square(float **U, unsigned int j, unsigned int rows); 

int sign_d(double x);

void pinv(float **U, float **S, float **V, unsigned int n, float **X);

void Forward(float **B, float **A, float *prior, float **alpha, unsigned int N, unsigned int T, float *likelihood);

void Backward(float **A, float **B, float **beta, unsigned int N, unsigned int T);

void EM(float **observations, float *prior, float **A, float **mu, float **B, float ***Sigma, float **alpha, float **beta, unsigned int D, unsigned int N, unsigned int T);

void normalise_2d_f(float **x, unsigned int row, unsigned int col);

void normalise_1d_f(float *x, unsigned int len);

void mk_stochastic_2d_f(float **x, unsigned int row, unsigned int col, float **out);

void jacobi_eig(float **matrix, float **U, float **D, unsigned int n);

// void sort_eig(float **U, float **D, unsigned int n);

void PowerMethod(float **matrix, float **U, float **D, unsigned int n, unsigned int loopID, unsigned int state);

void power_method (unsigned int n, float **a, float *eigvec, int iterMax, float TOL, float *lambda, int *iter_num);

float r8vec_norm_l2 (unsigned int n, float *y);

void r8mat_mv(unsigned int m, unsigned int n, float **a, float *x, float *y);

float r8vec_dot(unsigned int n, float *a1, float *a2 );


#ifdef __cplusplus
}
#endif

#endif

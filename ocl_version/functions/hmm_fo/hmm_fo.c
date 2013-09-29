#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include <cuda.h>
//#include <cutil.h>
//#include "/usr/local/cuda/include/cublas.h"

#include "hmm.h"

extern "C" {         
#include "hmm_fo.h"
}

#define EXIT_ERROR     1.0f  

enum {
	MAX_THREADS_PER_BLOCK = 256
};



//  Kernels
 

/* Initialize alpha variables */
__global__ void init_alpha_dev( float *b_d,
		float *pi_d,
		int nstates,
		float *alpha_d,
		float *ones_d,
		int obs_t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nstates) {
		alpha_d[idx] = pi_d[idx] * b_d[(obs_t * nstates) + idx];
		ones_d[idx] = 1.0f;
	}
}







/* Calculate alpha variables */
__global__ void calc_alpha_dev( int nstates,
		float *alpha_t_d,
		float *b_d,
		int obs_t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nstates) {
		alpha_t_d[idx] = alpha_t_d[idx] * b_d[(obs_t * nstates) + idx];
	}
}

/* Scale alpha values */
__global__ void scale_alpha_dev( int nstates, float *alpha_t_d, float scale)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nstates) {
		alpha_t_d[idx] = alpha_t_d[idx] / scale;
	}
}



float run_hmm_fo(Hmm *in_hmm, Obs *in_obs)
{
	/* Host-side variables */
	int size;
	float *scale;
	float *a = in_hmm->a;
	float *b = in_hmm->b;
	float *pi = in_hmm->pi;
	int nsymbols = in_hmm->nsymbols;
	int nstates = in_hmm->nstates;
	int *obs = in_obs->data;
	int length = in_obs->length;
	int threads_per_block;
	int nblocks;
	int t;
	float log_lik;

	/* Device-side variables */
	float *a_d;
	float *b_d;
	float *pi_d;
	float *obs_d;
	float *alpha_d;             /* All alpha variables */
	float *alpha_t_d;           /* nstate alpha values at time t */
	float *ones_d;


	/* Allocate host memory */
	scale = (float *) malloc(sizeof(float) * length);
	if (scale == 0) {
		fprintf (stderr, "ERROR: Host memory allocation error (scale)\n");
		return EXIT_ERROR;
	}

	/* Allocate device memory */
	size = sizeof(float) * nstates * nstates;
	CUDA_SAFE_CALL( cudaMalloc((void**)&a_d, size) );

	CUDA_SAFE_CALL( cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice) );
	size = sizeof(float) * nstates * nsymbols;

	CUDA_SAFE_CALL( cudaMalloc((void**)&b_d, size) );

	CUDA_SAFE_CALL( cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice) );
	size = sizeof(float) * nstates;
	CUDA_SAFE_CALL( cudaMalloc((void**)&pi_d, size) );
	CUDA_SAFE_CALL( cudaMemcpy(pi_d, pi, size, cudaMemcpyHostToDevice) );
	size = sizeof(float) * length;
	CUDA_SAFE_CALL( cudaMalloc((void**)&obs_d, size) );
	CUDA_SAFE_CALL( cudaMemcpy(obs_d, obs, size, cudaMemcpyHostToDevice) );
	size = sizeof(float) * nstates * length;
	CUDA_SAFE_CALL( cudaMalloc((void**)&alpha_d, size) );
	size = sizeof(float) * nstates;
	CUDA_SAFE_CALL( cudaMalloc((void**)&alpha_t_d, size) );
	size = sizeof(float) * nstates;
	CUDA_SAFE_CALL( cudaMalloc((void**)&ones_d, size) );

	/* Initialize alpha variables */
	threads_per_block = MAX_THREADS_PER_BLOCK;
	nblocks = (nstates + threads_per_block - 1) / threads_per_block;
	init_alpha_dev<<<nblocks, threads_per_block>>>( b_d,
			pi_d,
			nstates,
			alpha_d,
			ones_d,
			obs[0]);
	size = sizeof(float) * nstates;
	CUDA_SAFE_CALL( cudaMemcpy( alpha_t_d,
				alpha_d,
				size,
				cudaMemcpyDeviceToDevice) );





	sGetError();
	scale[0] = cublasSdot(nstates, alpha_t_d, 1, ones_d, 1);
	cublas_status = cublasGetError();

	if (cublas_status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "ERROR: Kernel execution error\n");
		return EXIT_ERROR;
	}

	/* Scale alpha values */
	scale_alpha_dev<<<nblocks, threads_per_block>>>(    nstates,
			alpha_t_d,
			scale[0]);

	/* Copy temporary alpha values back to alpha matrix */
	CUDA_SAFE_CALL( cudaMemcpy( alpha_d,
				alpha_t_d,
				size,
				cudaMemcpyDeviceToDevice) );

	/* Initialize log likelihood */
	log_lik = log10(scale[0]);

	/* Calculate the rest of the alpha variables */
	for (t = 1; t < length; t++) {

		/* Multiply transposed A matrix by alpha(t-1) */
		/* Note: the matrix is auto-transposed by cublas reading column major */
		cublasSgemv( 'N', nstates, nstates, 1.0f, a_d, nstates, alpha_t_d, 1, 0,
				alpha_t_d, 1);
		cublas_status = cublasGetError();
		if (cublas_status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "ERROR: Kernel execution error\n");
			return EXIT_ERROR;
		}

		/* Calculate alpha(t) */
		calc_alpha_dev<<<nblocks, threads_per_block>>>( nstates,
				alpha_t_d,
				b_d,
				obs[t]);

		/* Sum alpha values to get scaling factor */
		scale[t] = cublasSdot(nstates, alpha_t_d, 1, ones_d, 1);
		cublas_status = cublasGetError();
		if (cublas_status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "ERROR: Kernel execution error\n");
			return EXIT_ERROR;
		}

		/* Scale alpha values */
		scale_alpha_dev<<<nblocks, threads_per_block>>>(    nstates,
				alpha_t_d,
				scale[t]);

		/* Copy temporary alpha values back to alpha matrix */
		CUDA_SAFE_CALL( cudaMemcpy( alpha_d + (t * nstates),
					alpha_t_d,
					size,
					cudaMemcpyDeviceToDevice) );

		/* Update log likelihood */
		log_lik += log10(scale[t]);
	}

	/* Free device memory */
	CUDA_SAFE_CALL( cudaFree(a_d) );
	CUDA_SAFE_CALL( cudaFree(b_d) );
	CUDA_SAFE_CALL( cudaFree(pi_d) );
	CUDA_SAFE_CALL( cudaFree(obs_d) );
	CUDA_SAFE_CALL( cudaFree(alpha_d) );

	return log_lik;
}



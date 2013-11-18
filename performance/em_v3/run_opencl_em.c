#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "hmm.h"
#include "speech_lib.h"
#include "run_opencl_em.h"
#include "../../utils/ocl_utils.h"

#define TILE 16

void run_opencl_em(HMM *word)
{

	puts("\n=>GPU");

	int i,j,k;
	int N = word->nstates;
	int T = word->len;
	int D = word->features;

	float *B = word->b; // N x T

	float *A = word->a; // N x N
	float *alpha = word->alpha; // N x T
	float *beta  = word->beta;  // N x T
	float *observations = word->obs; // D x T

	// we need to transpose the B/alpha/beta/observations
	// in order to increase the memory coalescing
	// this will be avoided when we do data arrangement early

	float *B_h = (float*)malloc(sizeof(float)*T*N);
	transpose(B,B_h,N,T);

	float *alpha_h = (float*)malloc(sizeof(float)*T*N);
	transpose(alpha,alpha_h,N,T);

	float *beta_h = (float*)malloc(sizeof(float)*T*N);
	transpose(beta,beta_h,N,T);

	float *observations_t = (float*)malloc(sizeof(float)*T*D); // TxD
	transpose(observations,observations_t,D,T);




	float *xisum = (float*)malloc(sizeof(float)*N*N);
	memset(xisum,0,sizeof(float)*N*N);

#ifdef DEBUG
	float *gamma_h = (float*)malloc(sizeof(float)*T*N); // T x N
#endif


	uint tPos, tPos_pre;	

	int blks = (N+255)/256;

	int bx = (N+15)/16;
	int by = (N+15)/16;
	int bxy = bx * by; // xisum_mid


	// gpu timer
	cl_ulong gstart, gend;
	double gpuTime;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_int err;

	int numK = 12;
	int numE = 1;

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*numK);

	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*numE);    


	// read kernel file
	char *fileName = "em_kernel.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';

	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);


	// locate kernels
	kernel[0] = clCreateKernel(program, "vector_mul", &err);
	OCL_CHECK(err);

	kernel[1] = clCreateKernel(program, "state_dev", &err);
	OCL_CHECK(err);

	kernel[2] = clCreateKernel(program, "normalise_1d_gamma", &err);
	OCL_CHECK(err);

	kernel[3] = clCreateKernel(program, "normalise_xisum_s1", &err);
	OCL_CHECK(err);

	kernel[4] = clCreateKernel(program, "normalise_xisum_s2", &err);
	OCL_CHECK(err);

	kernel[5] = clCreateKernel(program, "normalise_xisum_s3", &err);
	OCL_CHECK(err);

	kernel[6] = clCreateKernel(program, "xisum_addon", &err);
	OCL_CHECK(err);

	kernel[7] = clCreateKernel(program, "gammaT", &err);
	OCL_CHECK(err);

	kernel[8] = clCreateKernel(program, "mk_stochastic", &err);
	OCL_CHECK(err);

	kernel[9] = clCreateKernel(program, "gamma_state_sum_dev", &err);
	OCL_CHECK(err);

	kernel[10] = clCreateKernel(program, "copy_currentgamma", &err);
	OCL_CHECK(err);

	kernel[11] = clCreateKernel(program, "gamma_obs_dev", &err);
	OCL_CHECK(err);

	// device data
	cl_mem B_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);

	cl_mem alpha_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);

	cl_mem beta_d 
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);

	cl_mem observations_d 
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*D*T,  NULL, NULL); // D x T

	cl_mem observations_t_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*T*D,  NULL, NULL); // T x D

	cl_mem A_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL); // N x N

	cl_mem beta_B_d 
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,    NULL, NULL);

	cl_mem gamma_norm_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,  NULL, NULL); // N 

	cl_mem gamma_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL); // N 

	cl_mem xisum_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL); // N 

	cl_mem xisum_norm_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL); // N 

	cl_mem xisum_mid_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*bxy,  NULL, NULL); // bxy 

	cl_mem xisum_fin_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),  NULL, NULL); //  1 

//	cl_mem Z1_d
//	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,  NULL, NULL); //  mk_stochastic 

	cl_mem gamma_state_sum_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,  NULL, NULL); // N 


	cl_mem expect_mu_d
	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*D,  NULL, NULL); // NxD

	cl_mem gamma_tmp_d
	= clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(float)*T,  NULL, NULL); // T 

	cl_mem currentgamma_d
	= clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*T,  NULL, NULL); // T 

	cl_mem gamma_obs_d
	= clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*T*D,  NULL, NULL); // DxT as observations 


	// copy from host to device
	err = clEnqueueWriteBuffer(queue, B_d, 		CL_TRUE, 	0, sizeof(float)*N*T, 	B_h, 	0, NULL, &event[0]); 
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, beta_d, 	CL_TRUE, 	0, sizeof(float)*N*T, 	beta_h, 	0, NULL, NULL); 
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, A_d, 	    	CL_TRUE, 	0, sizeof(float)*N*N, 	A,	0, NULL, NULL); 
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, alpha_d, 	CL_TRUE, 	0, sizeof(float)*N*T, 	alpha_h, 	0, NULL, NULL); 
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, observations_d, 	CL_TRUE, 	0, sizeof(float)*D*T, 	observations,	0, NULL, NULL); 
	OCL_CHECK(err);

	// hint : transpose  in OpenCL
	err = clEnqueueWriteBuffer(queue, observations_t_d, 	CL_TRUE, 	0, sizeof(float)*T*D, 	observations_t,	0, NULL, NULL); 
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, xisum_d, 	CL_TRUE, 	0, sizeof(float)*N*N, 	xisum,	0, NULL, NULL); 
	OCL_CHECK(err);


	//------------------------- E ---------------------------------// 
	// kernel 1 : beta x B
	size_t l0 = 256;
	size_t g0 = N ;

	err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &beta_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &alpha_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &beta_B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[0], 4, sizeof(cl_mem), &gamma_norm_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 2:
	size_t l1[2];
	size_t g1[2];
	l1[0] = l1[1] =  16;
	g1[0] = g1[1] =  N;

	err = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &beta_B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &alpha_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), &xisum_norm_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[1], 4, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 3: normalize to produce gamma
	size_t l2 = 256;
	size_t g2 = 256;

	err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &gamma_norm_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &gamma_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[2], 2, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[2], 3, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 4: normalise to produce xisum
	// xisum_norm_d is NxN
	// step 1
	size_t l3[2];
	size_t g3[2];
	l3[0] = l3[1] =  16;
	g3[0] = g3[1] =  N;

	err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &xisum_norm_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	
	err = clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &xisum_mid_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[3], 2, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[3], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}



	// kernel 5: step 2
	// user 256 as tpb
	size_t l4 = 256;
	size_t g4 = 256; // bxy should be multiples of 256 
	uint iters = (bxy+255)/256;

	err = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &xisum_mid_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	
	err = clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &xisum_fin_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[4], 2, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[4], 3, sizeof(uint), &iters);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 6: step 3
	size_t l5[2];
	size_t g5[2];
	l5[0] = l5[1] =  16;
	g5[0] = g5[1] =  N;

	err = clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &xisum_norm_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[5], 1, sizeof(cl_mem), &xisum_fin_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[5], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}



	// kernel 7: xisum addon
	size_t l6[2];
	size_t g6[2];
	l6[0] = l6[1] =  16;
	g6[0] = g6[1] =  N;

	err = clSetKernelArg(kernel[6], 0, sizeof(cl_mem), &xisum_norm_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[6], 1, sizeof(cl_mem), &xisum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[6], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	// kernel 8
	size_t l7 = 256;
	size_t g7 = N; 

	err = clSetKernelArg(kernel[7], 0, sizeof(cl_mem), &beta_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	
	err = clSetKernelArg(kernel[7], 1, sizeof(cl_mem), &alpha_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[7], 2, sizeof(cl_mem), &gamma_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	int TimeLine;
	for(TimeLine = 0 ; TimeLine <= T-2; ++TimeLine)
	{
		tPos_pre = TimeLine * N;
		tPos = tPos_pre + N;

		// k1: beta x B	  and 		alpha .* beta
		err = clSetKernelArg(kernel[0], 5, sizeof(uint), &tPos);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clSetKernelArg(kernel[0], 6, sizeof(uint), &tPos_pre);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, &g0, &l0, 0, NULL, NULL);
		OCL_CHECK(err);

		// k2 : A .* (alpha * beta_B)      
		err = clSetKernelArg(kernel[1], 5, sizeof(uint), &tPos_pre);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[1], 2, NULL, g1, l1, 0, NULL, NULL);
		OCL_CHECK(err);

		// k3 : normalise (alpha.*beta)	and update gamma
		err = clSetKernelArg(kernel[2], 4, sizeof(uint), &tPos_pre);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[2], 1, NULL, &g2, &l2, 0, NULL, NULL);
		OCL_CHECK(err);


		// k4-7: normalise xi_sum
		// generate intermediate data for reduction
		err = clEnqueueNDRangeKernel(queue, kernel[3], 2, NULL, g3, l3, 0, NULL, NULL);
		OCL_CHECK(err);

		// sum up
		err = clEnqueueNDRangeKernel(queue, kernel[4], 1, NULL, &g4, &l4, 0, NULL, NULL);
		OCL_CHECK(err);
	
		// scale
		err = clEnqueueNDRangeKernel(queue, kernel[5], 2, NULL, g5, l5, 0, NULL, NULL);
		OCL_CHECK(err);

		// add result to xi_sum
		err = clEnqueueNDRangeKernel(queue, kernel[6], 2, NULL, g6, l6, 0, NULL, NULL);
		OCL_CHECK(err);
	}

	// TimeLine = T-1 , at tPos
	err = clSetKernelArg(kernel[7], 3, sizeof(uint), &tPos);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[7], 1, NULL, &g7, &l7, 0, NULL, NULL);
	OCL_CHECK(err);

#ifdef DEBUG
	clFinish(queue);
	clEnqueueReadBuffer(queue, gamma_d, CL_TRUE, 0, sizeof(float)*T*N, gamma_h, 0, NULL , NULL);
	printf("\ngamma = \n");
	for(i=0;i<T;++i){
		for(j=0;j<N;++j){
			printf("(%d,%d) = %.4e\n",i,j,gamma_h[i*N+j]);
		}	
		printf("\n");
	}

	clEnqueueReadBuffer(queue, xisum_d, CL_TRUE, 0, sizeof(float)*N*N, xisum, 0, NULL , NULL);
	printf("\n\n xi_sum = \n");
	for(i=0;i<N;++i){
		for(j=0;j<N;++j){
			printf("(%d,%d) = %.4e\n",i,j,xisum[i*N+j]);
		}	
		printf("\n");
	}
#endif

	//----------------------------- M --------------------------------------// 

	// => expect_prior is the first row of gamma 
	// => expect_A is the mk_stochastic() on xi_sum (write kernel)
	// => expect_mu: it needs to be fed to mvnpdf() for each iteration
	// => expect_sigma: it needs to be fed to mvnpdf()


	float *expect_prior = (float*)malloc(sizeof(float)*N);
	clEnqueueReadBuffer(queue, gamma_d, CL_TRUE, 0, sizeof(float)*N, expect_prior, 0, NULL , NULL);

//	for(i=0;i<N;++i)
//	{
//		printf("%.4e\n",expect_prior[i]);
//	}

	// implement mk_stochastic() in OpenCL
	float *expect_A = (float*)malloc(sizeof(float)*N*N);
	size_t l8[2];
	size_t g8[2];

	l8[0] = 16;
	g8[0] = 16;
	
	l8[1] = 16;
	g8[1] = N;

	err = clSetKernelArg(kernel[8], 0, sizeof(cl_mem), &xisum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[8], 1, sizeof(int), &bx);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[8], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[8], 2, NULL, g8, l8, 0, NULL, NULL);
	OCL_CHECK(err);

	clEnqueueReadBuffer(queue, xisum_d, CL_TRUE, 0, sizeof(float)*N*N, expect_A, 0, NULL , NULL);

	
	//kernel 10 : gamma_state_sum
	size_t l9[2];
	size_t g9[2];

	l9[0] = 16;
	l9[1] = 16;

	g9[0] = N;
	g9[1] = 16;

	uint stride = TILE * N;

	err = clSetKernelArg(kernel[9], 0, sizeof(cl_mem), &gamma_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[9], 1, sizeof(cl_mem), &gamma_state_sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[9], 2, sizeof(int), &bx);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[9], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[9], 4, sizeof(int), &stride);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[9], 2, NULL, g9, l9, 0, NULL, NULL);
	OCL_CHECK(err);

	//-------------------------
	// Loop Through Each State
	//-------------------------

	// kernel 11:
	// copy a row of gamma out to gamma_tmp(writeonly)
	int stateNum;
	stateNum = 0;	

	size_t l10 = 256;
	size_t g10 = N;

	err = clSetKernelArg(kernel[10], 0, sizeof(cl_mem), &gamma_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[10], 1, sizeof(cl_mem), &gamma_tmp_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[10], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	
	err = clSetKernelArg(kernel[10], 3, sizeof(int), &stateNum);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[10], 1, NULL, &g10, &l10, 0, NULL, NULL);
	OCL_CHECK(err);

	// then copybuffer from gamma_tmp to currentgamma
	clEnqueueCopyBuffer(queue,gamma_tmp_d,currentgamma_d,0,0,sizeof(float)*T,0,NULL,NULL);

	// kenel 12: obs (DxT) * repmat(currentgamma, D) = gamma_obs 
	size_t l11[2];
	size_t g11[2];

	l11[0] = 16;
	l11[1] = 16;

	g11[0] = T;
	g11[1] = D;

	err = clSetKernelArg(kernel[11], 0, sizeof(cl_mem), &observations_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[11], 1, sizeof(cl_mem), &currentgamma_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[11], 2, sizeof(cl_mem), &gamma_obs_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clSetKernelArg(kernel[11], 3, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[11], 2, NULL, g11, l11, 0, NULL, NULL);
	OCL_CHECK(err);

	// kernel 13: calcuate expect_mu (N X D), write features D  to earch state(row)


	// kernel 14:  gamma_obs * observations_t (matrix multiplication), 
	// then /gamma_state_sum(s) (sclar)


	for(stateNum = 0; stateNum < N ; ++stateNum)
	{
	
	}

/*



	// NxN
	float *xi_sum	= (float*)malloc(sizeof(float)*N*N);
	init_2d_f(xi_sum,N,N,0.f);

	// NxT
	float *gamma= (float*)malloc(sizeof(float)*N*T);
	init_2d_f(gamma,N,T,0.f);


	// intermediate data array
	float *beta_B 		  = (float*)malloc(sizeof(float)*N);
	float *alpha_beta     = (float*)malloc(sizeof(float)*N); // Nx1
	//float *alpha_beta_B   = (float*)malloc(sizeof(float)*N*N); // NxN
	float *A_alpha_beta_B = (float*)malloc(sizeof(float)*N*N); // NxN

	float *gamma_obs 		= (float*)malloc(sizeof(float)*D*T); // DxT
	//float *gamma_obs_mul	= (float*)malloc(sizeof(float)*D*D); // DxD
	//float *exp_mu_mul		= (float*)malloc(sizeof(float)*D*D); // DxD
	float *gammaob_ob_t	    = (float*)malloc(sizeof(float)*D*D); // DxD

	// results
	float *exp_prior  = (float*)malloc(sizeof(float)*N); // Nx1
	float *exp_A	  = (float*)malloc(sizeof(float)*N*N);
	float *exp_mu	  = (float*)malloc(sizeof(float)*D*N);
	init_2d_f(exp_mu,D,N,0.f);
	float *exp_sigma;
	exp_sigma = (float*)malloc(sizeof(float*)*D*D*N); // row x col x height
	init_3d_f(exp_sigma,D,D,N,0.f);

	float *gamma_state_sum	= (float*)malloc(sizeof(float)*N); // Nx1




	kernel[1] = clCreateKernel(program, "A_alpha_beta_B_dev", &err);
	OCL_CHECK(err);
	kernel[2] = clCreateKernel(program, "gamma_obs_dev", &err);
	OCL_CHECK(err);
	kernel[3] = clCreateKernel(program, "exp_mu_dev", &err);
	OCL_CHECK(err);
	kernel[4] = clCreateKernel(program, "gammaob_ob_t_dev", &err);
	OCL_CHECK(err);
	kernel[5] = clCreateKernel(program, "exp_mu_mul_dev", &err);
	OCL_CHECK(err);


	cl_mem A_alpha_beta_B_d 		= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem A_d 				= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem alpha_beta_d			= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,    NULL, NULL);
	cl_mem gamma_obs_d			= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*D*T,  NULL, NULL); // D x T
	cl_mem gamma_d				= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL); // N x T

	cl_mem gamma_state_sum_d		= clCreateBuffer(context, CL_MEM_READ_ONLY,   sizeof(float)*N,    NULL, NULL); // N 
	cl_mem exp_mu_d				= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*D*N,  NULL, NULL); // D x N 
	cl_mem gammaob_ob_t_d			= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*D*D,  NULL, NULL); // D x D 
	//cl_mem exp_mu_mul_d			= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*D*D,  NULL, NULL); // D x D 

	// warm up() device
	float *dummy = (float*)malloc(sizeof(float));
	cl_mem dummy_d= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),  NULL, NULL);
	for(i=0;i<50;++i){
		err = clEnqueueWriteBuffer(queue, dummy_d,CL_TRUE,0,sizeof(float),dummy,0, NULL, NULL);
	}



	//------------------------------------------------
	//  Start OpenCL Execution 
	//------------------------------------------------
	int window;
	double tmp;


	for(window = 0; window < (T-1) ; ++window)
	{
	//	printf("window = %d\n", window);
		// kernel 1
		err = clSetKernelArg(kernel[0], 7, sizeof(int), &window);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	
		if( window == 0 && 0)
		{
			clFinish(queue);
			//clEnqueueReadBuffer(queue, beta_B_d, 	 CL_TRUE, 0, sizeof(float)*N, beta_B, 	  0, NULL , NULL);
			//check_1d_f(beta_B, N);	
			clEnqueueReadBuffer(queue, alpha_beta_d, CL_TRUE, 0, sizeof(float)*N, alpha_beta, 0, NULL , NULL);
			check_1d_f(alpha_beta, N);	

			return;
		}
	

		// kernel 2
		err  = clSetKernelArg(kernel[1], 6, sizeof(int), &window);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	
		err = clEnqueueNDRangeKernel(queue, kernel[1], 2, NULL, global_2, local_2, 0, NULL, NULL);
		OCL_CHECK(err);

		if( window == 0 && 0)
		{
			check_2d_f(A_alpha_beta_B, N, N);	
			return;
		}

		// copy back A_alpha_beta_B and alpha_beta
		clEnqueueReadBuffer(queue, A_alpha_beta_B_d, CL_TRUE, 0, sizeof(float)*N*N, A_alpha_beta_B, 0, NULL , NULL);
		clEnqueueReadBuffer(queue, alpha_beta_d, 	 CL_TRUE, 0, sizeof(float)*N,   alpha_beta, 	0, NULL , NULL);

		normalise_2d_f(A_alpha_beta_B,N,N);
		normalise_1d_f(alpha_beta,N); 

		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				xi_sum[j*N + k] += A_alpha_beta_B[j*N + k];
			}
		}

		for(j=0;j<N;++j){
			gamma[j*T + window]= alpha_beta[j];
		}

		// kernel 3 :
	}

	// update gamma for the last window (T -1)
	for(j=0;j<N;++j){
		alpha_beta[j] = alpha[j*T + T-1] * beta [j*T + T-1];
	}
	normalise_1d_f(alpha_beta,N);
	for(j=0;j<N;++j){
		gamma[j*T + T -1]= alpha_beta[j];
	}

	//check_2d_f(gamma,N,T);

	// save the expected prior
	for(i=0;i<N;++i){
		exp_prior[i] = gamma[i*T + 0];
	}

	// save the expected A
	mk_stochastic_2d_f(xi_sum,N,N,exp_A);

	for(i=0;i<N;++i)
	{
		tmp = 0.0;
		for(j=0;j<T;++j){
			tmp += gamma[i*T + j];
		}
		if(tmp == 0.0){
			tmp = 1.0;
		}
		gamma_state_sum[i]=(float)tmp;
	}	

	


	//---------------------------------------
	// maximization
	//---------------------------------------
	err = clEnqueueWriteBuffer(queue, gamma_d, 	CL_TRUE, 	0, sizeof(float)*N*T, 	gamma, 	0, NULL, NULL); 
	OCL_CHECK(err);
	//check_2d_f(gamma, N, T);

	// cache gamma state sum to constant memory
	err = clEnqueueWriteBuffer(queue, gamma_state_sum_d, 	CL_TRUE, 0, sizeof(float)*N, gamma_state_sum,0, NULL, NULL); 
	OCL_CHECK(err);


	// kernel 3 :  repeat gamma[T] D times
	//			   then dot multiply observations
	size_t local_3[2];
	size_t global_3[2];

	local_3[0] = 16;
	local_3[1] = 16;
	global_3[0] = ((D + 15)/16) * 16;
	global_3[1] = ((T + 15)/16) * 16;

	err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &gamma_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &observations_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &gamma_obs_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[2], 3, sizeof(int), &D);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[2], 4, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	//err  = clSetKernelArg(kernel[2], 5, sizeof(int), &k);
	//if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 4 :  
	size_t local_4 = 64;
	size_t global_4 = chunks*64;

	err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &gamma_obs_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &gamma_state_sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3], 2, sizeof(cl_mem), &exp_mu_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3], 3, sizeof(int), &D);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3], 4, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3], 5, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	//err = clSetKernelArg(kernel[3], 6, sizeof(int), &k);
	//if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 5 : 
	size_t local_5[2];
	size_t global_5[2];

	// TILE = 16
	local_5[0] = 16;
	local_5[1] = 16;
	global_5[0] = ((D + 15)/16) * 16;
	global_5[1] = ((D + 15)/16) * 16;

	err  = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &gamma_obs_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &observations_t_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 2, sizeof(cl_mem), &gammaob_ob_t_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 3, sizeof(cl_mem), &gamma_state_sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 4, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 5, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 6, sizeof(int), &D);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[4], 7, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	//err  = clSetKernelArg(kernel[4], 8, sizeof(int), &k);
	//if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 6 : 
	size_t local_6[2];
	size_t global_6[2];

	local_6[0] = 16;
	local_6[1] = 16;
	global_6[0] = ((D + 15)/16) * 16;
	global_6[1] = ((D + 15)/16) * 16;

	err  = clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &exp_mu_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[5], 1, sizeof(cl_mem), &gammaob_ob_t_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[5], 2, sizeof(int), &D);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[5], 3, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	//err  = clSetKernelArg(kernel[5], 4, sizeof(int), &k);
	//if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}



	for(k = 0; k < N; ++k)
	{

		// kernel 3:
		err  = clSetKernelArg(kernel[2], 5, sizeof(int), &k);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[2], 2, NULL, global_3, local_3, 0, NULL, NULL);
		OCL_CHECK(err);

		if( k == 0 && 0)
		{
			clFinish(queue);
			clEnqueueReadBuffer(queue, gamma_obs_d, CL_TRUE, 0, sizeof(float)*D*T, gamma_obs, 0, NULL , NULL);
			check_2d_f(gamma_obs, D,T);	
			return;
		}

		// kernel 4
		err = clSetKernelArg(kernel[3], 6, sizeof(int), &k);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_4, &local_4, 0, NULL, NULL);
		OCL_CHECK(err);

		if( k == 0 && 0){
			clFinish(queue);
			clEnqueueReadBuffer(queue, exp_mu_d, CL_TRUE, 0, sizeof(float)*D*N, exp_mu, 0, NULL , NULL);
			check_2d_f(exp_mu, D,N);	
			return;
		}

		//  kernel 5
		err  = clSetKernelArg(kernel[4], 8, sizeof(int), &k);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[4], 2, NULL, global_5, local_5, 0, NULL, NULL);
		OCL_CHECK(err);

		if( k == 0 && 0){
			clFinish(queue);
			clEnqueueReadBuffer(queue, gammaob_ob_t_d, CL_TRUE, 0, sizeof(float)*D*D, gammaob_ob_t, 0, NULL , NULL);
			check_2d_f(gammaob_ob_t, D,D);	
			return;
		}


		// kernel 6 : 
		err  = clSetKernelArg(kernel[5], 4, sizeof(int), &k);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[5], 2, NULL, global_6, local_6, 0, NULL, NULL);
		OCL_CHECK(err);

		if( k == 0 && 0){
			clFinish(queue);
			clEnqueueReadBuffer(queue, gammaob_ob_t_d, CL_TRUE, 0, sizeof(float)*D*D, gammaob_ob_t, 0, NULL , NULL);
			check_2d_f(gammaob_ob_t, D,D);	
			return;
		}

		clEnqueueReadBuffer(queue, gammaob_ob_t_d, CL_TRUE, 0, sizeof(float)*D*D, gammaob_ob_t, 0, NULL , NULL);

		// symmetrize()	
		symmetrize_f(gammaob_ob_t,D);

		// save to exp_sigma
		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				exp_sigma[(i*D+j)*D + k] = gammaob_ob_t[i*D+j];
			}
		}


	}

	// % Ninja trick to ensure positive semidefiniteness
	for(k=0;k<N;++k)
	{
		for(i=0;i<D;++i)
		{
			for(j=0;j<D;++j)
			{
				if(i==j)	exp_sigma[(i*D+j)*D + k] += 0.01;
			}
		}
	}


	clFinish(queue);
	clEnqueueReadBuffer(queue, exp_mu_d, CL_TRUE, 0, sizeof(float)*D*N, exp_mu, 0, NULL , &event[1]);

	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	printf("oclTime = %lf (s)\n", gpuTime );




	clReleaseMemObject(A_alpha_beta_B_d);
	clReleaseMemObject(exp_mu_d);
	clReleaseMemObject(gammaob_ob_t_d);
	//clReleaseMemObject(exp_mu_mul_d);
	
	clReleaseMemObject(dummy_d);




	free(xi_sum);
	free(gamma);
	free(beta_B);
	free(alpha_beta);
	free(A_alpha_beta_B);

	free(gamma_obs);
	free(gammaob_ob_t);
	//free(exp_mu_mul);

//
	free(exp_prior);
	free(exp_A);
	free(exp_mu);
	free(exp_sigma);
	free(gamma_state_sum);
	free(dummy);
*/


	// opencl free
	clReleaseMemObject(B_d);
	clReleaseMemObject(alpha_d);
	clReleaseMemObject(beta_d);
	clReleaseMemObject(observations_d);
	clReleaseMemObject(observations_t_d);
	clReleaseMemObject(A_d);
	clReleaseMemObject(beta_B_d);
	clReleaseMemObject(gamma_norm_d);
	clReleaseMemObject(gamma_d);

	clReleaseMemObject(xisum_d);
	clReleaseMemObject(xisum_norm_d);
	clReleaseMemObject(xisum_mid_d);
	clReleaseMemObject(xisum_fin_d);

	clReleaseMemObject(gamma_state_sum_d);
	clReleaseMemObject(expect_mu_d);
	clReleaseMemObject(gamma_tmp_d);
	clReleaseMemObject(currentgamma_d);

	clReleaseMemObject(gamma_obs_d);

	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);


	for(i=0;i<numK;++i){
		err = clReleaseKernel(kernel[i]);
		OCL_CHECK(err);
	}

	for(j=0;j<numE;++j){
		err = clReleaseEvent(event[j]);
		OCL_CHECK(err);
	}


	// host
	free(kernelSource);
	free(B_h);
	free(alpha_h);
	free(beta_h);
	free(observations_t);
	free(xisum);

	free(expect_prior);
	free(expect_A);

#ifdef DEBUG
	free(gamma_h);
#endif
	return;
}





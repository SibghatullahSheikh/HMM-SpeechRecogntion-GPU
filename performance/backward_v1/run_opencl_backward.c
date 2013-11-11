#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "run_opencl_backward.h"
#include "hmm.h"
#include "../../utils/ocl_utils.h"

void run_opencl_backward(HMM *word)
{

	puts("\n=>GPU");

	int i;
	int N = word->nstates;
	int T = word->len;
	float *B = word->b;
	float *A = word->a;

	// gpu timer
	cl_ulong gstart, gend;
	double gpuTime;

	float *beta; // T x N
	beta = (float*)malloc(sizeof(float)*N*T);


	uint startPos, startPos_pre;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------

	int blks;
	blks = (N+255)/256;

	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	int numK = 4;
	int numE = 2;

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*numK);
	
	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*numE);    

	// read kernel file
	char *fileName = "backward_kernel.cl";
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


	kernel[0] = clCreateKernel(program, "mat_vec", &err);
	OCL_CHECK(err);

	kernel[1] = clCreateKernel(program, "scale_beta", &err);
	OCL_CHECK(err);

	kernel[2] = clCreateKernel(program, "beta_dev", &err);
	OCL_CHECK(err);

	kernel[3] = clCreateKernel(program, "normalize_beta", &err);
	OCL_CHECK(err);


	// allocate memory on device 
	cl_mem A_d      	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem B_d      	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);
	cl_mem beta_d		= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);

	// warm up() device
	float *dummy = (float*)malloc(sizeof(float));
	cl_mem dummy_d= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),  NULL, NULL);
	for(i=0;i<50;++i){
		err = clEnqueueWriteBuffer(queue, dummy_d, 		CL_TRUE, 	0, sizeof(float), 	dummy, 		0, NULL, NULL);
	}



	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 		CL_TRUE, 	0, sizeof(float)*N*N, 	A, 		0, NULL, &event[0]);
	OCL_CHECK(err);
	err = clEnqueueWriteBuffer(queue, B_d, 		CL_TRUE, 	0, sizeof(float)*N*T, 	B, 		0, NULL, NULL); 
	OCL_CHECK(err);



	//----------------------------------------------  A * ( 1 . * B) ----------------------------------------//
	//  1st kernel: beta * B 

	startPos = (T - 2) * N;
	startPos_pre = startPos + N;

	size_t local_0[2]; 
	size_t global_0[2]; 

	local_0[0] = 16;
	local_0[1] = 16;

	global_0[0] = 16;
	global_0[1] = N;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &beta_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 4, sizeof(uint), &startPos);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 5, sizeof(uint), &startPos_pre);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

 	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, global_0, local_0, 0, NULL, NULL );
         OCL_CHECK(err);

	//----------------------------------------------  scale_beta ----------------------------------------//
	 size_t local_1;
	 size_t global_1;
	 local_1  =  256;
	 global_1 =  256;

	 err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &beta_d);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[1], 1, sizeof(uint), &startPos);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[1], 2, sizeof(uint), &startPos_pre);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[1], 3, sizeof(float)*257, NULL);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[1], 4, sizeof(int), &N);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[1], 5, sizeof(int), &blks);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	 err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &global_1, &local_1, 0, NULL, NULL );
	 OCL_CHECK(err);



	//----------------------------------------------  dev beta----------------------------------------//

	 size_t local_2[2];
	 size_t global_2[2];
	 local_2[0]= 16;
	 local_2[1]= 16;
	 global_2[0] =  16;
	 global_2[1] =  N;

	 err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &beta_d);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &B_d);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &A_d);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err |= clSetKernelArg(kernel[2], 3, sizeof(int), &N);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	//----------------------------------------------  normalize beta----------------------------------------//
	 size_t local_3;
	 size_t global_3;
	 local_3  =  256;
	 global_3 =  256;

	 err  = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &beta_d);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 //err = clSetKernelArg(kernel[3], 1, sizeof(uint), &startPos);
	 //if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 //err = clSetKernelArg(kernel[3], 2, sizeof(uint), &startPos_pre);
	 //if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[3], 3, sizeof(float)*257, NULL);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[3], 4, sizeof(int), &N);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	 err = clSetKernelArg(kernel[3], 5, sizeof(int), &blks);
	 if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// time capsule
	int frame;

	//for(frame = (T-3) ; frame >= (T-3); frame--)
	for(frame = (T-3) ; frame >= 0; frame--)
	{
		startPos =  frame * N;
		startPos_pre = startPos + N;

		err |= clSetKernelArg(kernel[2], 4, sizeof(uint), &startPos);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err |= clSetKernelArg(kernel[2], 5, sizeof(uint), &startPos_pre);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[2], 2, NULL, global_2, local_2, 0, NULL, NULL );
		OCL_CHECK(err);

		// nomalize 
		err = clSetKernelArg(kernel[3], 1, sizeof(uint), &startPos);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clSetKernelArg(kernel[3], 2, sizeof(uint), &startPos_pre);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_3, &local_3, 0, NULL, NULL );
		OCL_CHECK(err);

	}


	clFinish(queue);

	clEnqueueReadBuffer(queue, beta_d, CL_TRUE, 0, sizeof(float)*N*T, beta, 0, NULL , &event[1]);


	//check 2d



	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	printf("oclTime = %lf (s)\n", gpuTime );

	// check
	//check_2d_f(beta,N,T);
	/*
	int j;
	for(i = 0 ; i < T; ++i){
		printf("row = %d\n", i);
		for( j = 0 ; j < N; ++j) {
			printf("%.4e\n",beta[i*N + j]);	
		
		}
		printf("\n");
	}
*/

	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(beta_d);
	clReleaseMemObject(dummy_d);


	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<numK;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<numE;++i){
		clReleaseEvent(event[i]);
	}


	free(beta);
	free(kernelSource);
	free(dummy);

	return;
}




